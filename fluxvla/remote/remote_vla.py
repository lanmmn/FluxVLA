"""RemoteVLAZmq: 本地 VLA 模型的远程替身（drop-in replacement），
将推理请求委托给远程 ZMQ 服务器。

对外暴露与真实 VLA 模型完全相同的 ``predict_action`` 接口,
调用方无需感知推理发生在本地还是远端。

Usage::

    from fluxvla.remote import RemoteVLAZmq

    vla = RemoteVLAZmq(host="192.168.1.100", port=5555, device="cuda:0")
    actions = vla.predict_action(**batch)   # 与本地模型接口完全一致
"""
from __future__ import annotations       # 允许在类型注解中使用 str | None 等新语法

import io                                # 内存字节流,用于 torch.load 从 bytes 中反序列化
import threading                         # 线程锁,保护 ZMQ REQ socket 的并发安全
import time                              # 高精度计时器,用于性能 profiling

import numpy as np                       # NumPy: 数组运算,raw observation 的载体
import torch                             # PyTorch: tensor 操作,action 反序列化后在 GPU 上
import zmq                               # PyZMQ: ZeroMQ 的 Python 绑定,提供 REQ/REP 通信
import msgpack                           # MessagePack: 高效二进制序列化,比 JSON 更快更紧凑

from .server_client import ObsSerializer  # 从同包导入观测序列化器: 图像 JPEG 压缩 + msgpack


class RemoteVLAZmq:
    """客户端代理类,镜像真实 VLA 模型的 ``predict_action`` 接口,
    但将实际推理转发到远程 ZMQ 服务器。

    通过 duck typing 实现与本地模型的接口兼容:
    - eval() / cuda() 等方法直接返回 self
    - freeze_* 属性静默吸收赋值
    - norm_stats 属性可读写

    这使得 LiberoEvalRunner / BaseInferenceRunner 的 run_setup()
    不需要区分本地还是远程模型。
    """

    def __init__(self,
                 host: str = "localhost",      # 远程服务器地址
                 port: int = 5555,             # 远程服务器端口
                 timeout_s: float = 30.0,      # 单次请求超时秒数,超时后抛 ZMQError
                 device: str = "cuda:0",       # 反序列化 action 时放到哪个设备
                 enable_profiling: bool = True):  # 是否启用性能统计
        self._host = host                      # 保存 host,用于日志输出
        self._port = port                      # 保存 port,用于日志输出
        self._address = f"tcp://{host}:{port}"  # 拼接 ZMQ 连接地址,格式: tcp://host:port
        self._timeout_ms = int(timeout_s * 1000)  # 将秒转为毫秒,ZMQ 超时参数以毫秒为单位
        self._device = torch.device(device)    # 解析设备字符串为 torch.device 对象
        self._enable_profiling = enable_profiling  # 保存 profiling 开关

        # --- ZMQ socket 初始化 ---
        self._context = zmq.Context()          # 创建 ZMQ 上下文: 管理所有 socket 生命周期
        self._socket = self._context.socket(zmq.REQ)  # 创建 REQ (Request) socket: 发一条请求,等一条响应
        self._socket.setsockopt(zmq.RCVTIMEO, self._timeout_ms)  # 设置接收超时,防止 recv() 永久阻塞
        self._socket.setsockopt(zmq.SNDTIMEO, self._timeout_ms)  # 设置发送超时,防止 send() 在网络不通时卡住
        self._socket.connect(self._address)    # 连接到远程 server (非阻塞,ZMQ 底层会自动重连)

        # --- 线程安全: ZMQ REQ socket 不支持多线程并发,需要加锁保护 ---
        self._lock = threading.Lock()
        self._closed = False                   # close() 幂等标记,防止 __del__ 重复释放

        # --- Duck typing 属性: 让 LiberoEvalRunner.run_setup() 不报错 ---
        self.norm_stats = {}                   # 本地不使用,仅满足 runner 对 vla.norm_stats 的访问
        self.freeze_vision_backbone = True     # 静默吸收 freeze 设置,远程模型不需要本地 freeze
        self.freeze_llm_backbone = True        # 同上
        self.freeze_projector = True           # 同上
        self.freeze_vlm_backbone = True        # 同上

        # --- Per-call profiling: 每次 predict_action 调用后可读取 ---
        self._last_profile = {}                # 最近一次调用的详细耗时,外部 profiler 读取此字段

        # --- 全局累积 profiling: 每 50 次调用打印平均值 ---
        self._call_count = 0                   # 总调用次数
        self._t_serialize = 0.0                # 累积序列化耗时(秒)
        self._t_zmq = 0.0                      # 累积 ZMQ 往返耗时(秒),包含网络 + server 推理
        self._t_deserialize = 0.0              # 累积反序列化耗时(秒)
        self._t_total = 0.0                    # 累积总耗时(秒)
        self._t_server_infer = 0.0             # 累积 server 报告的推理耗时(秒)
        self._payload_bytes = 0                # 累积发送的 payload 字节数

    # ------------------------------------------------------------------
    # Duck-typing helpers: 让 run_setup() 无需区分本地/远程模型
    # ------------------------------------------------------------------
    def eval(self):
        """模拟 model.eval(): 远程模型无需切换模式,直接返回 self。"""
        return self

    def cuda(self, device=None):
        """模拟 model.cuda(device_id): 远程模型不需要移到 GPU,
        但记录 device 用于后续 action 反序列化时的 map_location。"""
        if device is not None:
            self._device = torch.device(
                f"cuda:{device}" if isinstance(device, int) else device)
            # 如果传的是整数 (如 0),包装为 "cuda:0"; 否则直接用字符串 (如 "cuda:1")
        return self                            # 返回 self 支持链式调用: vla.cuda(0).eval()

    def __setattr__(self, name, value):
        """拦截所有属性赋值: freeze_* 属性静默吸收 (不做任何事),
        其余属性正常设置。

        这使得 run_setup() 中 `self.vla.freeze_vision_backbone = True`
        不会因为 RemoteVLAZmq 没有这些属性而报错。"""
        if name.startswith("freeze_"):         # 以 "freeze_" 开头的属性
            object.__setattr__(self, name, value)  # 直接存储,不做其他处理
            return                             # 提前返回
        object.__setattr__(self, name, value)  # 其余属性正常处理

    # ------------------------------------------------------------------
    # Core inference: 核心推理接口
    # ------------------------------------------------------------------
    def predict_action(self, **kwargs):
        """将 raw observation 序列化,发送给远程 server,返回反归一化后的 action。

        Server 负责完整的推理流水线:
        1. 图像预处理 (resize, normalize)
        2. Tokenization (task description → token ids)
        3. 状态归一化 (proprio state → normalized)
        4. 模型推理 (GPU forward pass)
        5. Action 反归一化 (normalized → robot command space)

        Client 只需传 raw observation:
        - numpy images (uint8, HWC 格式)
        - numpy arrays (关节状态等)
        - strings (task description)
        - torch.Tensor (自动转为 numpy)

        Returns:
            torch.Tensor: 反归一化后的 action tensor,已在 self._device 上,
                shape 为 (1, n_action_steps, action_dim)。
        """
        t_total_start = time.perf_counter()    # 记录总计时起点 (perf_counter 精度约 ns 级)

        unnorm_key = kwargs.pop("unnorm_key", "")  # 取出 unnorm_key (用于 server 端反归一化),从 kwargs 中移除
        # pop 而非 get: unnorm_key 不是 observation 的一部分,不应序列化发送

        # --- Phase 1: 序列化 raw observation ---
        t0 = time.perf_counter()               # 序列化计时起点
        obs = {}                               # 构建纯 numpy/string 的 observation dict
        for k, v in kwargs.items():            # 遍历所有 keyword arguments
            if isinstance(v, torch.Tensor):    # 如果是 torch.Tensor
                obs[k] = v.cpu().numpy()       # 先移到 CPU,再转为 numpy (ZMQ 不能直接传 tensor)
            else:
                obs[k] = v                     # numpy array / string / int 等直接保留
        payload = ObsSerializer.to_bytes(obs)  # 序列化: 图像 → JPEG 压缩 (~10x 压缩),其余 → npy 格式
        t_serialize = time.perf_counter() - t0  # 序列化耗时

        # --- Phase 2: ZMQ 请求-响应 (加锁保护 REQ socket 线程安全) ---
        t1 = time.perf_counter()               # ZMQ 往返计时起点
        request = msgpack.packb({              # 构造请求消息并用 msgpack 序列化
            "endpoint": "predict_action",      # 指定要调用 server 的哪个 endpoint
            "data": {                          # endpoint 的参数
                "obs_data": payload,           # 序列化后的 raw observation (bytes)
                "unnorm_key": str(unnorm_key),  # 反归一化 key,如 "libero_10"
            },
        })
        with self._lock:
            self._socket.send(request)             # 发送请求 (受 SNDTIMEO 超时保护)
            raw_response = self._socket.recv()     # 阻塞等待响应 (受 RCVTIMEO 超时保护)
        response = msgpack.unpackb(raw_response, raw=False)  # raw=False: key 统一为 str
        t_zmq = time.perf_counter() - t1       # ZMQ 往返耗时 (包含: 网络传输 + server 全部处理时间)

        # --- 错误检查 ---
        if isinstance(response, dict) and "error" in response:  # server 返回了错误
            raise RuntimeError(f"ZMQ server error: {response['error']}")
            # 将 server 端的异常信息传播到 client,方便调试

        # --- Phase 3: 反序列化 action tensor ---
        t2 = time.perf_counter()               # 反序列化计时起点
        action_buf = io.BytesIO(response["action_data"])
        # io.BytesIO 将 bytes 包装为文件流,供 np.load 读取
        arr = np.load(action_buf, allow_pickle=False)
        actions = torch.from_numpy(arr.copy()).to(self._device)
        # .copy() 确保内存连续; .to(device) 直接放到目标 GPU
        t_deserialize = time.perf_counter() - t2  # 反序列化耗时

        t_total = time.perf_counter() - t_total_start  # 总耗时

        # --- 提取 server 报告的推理耗时 ---
        server_infer = response.get("infer_time", 0.0)
        # server_infer 包含 server 端的 preprocess + model inference 时间
        # 用于计算 network_ms = zmq_roundtrip - server_infer (纯网络传输时间)

        # --- 记录本次调用的 profiling 数据 ---
        self._last_profile = {
            "serialize_ms": t_serialize * 1000,      # client 序列化耗时 (ms)
            "zmq_roundtrip_ms": t_zmq * 1000,        # ZMQ 往返总耗时 (ms),含网络+server
            "server_infer_ms": server_infer * 1000,   # server 报告的推理耗时 (ms)
            "network_ms": (t_zmq - server_infer) * 1000,  # 纯网络传输耗时 (ms) = 往返 - server推理
            "deserialize_ms": t_deserialize * 1000,   # client 反序列化耗时 (ms)
            "total_ms": t_total * 1000,               # 本次调用总耗时 (ms)
            "payload_kb": len(payload) / 1024,        # 发送的 payload 大小 (KB)
        }
        # _last_profile 供 LiberoEvalRunner 在每个 step 后读取,用于构建逐阶段 profiling 报告

        # --- 全局累积统计 ---
        if self._enable_profiling:
            self._call_count += 1              # 调用计数 +1
            self._t_serialize += t_serialize   # 累加序列化耗时
            self._t_zmq += t_zmq               # 累加 ZMQ 往返耗时
            self._t_deserialize += t_deserialize  # 累加反序列化耗时
            self._t_total += t_total           # 累加总耗时
            self._t_server_infer += server_infer  # 累加 server 推理耗时
            self._payload_bytes += len(payload)  # 累加 payload 字节数

            if self._call_count % 50 == 0:     # 每 50 次调用打印一次平均统计
                n = self._call_count
                print(f"[RemoteVLAZmq profiling] calls={n}  "
                      f"avg_total={self._t_total/n*1000:.1f}ms  "
                      f"avg_serialize={self._t_serialize/n*1000:.1f}ms  "
                      f"avg_zmq_roundtrip={self._t_zmq/n*1000:.1f}ms  "
                      f"avg_server_infer={self._t_server_infer/n*1000:.1f}ms  "
                      f"avg_deserialize={self._t_deserialize/n*1000:.1f}ms  "
                      f"avg_payload={self._payload_bytes/n/1024:.0f}KB",
                      flush=True)              # flush=True: 立即刷新缓冲区,防止日志延迟输出

        return actions                         # 返回 action tensor,已在目标设备上,已反归一化

    def ping(self) -> bool:
        """健康检查: 向 server 发送 ping 请求,成功返回 True,失败返回 False。

        失败时不抛异常,静默返回 False,调用方可据此判断 server 是否可用。"""
        try:
            request = msgpack.packb({"endpoint": "ping"})  # 构造 ping 请求
            with self._lock:
                self._socket.send(request)         # 发送
                raw = self._socket.recv()          # 等待响应 (受超时保护)
            resp = msgpack.unpackb(raw, raw=False)  # raw=False: key 统一为 str
            return resp.get("status") == "ok"
        except zmq.error.ZMQError:             # 超时或网络错误
            return False                       # 返回不可用

    # ------------------------------------------------------------------
    # Lifecycle: 资源管理
    # ------------------------------------------------------------------
    def close(self):
        """释放 ZMQ 资源: 关闭 socket 和 context。

        LINGER=0 表示不等待未发送的消息,立即关闭。
        这防止 close() 在 server 不可达时挂起。
        _closed 标记确保幂等: close() + __del__ 不会 double-close context。"""
        if getattr(self, '_closed', False):    # 已经关闭过,直接返回
            return
        self._closed = True
        if hasattr(self, '_socket') and not self._socket.closed:  # socket 存在且未关闭
            self._socket.setsockopt(zmq.LINGER, 0)  # 设置 LINGER=0: 丢弃所有待发送消息
            self._socket.close()               # 关闭 socket,释放文件描述符
        if hasattr(self, '_context'):          # context 存在
            self._context.term()               # 终止 ZMQ 上下文,释放所有底层资源

    def __del__(self):
        """析构函数: 对象被垃圾回收时自动释放资源。

        用 try/except 保护,因为 __del__ 中抛异常会被 Python 忽略并打印警告,
        且在解释器退出时全局变量可能已被清理 (zmq 可能为 None)。"""
        try:
            self.close()
        except Exception:
            pass                               # 静默忽略所有异常
