"""轻量级 ZMQ + msgpack 策略推理 server-client。

从 NVIDIA Isaac GR00T 提取,去掉了模型依赖。
通过 ZeroMQ REQ/REP 模式在 TCP 上暴露任意 BasePolicy。
"""

import io  # 内存字节流,用于 numpy 数组的序列化缓冲区
from dataclasses import dataclass  # 数据类装饰器,简化 EndpointHandler 的定义
from typing import Any, Callable  # Any: 任意类型; Callable: 可调用对象类型

import cv2  # OpenCV: 图像编解码 (JPEG 压缩)
import msgpack  # MessagePack: 高效二进制序列化库,比 JSON 更快更小
import numpy as np  # NumPy: 数组运算库,机器人数据的核心载体
import zmq  # PyZMQ: ZeroMQ 的 Python 绑定,提供高性能消息队列

from .policy import BasePolicy  # 从同包导入策略抽象基类
from .serializers import (FORMAT_PROTOBUF, decode_predict_request,
                          encode_predict_response, detect_format)


# =============================================================================
# 序列化层: 负责 Python 对象 <-> 二进制字节 的互相转换
# =============================================================================
class MsgSerializer:
    """通过 msgpack 序列化/反序列化 Python 对象,内置 numpy 数组的自定义编解码。"""

    @staticmethod
    def to_bytes(data: Any) -> bytes:
        """将任意 Python 对象序列化为 bytes,遇到 numpy 数组时走自定义编码器。"""
        return msgpack.packb(data, default=MsgSerializer._encode)
        # default=_encode: 当 msgpack 遇到不认识的类型时,调用 _encode 尝试转换

    @staticmethod
    def from_bytes(data: bytes) -> Any:
        """将 bytes 反序列化回 Python 对象,遇到 ndarray 标记时还原为 numpy 数组。"""
        return msgpack.unpackb(data, object_hook=MsgSerializer._decode)
        # object_hook=_decode: 每个反序列化出的 dict 都会经过 _decode 检查是否需要特殊处理

    @staticmethod
    def _decode(obj):
        """msgpack 反序列化钩子: 检测并还原自定义类型。
        注意: object_hook 保证 obj 一定是 dict,无需 isinstance 检查。"""
        if '__ndarray__' in obj:  # 发现 ndarray 标记 → 这是我们自己编码的 numpy 数组
            return np.load(  # 从 .npy 格式的字节流中重建 numpy 数组
                io.BytesIO(obj['data']),  # 把 bytes 包装成文件流供 np.load 读取
                allow_pickle=False  # 禁止 pickle 反序列化,防止任意代码执行漏洞
            )
        return obj  # 普通 dict,原样返回

    @staticmethod
    def _encode(obj):
        """msgpack 序列化钩子: 将 msgpack 不认识的类型转换为可序列化的 dict。"""
        if isinstance(obj, np.ndarray):  # 遇到 numpy 数组
            buf = io.BytesIO()  # 创建内存缓冲区
            np.save(buf, obj, allow_pickle=False)  # 将数组以 .npy 二进制格式写入缓冲区
            return {  # 返回带标记的 dict,供 _decode 识别
                '__ndarray__': True,  # 标记位: 告诉解码器这是一个 numpy 数组
                'data': buf.getvalue()  # 实际数据: .npy 格式的 bytes
            }
        raise TypeError(f'Cannot serialize {type(obj)}')  # 其他未知类型,抛异常


# =============================================================================
# 原始观测序列化层: JPEG 压缩图像 + msgpack 其余字段
# =============================================================================
class ObsSerializer:
    """序列化/反序列化 raw observation dict。

    图像 (HWC uint8) 用 JPEG 压缩,其余 numpy 数组用 npy 格式,字符串直接传。
    相比 torch.save tensor batch,传输量减少 10-30x。
    """

    JPEG_QUALITY = 95

    # 白名单: 只对已知 RGB 相机 key 做 JPEG 有损压缩,
    # 避免深度图/分割掩码等 uint8 HWC 数据被 JPEG 引入不可逆伪影
    JPEG_KEYS = {
        'cam_high', 'cam_left_wrist', 'cam_right_wrist', 'agentview_image',
        'robot0_eye_in_hand_image'
    }

    @staticmethod
    def to_bytes(obs: dict, compress: bool = True) -> bytes:
        encoded = {}
        for k, v in obs.items():
            if (compress and isinstance(v, np.ndarray) and v.ndim == 3
                    and v.dtype == np.uint8 and k in ObsSerializer.JPEG_KEYS):
                _, jpg = cv2.imencode(
                    '.jpg', v,
                    [cv2.IMWRITE_JPEG_QUALITY, ObsSerializer.JPEG_QUALITY])
                encoded[k] = {'__jpeg__': True, 'data': jpg.tobytes()}
            elif isinstance(v, np.ndarray):
                buf = io.BytesIO()
                np.save(buf, v, allow_pickle=False)
                encoded[k] = {'__ndarray__': True, 'data': buf.getvalue()}
            else:
                encoded[k] = v
        return msgpack.packb(encoded)

    @staticmethod
    def from_bytes(data: bytes) -> dict:
        raw = msgpack.unpackb(data, raw=False)  # raw=False: key 统一为 str
        obs = {}
        for k, v in raw.items():
            if isinstance(v, dict):
                if '__jpeg__' in v:
                    obs[k] = cv2.imdecode(
                        np.frombuffer(v['data'], np.uint8), cv2.IMREAD_COLOR)
                elif '__ndarray__' in v:
                    obs[k] = np.load(io.BytesIO(v['data']), allow_pickle=False)
                else:
                    obs[k] = v
            elif isinstance(v, bytes):
                obs[k] = v.decode()
            else:
                obs[k] = v
        return obs


# =============================================================================
# Endpoint 处理器: 将一个 handler 函数和它是否需要输入数据的属性绑定在一起
# =============================================================================
@dataclass
class EndpointHandler:
    handler: Callable  # 处理函数,如 policy.get_action 或 自定义函数
    requires_input: bool = True  # True=调用时需传入参数(如 observation); False=无参调用(如 ping)


# =============================================================================
# Server 端: 监听 TCP 端口,接收请求,路由到对应 endpoint,返回结果
# =============================================================================
class PolicyServer:
    """ZMQ REP 服务器,通过 TCP 暴露一个 BasePolicy 的所有接口。"""

    def __init__(
            self,
            policy: BasePolicy,  # 要服务的策略实例(实现了 get_action/reset)
            host: str = '*',  # 绑定地址, "*" 表示监听所有网卡, "127.0.0.1" 仅本机
            port: int = 5555,  # 监听端口号
            api_token: str | None = None,  # 可选的 API 令牌,用于简单的身份验证
    ):
        self.policy = policy  # 保存策略引用,后续请求会调用它的方法
        self.running = True  # 服务器运行标志,设为 False 时主循环退出
        self.context = zmq.Context()  # ZMQ 上下文: 管理所有 socket 的生命周期,一个进程通常一个
        self.socket = self.context.socket(
            zmq.REP)  # 创建 REP(Reply) socket: 收一条请求,回一条响应
        self.socket.bind(f'tcp://{host}:{port}')  # 绑定到指定地址和端口,开始监听
        self._endpoints: dict[str,
                              EndpointHandler] = {}  # endpoint 名称 → 处理器的映射表
        self.api_token = api_token  # 保存 token,每次请求都会校验(如果设置了的话)

        # 注册默认的内置 endpoint
        self.register_endpoint(
            'ping', self._handle_ping, requires_input=False)  # 健康检查
        self.register_endpoint(
            'kill', self._kill_server, requires_input=False)  # 远程关闭服务器
        self.register_endpoint('get_action',
                               self.policy.get_action)  # 核心: 推理动作
        self.register_endpoint('reset', self.policy.reset)  # 重置策略状态

    def register_endpoint(self,
                          name: str,
                          handler: Callable,
                          requires_input: bool = True):
        """注册自定义 endpoint,允许在 server 上挂载任意函数。"""
        self._endpoints[name] = EndpointHandler(handler, requires_input)

    def _handle_ping(self) -> dict:
        """内置 ping endpoint: 返回服务器状态,用于 client 端判断连接是否正常。"""
        return {'status': 'ok', 'message': 'Server is running'}

    def _kill_server(self):
        """内置 kill endpoint: 设置 running=False,主循环将在下次迭代时退出。"""
        self.running = False
        return {'status': 'ok', 'message': 'Server shutting down'}

    def _validate_token(self, request: dict) -> bool:
        """校验请求中的 API token。未配置 token 时直接放行。"""
        if self.api_token is None:  # 服务器没设置 token → 不需要认证
            return True
        return request.get('api_token') == self.api_token  # 比较 token 是否匹配

    def run(self):
        """主事件循环: 不断接收请求 → 路由 → 执行 → 返回结果。

        支持两种序列化格式 (由 client 选择, server 自动检测):
        - msgpack (默认): 第一字节 != 0x01
        - protobuf: 第一字节 == 0x01, 仅用于 predict_action 热路径
        """
        addr = self.socket.getsockopt_string(zmq.LAST_ENDPOINT)
        print(f'Server is ready and listening on {addr}')
        poller = zmq.Poller()
        poller.register(self.socket, zmq.POLLIN)
        while self.running:
            try:
                socks = dict(poller.poll(timeout=500))
                if self.socket not in socks:
                    continue

                message = self.socket.recv()

                # --- Protobuf fast-path for predict_action ---
                if detect_format(message) == FORMAT_PROTOBUF:
                    self._handle_protobuf_predict(message)
                    continue

                # --- Msgpack path (default, all endpoints) ---
                request = MsgSerializer.from_bytes(message)

                if not self._validate_token(request):
                    self.socket.send(
                        MsgSerializer.to_bytes(
                            {'error': 'Unauthorized: Invalid API token'}))
                    continue

                endpoint = request.get('endpoint', 'get_action')
                if endpoint not in self._endpoints:
                    raise ValueError(f'Unknown endpoint: {endpoint}')

                handler = self._endpoints[endpoint]
                result = (
                    handler.handler(**request.get('data', {}))
                    if handler.requires_input else handler.handler())
                self.socket.send(MsgSerializer.to_bytes(result))
            except Exception as e:
                print(f'Error in server: {e}')
                self.socket.send(
                    MsgSerializer.to_bytes({'error': str(e)}))

        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.close()
        self.context.term()

    def _handle_protobuf_predict(self, message: bytes):
        """Handle a protobuf-encoded predict_action request."""
        try:
            _, obs, unnorm_key = decode_predict_request(message)
            handler = self._endpoints.get('predict_action')
            if handler is None:
                resp = encode_predict_response(
                    b'', 0.0, FORMAT_PROTOBUF,
                    error='predict_action endpoint not registered')
            else:
                result = handler.handler(
                    obs_data=None, unnorm_key=unnorm_key,
                    _obs_dict=obs, _wire_format=FORMAT_PROTOBUF)
                action_data = result.get('action_data', b'')
                infer_time = result.get('infer_time', 0.0)
                error = result.get('error', '')
                resp = encode_predict_response(
                    action_data, infer_time, FORMAT_PROTOBUF, error=error)
            self.socket.send(resp)
        except Exception as e:
            print(f'Error in protobuf handler: {e}')
            resp = encode_predict_response(
                b'', 0.0, FORMAT_PROTOBUF, error=str(e))
            self.socket.send(resp)

    def close(self):
        """从外部(另一个线程)优雅地停止服务器。run() 会在下次 poll 超时后退出。"""
        self.running = False
