"""轻量级 ZMQ + msgpack 策略推理 server-client。

从 NVIDIA Isaac GR00T 提取,去掉了模型依赖。
通过 ZeroMQ REQ/REP 模式在 TCP 上暴露任意 BasePolicy。
"""

import io                              # 内存字节流,用于 numpy 数组的序列化缓冲区
from dataclasses import dataclass      # 数据类装饰器,简化 EndpointHandler 的定义
from typing import Any, Callable       # Any: 任意类型; Callable: 可调用对象类型

import msgpack                         # MessagePack: 高效二进制序列化库,比 JSON 更快更小
import numpy as np                     # NumPy: 数组运算库,机器人数据的核心载体
import zmq                             # PyZMQ: ZeroMQ 的 Python 绑定,提供高性能消息队列

from .policy import BasePolicy         # 从同包导入策略抽象基类


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
        """msgpack 反序列化钩子: 检测并还原自定义类型。"""
        if not isinstance(obj, dict):   # 非 dict 直接返回(str, int, list 等原生类型)
            return obj
        if "__ndarray__" in obj:        # 发现 ndarray 标记 → 这是我们自己编码的 numpy 数组
            return np.load(             # 从 .npy 格式的字节流中重建 numpy 数组
                io.BytesIO(obj["data"]),  # 把 bytes 包装成文件流供 np.load 读取
                allow_pickle=False        # 禁止 pickle 反序列化,防止任意代码执行漏洞
            )
        return obj                      # 普通 dict,原样返回

    @staticmethod
    def _encode(obj):
        """msgpack 序列化钩子: 将 msgpack 不认识的类型转换为可序列化的 dict。"""
        if isinstance(obj, np.ndarray):            # 遇到 numpy 数组
            buf = io.BytesIO()                     # 创建内存缓冲区
            np.save(buf, obj, allow_pickle=False)  # 将数组以 .npy 二进制格式写入缓冲区
            return {                               # 返回带标记的 dict,供 _decode 识别
                "__ndarray__": True,               # 标记位: 告诉解码器这是一个 numpy 数组
                "data": buf.getvalue()             # 实际数据: .npy 格式的 bytes
            }
        raise TypeError(f"Cannot serialize {type(obj)}")  # 其他未知类型,抛异常


# =============================================================================
# Endpoint 处理器: 将一个 handler 函数和它是否需要输入数据的属性绑定在一起
# =============================================================================
@dataclass
class EndpointHandler:
    handler: Callable          # 处理函数,如 policy.get_action 或 自定义函数
    requires_input: bool = True  # True=调用时需传入参数(如 observation); False=无参调用(如 ping)


# =============================================================================
# Server 端: 监听 TCP 端口,接收请求,路由到对应 endpoint,返回结果
# =============================================================================
class PolicyServer:
    """ZMQ REP 服务器,通过 TCP 暴露一个 BasePolicy 的所有接口。"""

    def __init__(
        self,
        policy: BasePolicy,            # 要服务的策略实例(实现了 get_action/reset)
        host: str = "*",               # 绑定地址, "*" 表示监听所有网卡, "127.0.0.1" 仅本机
        port: int = 5555,              # 监听端口号
        api_token: str | None = None,  # 可选的 API 令牌,用于简单的身份验证
    ):
        self.policy = policy           # 保存策略引用,后续请求会调用它的方法
        self.running = True            # 服务器运行标志,设为 False 时主循环退出
        self.context = zmq.Context()   # ZMQ 上下文: 管理所有 socket 的生命周期,一个进程通常一个
        self.socket = self.context.socket(zmq.REP)  # 创建 REP(Reply) socket: 收一条请求,回一条响应
        self.socket.bind(f"tcp://{host}:{port}")     # 绑定到指定地址和端口,开始监听
        self._endpoints: dict[str, EndpointHandler] = {}  # endpoint 名称 → 处理器的映射表
        self.api_token = api_token     # 保存 token,每次请求都会校验(如果设置了的话)

        # 注册默认的内置 endpoint
        self.register_endpoint("ping", self._handle_ping, requires_input=False)       # 健康检查
        self.register_endpoint("kill", self._kill_server, requires_input=False)        # 远程关闭服务器
        self.register_endpoint("get_action", self.policy.get_action)                   # 核心: 推理动作
        self.register_endpoint("reset", self.policy.reset)                             # 重置策略状态

    def register_endpoint(self, name: str, handler: Callable, requires_input: bool = True):
        """注册自定义 endpoint,允许在 server 上挂载任意函数。"""
        self._endpoints[name] = EndpointHandler(handler, requires_input)

    def _handle_ping(self) -> dict:
        """内置 ping endpoint: 返回服务器状态,用于 client 端判断连接是否正常。"""
        return {"status": "ok", "message": "Server is running"}

    def _kill_server(self):
        """内置 kill endpoint: 设置 running=False,主循环将在下次迭代时退出。"""
        self.running = False
        return {"status": "ok", "message": "Server shutting down"}

    def _validate_token(self, request: dict) -> bool:
        """校验请求中的 API token。未配置 token 时直接放行。"""
        if self.api_token is None:     # 服务器没设置 token → 不需要认证
            return True
        return request.get("api_token") == self.api_token  # 比较 token 是否匹配

    def run(self):
        """主事件循环: 不断接收请求 → 路由 → 执行 → 返回结果。"""
        addr = self.socket.getsockopt_string(zmq.LAST_ENDPOINT)  # 获取实际绑定的地址(含端口)
        print(f"Server is ready and listening on {addr}")
        poller = zmq.Poller()                    # 创建 poller: 用于非阻塞地检测 socket 是否有数据到达
        poller.register(self.socket, zmq.POLLIN) # 注册 socket 到 poller,监听"可读"事件
        while self.running:                      # 主循环,直到 running 被设为 False
            try:
                socks = dict(poller.poll(timeout=500))  # 最多等 500ms,没有数据就返回空 dict
                if self.socket not in socks:     # 500ms 内没收到请求 → 回到循环顶部检查 running
                    continue                     # 这使得 close() 设置 running=False 后最多 500ms 退出

                message = self.socket.recv()                 # 接收原始字节(此时一定有数据,不会阻塞)
                request = MsgSerializer.from_bytes(message)  # 反序列化为 Python dict

                if not self._validate_token(request):        # 校验 API token
                    self.socket.send(                        # token 无效 → 返回 401 错误
                        MsgSerializer.to_bytes({"error": "Unauthorized: Invalid API token"})
                    )
                    continue                                 # 跳过后续处理,等下一个请求

                endpoint = request.get("endpoint", "get_action")  # 读取请求的 endpoint 名称,默认 get_action
                if endpoint not in self._endpoints:               # endpoint 不存在 → 抛异常
                    raise ValueError(f"Unknown endpoint: {endpoint}")

                handler = self._endpoints[endpoint]  # 查找对应的 EndpointHandler
                result = (
                    handler.handler(**request.get("data", {}))  # 需要输入 → 解包 data 字典作为关键字参数传入
                    if handler.requires_input
                    else handler.handler()                      # 不需要输入 → 直接无参调用
                )
                self.socket.send(MsgSerializer.to_bytes(result))  # 将结果序列化后发回给 client
            except Exception as e:
                print(f"Error in server: {e}")                          # 打印错误日志(不崩溃,继续服务)
                self.socket.send(MsgSerializer.to_bytes({"error": str(e)}))  # 将错误信息返回给 client
        # 循环退出后,清理资源
        self.socket.setsockopt(zmq.LINGER, 0)  # LINGER=0: 立即丢弃未发送的消息,不等待
        self.socket.close()                     # 关闭 socket,释放端口
        self.context.term()                     # 终止 ZMQ 上下文,释放所有底层资源

    def close(self):
        """从外部(另一个线程)优雅地停止服务器。run() 会在下次 poll 超时后退出。"""
        self.running = False


# =============================================================================
# Client 端: 连接远程 Server,发送请求,接收响应
# =============================================================================
class PolicyClient(BasePolicy):
    """ZMQ REQ 客户端,与远程 PolicyServer 通信。继承 BasePolicy 使其接口透明。"""
    # 继承 BasePolicy 的意义: 调用方不需要知道推理是本地还是远程的,接口完全一致

    def __init__(
        self,
        host: str = "localhost",       # 服务器地址
        port: int = 5555,              # 服务器端口
        timeout_ms: int = 15000,       # 接收超时(毫秒),超时后抛 ZMQError
        api_token: str | None = None,  # API 令牌,需与 server 端一致
    ):
        super().__init__(strict=False)  # 调用 BasePolicy.__init__,关闭 strict 模式(校验在 server 端做)
        self.context = zmq.Context()    # 创建 ZMQ 上下文
        self.host = host                # 保存连接参数,重连时需要
        self.port = port
        self.timeout_ms = timeout_ms
        self.api_token = api_token
        self._init_socket()             # 初始化并连接 socket

    def _init_socket(self):
        """初始化(或重新初始化) REQ socket 并连接到 server。"""
        if hasattr(self, "socket") and not self.socket.closed:  # 如果已有旧 socket 且未关闭
            self.socket.close()                                  # 先关掉旧的,避免资源泄漏
        self.socket = self.context.socket(zmq.REQ)               # 创建 REQ(Request) socket
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)   # 设置接收超时,防止永久阻塞
        self.socket.connect(f"tcp://{self.host}:{self.port}")    # 连接到 server(非阻塞,ZMQ 会自动重试)

    def call_endpoint(
        self, endpoint: str, data: dict | None = None, requires_input: bool = True
    ) -> Any:
        """通用 endpoint 调用: 构造请求 → 序列化发送 → 等待响应 → 反序列化返回。"""
        request: dict = {"endpoint": endpoint}   # 构造请求字典,指定要调用的 endpoint
        if requires_input:                       # 如果 endpoint 需要输入数据
            request["data"] = data               # 将数据挂在 "data" 字段下
        if self.api_token:                       # 如果配置了 API token
            request["api_token"] = self.api_token  # 附加到请求中供 server 校验

        self.socket.send(MsgSerializer.to_bytes(request))  # 序列化并发送请求
        message = self.socket.recv()                        # 阻塞等待响应(受 RCVTIMEO 超时保护)
        response = MsgSerializer.from_bytes(message)        # 反序列化响应

        if isinstance(response, dict) and "error" in response:  # 检查 server 是否返回了错误
            raise RuntimeError(f"Server error: {response['error']}")  # 将 server 错误转为本地异常
        return response                                     # 返回正常结果

    def ping(self) -> bool:
        """健康检查: 尝试 ping server,成功返回 True,失败时重建 socket 并返回 False。"""
        try:
            self.call_endpoint("ping", requires_input=False)  # 发送 ping 请求
            return True                                        # 收到响应 → 连接正常
        except (zmq.error.ZMQError, RuntimeError):            # 超时或其他错误
            self._init_socket()                                # 重建 socket(ZMQ REQ socket 出错后状态会混乱)
            return False                                       # 返回连接失败

    def kill_server(self):
        """远程关闭 server: 发送 kill 指令。"""
        self.call_endpoint("kill", requires_input=False)

    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """实现 BasePolicy._get_action: 将推理请求转发到远程 server。"""
        response = self.call_endpoint(
            "get_action", {"observation": observation, "options": options}
        )
        return tuple(response)  # msgpack 反序列化后是 list,转回 tuple 匹配接口签名 (action, info)

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        """实现 BasePolicy.reset: 将重置请求转发到远程 server。"""
        return self.call_endpoint("reset", {"options": options})

    def close(self):
        """释放客户端资源: 关闭 socket 和 ZMQ 上下文。"""
        if not self.socket.closed:                       # 避免重复关闭
            self.socket.setsockopt(zmq.LINGER, 0)        # LINGER=0: 不等待未发送的消息,立即关闭
            self.socket.close()                          # 关闭 socket
        self.context.term()                              # 终止 ZMQ 上下文
