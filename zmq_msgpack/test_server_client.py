"""ZMQ+msgpack server-client 的完整测试套件。"""

import threading  # 多线程: server 在后台线程运行,主线程跑 client 测试
import time       # 时间工具: sleep 等待 server 启动完成
from typing import Any  # 类型注解

import numpy as np  # NumPy: 构造测试用的数组数据
import pytest        # pytest: Python 测试框架

from .policy import BasePolicy                                  # 导入策略基类
from .server_client import MsgSerializer, PolicyClient, PolicyServer  # 导入被测模块

# 全局端口计数器: 确保每个测试用不同端口,避免 "address already in use" 冲突
_port = 18700


def _next_port():
    """返回下一个可用端口号。每次调用递增,保证测试间端口不复用。"""
    global _port      # 声明修改的是模块级变量
    _port += 1        # 递增
    return _port      # 返回新端口


# =============================================================================
# 测试用的假策略: 不做真正的推理,只做简单变换方便验证 round-trip
# =============================================================================
class DummyPolicy(BasePolicy):
    """测试用策略: 将 state 乘以 2 作为 action 返回,用于验证数据完整性。"""

    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        action = {}                               # 初始化空 action 字典
        if "state" in observation:                # 如果 observation 包含 state
            action["joints"] = observation["state"]["joints"] * 2  # joints 乘以 2 → 方便断言验证
        if "video" in observation:                # 如果 observation 包含 video
            action["gripper"] = np.array([0.5], dtype=np.float32)  # 返回固定 gripper 值
        return action, {"policy": "dummy"}        # 返回 (action, info),info 标记策略名称

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        return {"reset": True}                    # 返回重置确认标志


# =============================================================================
# pytest Fixture: 每个测试自动获得独立的 server + client
# =============================================================================
@pytest.fixture()
def server_and_client():
    """启动 server(后台线程) + 创建 client,测试结束后自动清理。"""
    port = _next_port()                                       # 分配唯一端口
    server = PolicyServer(DummyPolicy(), host="127.0.0.1", port=port)  # 创建 server,绑定到 localhost
    thread = threading.Thread(target=server.run, daemon=True) # 创建守护线程运行 server
    thread.start()                                            # 启动线程 → server.run() 开始监听
    time.sleep(0.2)                                           # 等 200ms 确保 bind 完成

    client = PolicyClient(host="127.0.0.1", port=port, timeout_ms=5000)  # 创建 client 连接到 server
    yield server, client                                      # 将 server 和 client 交给测试用例使用

    # ---- 测试结束后自动执行 teardown ----
    server.close()          # 设置 running=False,server 线程将在 ≤500ms 内退出
    thread.join(timeout=2)  # 等待 server 线程结束,最多等 2 秒
    client.close()          # 释放 client 的 socket 和 ZMQ 上下文


# =============================================================================
# 第一组测试: MsgSerializer 序列化/反序列化的正确性
# =============================================================================
class TestMsgSerializer:
    def test_roundtrip_dict(self):
        """普通 dict 的 round-trip: 序列化后反序列化应得到相同数据。"""
        data = {"a": 1, "b": [2, 3], "c": "hello"}                  # 原始数据
        assert MsgSerializer.from_bytes(MsgSerializer.to_bytes(data)) == data  # 编码→解码→比较

    def test_roundtrip_numpy(self):
        """单个 numpy 数组的 round-trip: 内容和 dtype 都必须一致。"""
        arr = np.random.randn(2, 3).astype(np.float32)              # 随机 float32 数组
        result = MsgSerializer.from_bytes(MsgSerializer.to_bytes(arr))  # 编码→解码
        np.testing.assert_array_equal(arr, result)                   # 逐元素比较

    def test_roundtrip_nested_numpy(self):
        """嵌套 dict 中包含 numpy 数组: 模拟真实的 observation 结构。"""
        data = {
            "video": {"cam0": np.zeros((1, 2, 224, 224, 3), dtype=np.uint8)},  # 模拟相机图像
            "state": {"joints": np.array([1.0, 2.0, 3.0], dtype=np.float32)},  # 模拟关节状态
            "language": {"task": "pick up the cup"},                            # 语言指令(纯字符串)
        }
        result = MsgSerializer.from_bytes(MsgSerializer.to_bytes(data))     # 编码→解码
        np.testing.assert_array_equal(data["video"]["cam0"], result["video"]["cam0"])    # 验证图像
        np.testing.assert_array_equal(data["state"]["joints"], result["state"]["joints"])  # 验证关节
        assert result["language"]["task"] == "pick up the cup"                           # 验证文本

    def test_large_array(self):
        """大数组 round-trip: 模拟两帧 480x640 RGB 图像 (~1.8 MB)。"""
        arr = np.random.randint(0, 255, (2, 480, 640, 3), dtype=np.uint8)   # 约 1.8 MB
        result = MsgSerializer.from_bytes(MsgSerializer.to_bytes(arr))       # 编码→解码
        np.testing.assert_array_equal(arr, result)                           # 逐元素比较


# =============================================================================
# 第二组测试: Server-Client 集成测试(端到端通信)
# =============================================================================
class TestServerClient:
    def test_ping(self, server_and_client):
        """ping: 验证 client 能成功连接 server 并收到 ok 响应。"""
        _, client = server_and_client        # 解构 fixture,只需要 client
        assert client.ping() is True         # ping 应返回 True

    def test_get_action_with_state(self, server_and_client):
        """get_action (state): 发送 joints=[1,2,3],期望返回 [2,4,6](DummyPolicy 乘以 2)。"""
        _, client = server_and_client
        obs = {"state": {"joints": np.array([1.0, 2.0, 3.0], dtype=np.float32)}}  # 构造 observation
        action, info = client.get_action(obs)  # 通过 ZMQ 发送到 server 推理
        np.testing.assert_array_almost_equal(
            action["joints"], np.array([2.0, 4.0, 6.0], dtype=np.float32)  # 验证 joints 被乘以 2
        )
        assert info["policy"] == "dummy"     # 验证 info 中的策略标识

    def test_get_action_with_video(self, server_and_client):
        """get_action (video+state): 验证图像数据能正确传输,且返回 gripper 动作。"""
        _, client = server_and_client
        obs = {
            "video": {"cam0": np.zeros((1, 2, 224, 224, 3), dtype=np.uint8)},  # 模拟相机帧
            "state": {"joints": np.array([0.5], dtype=np.float32)},            # 模拟关节
        }
        action, _ = client.get_action(obs)            # 发送推理请求
        assert "joints" in action                      # 应包含 joints 动作
        assert "gripper" in action                     # 应包含 gripper 动作(因为 obs 有 video)
        np.testing.assert_array_almost_equal(action["gripper"], np.array([0.5]))  # gripper 固定为 0.5

    def test_reset(self, server_and_client):
        """reset: 验证策略重置功能正常。"""
        _, client = server_and_client
        result = client.reset()                # 发送 reset 请求
        assert result["reset"] is True         # DummyPolicy.reset 返回 {"reset": True}

    def test_custom_endpoint(self, server_and_client):
        """自定义 endpoint: 在 server 上动态注册 echo endpoint 并验证。"""
        server, client = server_and_client

        def echo_handler(msg=None):            # 定义一个简单的 echo 处理函数
            return {"echo": msg}               # 原样返回收到的 msg

        server.register_endpoint("echo", echo_handler)                      # 注册到 server
        result = client.call_endpoint("echo", {"msg": "hello"})             # client 调用
        assert result["echo"] == "hello"       # 验证 echo 返回了 "hello"

    def test_unknown_endpoint(self, server_and_client):
        """未知 endpoint: 调用不存在的 endpoint 应抛出 RuntimeError。"""
        _, client = server_and_client
        with pytest.raises(RuntimeError, match="Unknown endpoint"):         # 期望抛出含特定消息的异常
            client.call_endpoint("nonexistent", requires_input=False)       # 调用不存在的 endpoint

    def test_api_token_valid(self):
        """正确 token: 配置 token 后,携带正确 token 的请求应正常通过。"""
        port = _next_port()                    # 分配独立端口
        server = PolicyServer(DummyPolicy(), host="127.0.0.1", port=port, api_token="secret")  # server 设置 token
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()                         # 启动 server
        time.sleep(0.2)                        # 等待就绪

        client = PolicyClient(host="127.0.0.1", port=port, timeout_ms=5000, api_token="secret")  # client 使用相同 token
        assert client.ping() is True           # 应能正常 ping 通

        server.close()                         # 停止 server
        thread.join(timeout=2)                 # 等待线程退出
        client.close()                         # 释放 client 资源

    def test_api_token_invalid(self):
        """错误 token: 携带错误 token 的请求应被拒绝,返回 Unauthorized 错误。"""
        port = _next_port()                    # 分配独立端口
        server = PolicyServer(DummyPolicy(), host="127.0.0.1", port=port, api_token="secret")  # server 设置 token
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()                         # 启动 server
        time.sleep(0.2)                        # 等待就绪

        client = PolicyClient(host="127.0.0.1", port=port, timeout_ms=5000, api_token="wrong")  # client 使用错误 token
        with pytest.raises(RuntimeError, match="Unauthorized"):  # 期望抛出 Unauthorized 异常
            client.call_endpoint("ping", requires_input=False)   # 任何请求都应被拒绝

        server.close()                         # 停止 server
        thread.join(timeout=2)                 # 等待线程退出
        client.close()                         # 释放 client 资源


if __name__ == "__main__":
    pytest.main([__file__, "-v"])              # 直接运行此文件时启动 pytest
