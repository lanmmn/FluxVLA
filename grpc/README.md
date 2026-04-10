# VLA gRPC Service

基于 Protobuf + gRPC 的 VLA（Vision-Language-Action）推理通信模块，支持 Unary、双向流（Bidirectional Streaming）和服务端流（Server Streaming）三种通信模式。

## 目录结构

```
grpc/
├── vla_service.proto          # Protobuf 消息 + gRPC 服务定义
├── grpc_server.py             # 服务端（Unary / 双向流 / 状态流）
├── grpc_client.py             # 客户端（三种调用模式）
├── test_grpc.py               # 4 个端到端测试
├── requirements.txt           # 独立依赖（grpcio, grpcio-tools, protobuf, numpy）
├── run_test.sh                # 一键: 装依赖 → 编译 proto → 跑测试
├── vla_service_pb2.py         # (编译生成)
└── vla_service_pb2_grpc.py    # (编译生成)
```

## 快速开始

### 一键编译 + 测试

```bash
cd grpc
bash run_test.sh
```

### 手动步骤

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 编译 proto
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. vla_service.proto

# 3. 运行测试
python -m unittest test_grpc.TestVLAGRPC -v
```

## Proto 定义

### 消息类型

| 消息 | 说明 |
|------|------|
| `ActionVector` | 14 维动作向量（6 DOF + gripper，双臂共 14 维） |
| `InferRequest` | 推理请求：`images`（相机名 → JPEG 字节）、`action`（状态轨迹）、`state_delta`、`timestamp` |
| `InferResponse` | 推理响应：`action_list`（优化后动作）、`raw_action_list`（原始输出）、`infer_time`、`sequence_id` |
| `CameraConfig` | 相机配置：`camera_names`（服务端期望的相机名列表） |
| `CameraConfigRequest` | 获取相机配置的请求（空消息） |
| `ServerStatus` | 服务器状态：`status`、`uptime_s`、`total_requests`、`avg_infer_time` |
| `StatusRequest` | 状态订阅请求：`interval_s`（推送间隔秒数） |

### 服务接口

```protobuf
service VLAService {
  rpc Infer(InferRequest) returns (InferResponse);                    // Unary
  rpc InferStream(stream InferRequest) returns (stream InferResponse); // 双向流
  rpc StatusStream(StatusRequest) returns (stream ServerStatus);       // 服务端流
}
```

| RPC | 模式 | 用途 |
|-----|------|------|
| `Infer` | Unary | 单次请求-响应，兼容原有 HTTP 调用方式 |
| `InferStream` | Bidirectional Streaming | 客户端连续发送观测帧，服务端逐帧返回动作序列 |
| `StatusStream` | Server Streaming | 服务端按指定间隔推送运行状态（uptime、请求数、平均推理耗时） |
| `GetCameraConfig` | Unary | 查询服务端期望的相机名列表 |

## 使用示例

### 启动服务端

```bash
# 默认 3 相机（high, left_hand, right_hand）
python grpc_server.py --host 0.0.0.0 --port 50051 --workers 4

# 自定义相机列表（例如只有 2 个相机）
python grpc_server.py --cameras front back

# 单相机
python grpc_server.py --cameras high
```

默认使用 `MockInferPipeline`（随机动作），替换为实际推理只需传入自定义 `pipeline`：

```python
from grpc_server import serve

# 指定相机配置
server = serve(host="0.0.0.0", port=50051,
               pipeline=your_pipeline,
               camera_names=["front", "back"])
server.wait_for_termination()
```

自定义 pipeline 需实现 `infer(images, action, state_delta, timestamp) -> dict` 接口。

### 客户端调用

#### Unary（单次推理）

```python
from grpc_client import GRPCInferClient

with GRPCInferClient(host="localhost", port=50051) as client:
    result = client.infer(
        images={"high": jpeg_bytes, "left_hand": jpeg_bytes, "right_hand": jpeg_bytes},
        action=[list_of_14_floats],
        state_delta=0,
        timestamp=1234567890.0,
    )
    print(result["action_list"])   # [[float]*14, ...] x 50
    print(result["infer_time"])    # 推理耗时（秒）
```

#### 查询 & 同步相机配置

```python
with GRPCInferClient(host="localhost", port=50051) as client:
    # 查询服务端相机配置
    cameras = client.get_camera_config()
    print(cameras)  # e.g. ["front", "back"]

    # 同步到客户端（后续请求会自动校验）
    client.sync_camera_config()

    # 也可以在创建客户端时直接指定
    client2 = GRPCInferClient(camera_names=["front", "back"])
```

#### Bidirectional Streaming（双向流）

```python
def frame_generator():
    for _ in range(100):
        yield (images_dict, action_trajectory, state_delta, timestamp)

with GRPCInferClient(host="localhost", port=50051) as client:
    for response in client.infer_stream(frame_generator()):
        actions = response["action_list"]
        seq_id  = response["sequence_id"]
        # 实时处理每帧返回的动作序列
```

#### Server Streaming（状态监控）

```python
with GRPCInferClient(host="localhost", port=50051) as client:
    for status in client.status_stream(interval_s=1.0, max_updates=10):
        print(f"uptime={status['uptime_s']:.1f}s  requests={status['total_requests']}")
```

## 测试说明

`test_grpc.py` 包含 4 个端到端测试，内嵌启动 gRPC server（子线程），无需外部服务：

| 测试 | 说明 |
|------|------|
| `test_01_unary_infer` | 单次请求-响应，验证返回 50 组 14 维动作、infer_time > 0 |
| `test_02_bidirectional_stream` | 发送 5 帧观测，验证收到 5 组动作、sequence_id 递增 |
| `test_03_status_stream` | 接收 3 条状态推送，验证 uptime 递增 |
| `test_04_concurrent_unary` | 4 线程并发 Unary 请求，验证全部成功返回 |
| `test_05_get_camera_config` | 查询服务端相机配置，验证返回值 |
| `test_06_camera_mismatch_server` | 相机不匹配时服务端返回 INVALID_ARGUMENT |
| `test_07_client_camera_validation` | 客户端本地相机校验，发送前拦截错误 |
| `test_08_sync_camera_config` | 从服务端同步配置后客户端自动校验 |
| `TestCustomCameraConfig` | 自定义 2 相机配置的推理、查询、拒绝默认配置 |

## 集成到实际推理

将 `MockInferPipeline` 替换为真实推理 pipeline 即可，接口签名：

```python
class YourPipeline:
    def infer(self, images: dict[str, bytes], action: list[list[float]],
              state_delta: int, timestamp: float) -> dict:
        # images:      {"high": b"...", "left_hand": b"...", "right_hand": b"..."}
        # action:      [[float]*14, ...]  状态轨迹
        # state_delta: int                预测步数偏移
        # timestamp:   float              观测时间戳
        return {
            "action_list":     [[float]*14, ...],   # 优化后动作
            "raw_action_list": [[float]*14, ...],   # 原始模型输出
        }
```
