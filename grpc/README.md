# VLA gRPC Service

基于 Protobuf + gRPC 的 VLA（Vision-Language-Action）推理通信模块，支持 Unary、双向流（Bidirectional Streaming）和服务端流（Server Streaming）三种通信模式。

## 目录结构

```
grpc/
├── vla_service.proto          # Protobuf 消息 + gRPC 服务定义
├── grpc_server.py             # 服务端（Unary / 双向流 / 状态流 / TensorBatch）
├── grpc_client.py             # 客户端（三种调用模式）
├── remote_vla.py              # RemoteVLA 代理类（predict_action 接口）
├── serve_vla.py               # 一键启动真实 VLA 模型的 gRPC 推理服务
├── test_grpc.py               # 11 个端到端测试（基础 RPC + 多相机）
├── test_tensor_batch.py       # 8 个端到端测试（TensorBatch + RemoteVLA）
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
| `TensorBatchInferRequest` | 张量批次推理请求：`batch_data`（torch.save 序列化的 batch dict）、`unnorm_key` |
| `TensorBatchInferResponse` | 张量批次推理响应：`action_data`（torch.save 序列化的 action tensor）、`infer_time` |
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
| `TensorBatchInfer` | Unary | 张量批次推理：发送预处理好的 batch，返回 action tensor |

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

## 远程模型推理（TensorBatch 模式）

除了上述 JPEG 图像模式外，本模块还支持 **TensorBatch 模式**——将预处理后的 PyTorch tensor batch 直接发送到远程 GPU 服务器执行 `predict_action`，适用于 Eval Runner 和仿真环境场景。

```
Client (Eval / 仿真 / 机器人)              Server (GPU 推理)
┌────────────────────────────┐          ┌────────────────────────────┐
│  env.step() → obs          │          │  serve_vla.py              │
│  dataset(obs) → batch dict │          │  加载 config + checkpoint   │
│  RemoteVLA.predict_action  │  gRPC    │  VLAInferPipeline          │
│    torch.save(batch) ──────┼────────► │    torch.load → GPU        │
│                            │          │    vla.predict_action       │
│    actions ◄───────────────┼──────────┤    torch.save(actions)     │
│  denormalize + execute     │          │                            │
└────────────────────────────┘          └────────────────────────────┘
```

### 启动推理服务（serve_vla.py）

```bash
# 在 GPU 服务器上一键启动
python grpc/serve_vla.py \
  --config configs/pi05/pi05_paligemma_libero_10_full_finetune.py \
  --ckpt-path /path/to/checkpoint.pt \
  --host 0.0.0.0 --port 50051 --device cuda:0 --dtype bf16
```

参数说明：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--config` | mmengine 配置文件路径 | (必填) |
| `--ckpt-path` | 模型 checkpoint 路径 | (必填) |
| `--host` | 监听地址 | `0.0.0.0` |
| `--port` | gRPC 端口 | `50051` |
| `--device` | 推理设备 | `cuda:0` |
| `--dtype` | 混合精度类型 (`bf16`/`fp16`/`fp32`) | `bf16` |
| `--workers` | gRPC 线程池大小 | `4` |

### 在 Eval Config 中启用远程推理

在配置文件的 `eval` 段中添加 `remote_inference`：

```python
eval = dict(
    type='LiberoEvalRunner',
    remote_inference=dict(
        enabled=True,        # True 开启远程推理，False 走本地 GPU
        host='192.168.1.100', # 服务器 IP
        port=50051,
        timeout_s=30.0,
    ),
    # ... 其余配置不变
)
```

启用后，`LiberoEvalRunner` 会跳过本地模型构建，通过 `RemoteVLA` 代理发送推理请求到远程服务器。eval 循环代码无需任何修改。

```bash
# Client 端运行 eval（不需要 GPU 用于模型推理）
torchrun --nnodes 1 --nproc_per_node 2 scripts/eval.py \
  --config configs/pi05/pi05_paligemma_libero_10_full_finetune.py \
  --ckpt-path /path/to/checkpoint.pt
```

### 直接使用 RemoteVLA

也可以在自己的代码中直接使用 `RemoteVLA`：

```python
import sys
sys.path.insert(0, "grpc")
from remote_vla import RemoteVLA

# 创建代理（与真实 VLA 相同的接口）
vla = RemoteVLA(host="192.168.1.100", port=50051, device="cuda:0")

# 与本地 VLA 完全相同的调用方式
actions = vla.predict_action(
    images=images_tensor,       # (1, C, H, W)
    lang_tokens=tokens_tensor,  # (1, seq_len)
    states=states_tensor,       # (1, state_dim)
    img_masks=img_masks,
    lang_masks=lang_masks,
    unnorm_key="libero_10",
)
```

## 远程部署（JPEG 图像模式）

适用于实体机器人场景：**服务器（GPU）跑推理 Server**，**本地电脑（机器人控制端）跑 Client**。

```
本地电脑 (机器人控制)                    远程服务器 (GPU 推理)
┌──────────────────────┐              ┌──────────────────────┐
│  采集相机图像 (JPEG)   │   gRPC      │  grpc_server.py      │
│  grpc_client.py      │ ──────────► │  + 真实推理 Pipeline   │
│  发送 images+action   │  网络       │  返回 action_list     │
│  执行返回的动作        │ ◄────────── │                      │
└──────────────────────┘              └──────────────────────┘
```

### 服务器端（GPU 机器）

```bash
# 1. 安装依赖
pip install grpcio grpcio-tools numpy

# 2. 编译 proto（如果 pb2 文件不是最新的）
cd grpc/
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. vla_service.proto

# 3. 启动服务，指定相机配置
python grpc_server.py --host 0.0.0.0 --port 50051 --cameras high left_hand right_hand
```

接入真实推理 pipeline：

```python
# server_main.py
from grpc_server import serve

class MyRealPipeline:
    def __init__(self):
        self.model = load_your_model()

    def infer(self, images: dict[str, bytes], action: list[list[float]],
              state_delta: int, timestamp: float) -> dict:
        # images: {"high": jpeg_bytes, "left_hand": jpeg_bytes, ...}
        result = self.model.predict(images, action, state_delta)
        return {
            "action_list":     result.optimized,   # [[float]*14] * 50
            "raw_action_list": result.raw,          # [[float]*14] * 50
        }

server = serve(
    host="0.0.0.0",
    port=50051,
    pipeline=MyRealPipeline(),
    camera_names=["high", "left_hand", "right_hand"],
)
server.wait_for_termination()
```

### 本地电脑（机器人控制端）

只需将以下 4 个文件拷贝到本地：

- `grpc_client.py`
- `vla_service_pb2.py`
- `vla_service_pb2_grpc.py`
- `requirements.txt`

```bash
pip install grpcio protobuf
```

#### 单次推理（Unary）

```python
import time
import cv2
from grpc_client import GRPCInferClient

SERVER_IP = "192.168.1.100"  # 替换为你的服务器 IP

with GRPCInferClient(host=SERVER_IP, port=50051) as client:
    # 连接后先同步相机配置，后续自动校验
    cameras = client.sync_camera_config()
    print(f"服务端要求的相机: {cameras}")  # ['high', 'left_hand', 'right_hand']

    # 采集图像 → JPEG 编码
    images = {}
    for cam_name in cameras:
        frame = capture_from_camera(cam_name)            # 你的采集函数
        _, jpeg = cv2.imencode(".jpg", frame)
        images[cam_name] = jpeg.tobytes()

    # 推理
    action_trajectory = [list_of_14_floats]  # 当前状态
    result = client.infer(images, action_trajectory,
                          state_delta=0, timestamp=time.time())

    print(result["action_list"])   # 50 步动作序列
    print(result["infer_time"])    # 服务端推理耗时
```

#### 实时控制循环（Bidirectional Streaming）

```python
import cv2
from grpc_client import GRPCInferClient

with GRPCInferClient(host="192.168.1.100", port=50051) as client:
    cameras = client.sync_camera_config()

    def observation_stream():
        """持续采集观测帧"""
        while robot.is_running():
            images = {}
            for cam_name in cameras:
                frame = capture_from_camera(cam_name)
                _, jpeg = cv2.imencode(".jpg", frame)
                images[cam_name] = jpeg.tobytes()
            yield (images, current_action_trajectory, state_delta, time.time())

    # 流式推理：每发一帧观测，立刻收到一组动作
    for response in client.infer_stream(observation_stream()):
        actions = response["action_list"]     # 50x14
        seq_id  = response["sequence_id"]
        robot.execute(actions[0])             # 执行第一步动作
```

### 不同相机数量配置示例

```bash
# 单相机
python grpc_server.py --cameras high

# 双相机
python grpc_server.py --cameras front back

# 四相机
python grpc_server.py --cameras high left_hand right_hand wrist
```

客户端不需要硬编码相机名，连接后调用 `sync_camera_config()` 即可自动适配服务端配置。如果发送的相机名与服务端不一致，客户端本地和服务端都会报错拦截。

### 网络注意事项

| 项目 | 说明 |
|------|------|
| 防火墙 | 服务器需开放对应端口（默认 50051） |
| 带宽 | 3 路 480p JPEG ≈ 150-300KB/帧，30Hz ≈ 5-9MB/s，千兆内网足够 |
| 延迟 | 同内网通常 < 1ms 网络延迟，推理延迟取决于模型 |
| 跨网 | 如果不在同一内网，需要端口转发或 VPN |

## 测试说明

共 19 个端到端测试，内嵌启动 gRPC server（子线程），无需外部服务：

```bash
cd grpc
python -m unittest test_grpc test_tensor_batch -v
```

### test_grpc.py（11 个测试 — 基础 RPC + 多相机）

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

### test_tensor_batch.py（8 个测试 — TensorBatch + RemoteVLA）

| 测试 | 说明 |
|------|------|
| `TestTensorSerialization` | batch dict 序列化 round-trip，验证 dtype/shape/bfloat16 保持 |
| `test_01_basic_infer` | TensorBatchInfer RPC 端到端，验证 action shape |
| `test_02_unnorm_key_passthrough` | unnorm_key 透传到 server pipeline |
| `test_03_invalid_batch_data` | 损坏数据返回 INVALID_ARGUMENT |
| `test_01_predict_action` (RemoteVLA) | RemoteVLA.predict_action 端到端验证 |
| `test_02_duck_typing` (RemoteVLA) | eval()/cuda()/freeze_*/norm_stats 兼容性 |
| `TestNoVLAPipeline` | 无 VLA pipeline 时返回 UNIMPLEMENTED |

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
