# FluxVLA Remote Inference

基于 ZeroMQ 的远程 VLA 推理框架，专为**真机部署**设计。模型推理、数据预处理和 action 反归一化全部在 GPU 服务器上完成，机器人端只需发送 raw observation（JPEG 图像 + 关节状态），接收可直接执行的机器人指令。

支持两种序列化格式：
- **msgpack**（默认）：灵活、零 schema、Python 快速原型
- **protobuf**：schema 驱动、跨语言（C++/Rust 端侧）

支持的 Runner：
- `URInferenceRunner` — UR3/UR5 机械臂
- `AlohaInferenceRunner` — ALOHA 双臂机器人

## 架构

```
┌───────────────────────┐   ZMQ (TCP)   ┌────────────────────────────────┐
│  机器人工控机 (Client)  │  ~100KB/req   │  GPU 服务器 (Server)            │
│                       │ ────────────► │                                │
│  ROS + 相机 + 关节     │               │  ObsDeserialize (JPEG→numpy)   │
│  (numpy img + qpos    │               │  ↓                             │
│   + task string)      │               │  dataset(obs) → tensor batch   │
│                       │               │  ↓                             │
│  RemoteVLAZmq         │               │  VLAInferPipeline (GPU)        │
│   .predict_action()   │  denormed     │  ↓                             │
│                       │  action       │  denormalize → serialize       │
│  直接执行关节指令      │ ◄──────────── │                                │
│                       │               │                                │
│  无需 GPU / 模型权重   │               │  模型 + 权重 + 预处理 + 反归一化 │
└───────────────────────┘               └────────────────────────────────┘
```

## 快速开始（真机部署）

### Step 1: 在 GPU 服务器上启动推理服务

```bash
# UR3/UR5 单臂
CUDA_VISIBLE_DEVICES=0 python -m fluxvla.remote.serve \
    --config configs/pi05/pi05_paligemma_ur3_full_finetune.py \
    --ckpt-path /data/checkpoints/ur3/step-003000.pt \
    --host 0.0.0.0 --port 5555 --dtype bf16

# ALOHA 双臂
CUDA_VISIBLE_DEVICES=0 python -m fluxvla.remote.serve \
    --config configs/pi05/pi05_paligemma_aloha_full_finetune.py \
    --ckpt-path /data/checkpoints/aloha/step-005000.pt \
    --host 0.0.0.0 --port 5555 --dtype bf16
```

启动成功后会打印：

```
[serve] Building VLA model from config ...
[serve] Loading checkpoint: /data/checkpoints/ur3/step-003000.pt
[serve] Loaded norm_stats from /data/checkpoints/ur3/dataset_statistics.json
[serve] Dataset pipeline built from cfg.inference.dataset
[serve] Denormalize action transform built from cfg.inference.denormalize_action
[serve] Model on cuda:0 (bf16), ready to serve.
Server is ready and listening on tcp://0.0.0.0:5555
```

**Server 参数：**

| 参数            | 默认值    | 说明                                                         |
| --------------- | --------- | ------------------------------------------------------------ |
| `--config`      | (必填)    | mmengine config 文件路径                                     |
| `--ckpt-path`   | (必填)    | 模型 checkpoint 路径（同目录需有 `dataset_statistics.json`） |
| `--host`        | `0.0.0.0` | 监听地址（`0.0.0.0` 监听所有网卡）                           |
| `--port`        | `5555`    | 监听端口                                                     |
| `--device`      | `cuda:0`  | 推理设备                                                     |
| `--dtype`       | `bf16`    | 精度类型（bf16/fp16/fp32）                                   |
| `--dataset-key` | 自动检测  | 从哪个 config key 读 dataset（`inference` 或 `eval`）        |

### Step 2: 配置机器人端 Client

在 config 的 `inference` 段中添加 `offload_inference`：

#### UR3/UR5

```python
inference = dict(
    type='URInferenceRunner',
    offload_inference=dict(
        enabled=True,
        host='192.168.1.100',   # ← GPU 服务器的局域网 IP
        port=5555,
        timeout_s=30.0,
    ),
    seed=7,
    dataset=dict(...),          # config 中保留，但 client 端不会加载
    denormalize_action=dict(...),
    action_chunk=50,
    operator=dict(type='UROperator', ...),
)
```

#### ALOHA 双臂

```python
inference = dict(
    type='AlohaInferenceRunner',
    offload_inference=dict(
        enabled=True,
        host='192.168.1.100',
        port=5555,
        timeout_s=30.0,
    ),
    seed=7,
    dataset=dict(...),
    denormalize_action=dict(...),
    action_chunk=50,
    operator=dict(type='AlohaOperator', ...),
)
```

`offload_inference` 字段说明：

| 字段        | 类型  | 说明                       |
| ----------- | ----- | -------------------------- |
| `enabled`   | bool  | 是否启用远程推理           |
| `host`      | str   | GPU 服务器地址             |
| `port`      | int   | 服务器端口（默认 5555）    |
| `timeout_s` | float | 单次请求超时秒数           |

### Step 3: 启动机器人端

```bash
# 方式 1: config 中已写好 offload_inference
python scripts/inference.py \
    --config configs/pi05/pi05_paligemma_ur3_full_finetune.py

# 方式 2: 命令行覆盖（无需修改 config 文件）
python scripts/inference.py \
    --config configs/pi05/pi05_paligemma_ur3_full_finetune.py \
    --cfg-options inference.offload_inference.enabled=True \
                  inference.offload_inference.host=192.168.1.100 \
                  inference.offload_inference.port=5555
```

Offload 模式下 client **不加载模型权重、不加载 tokenizer、不构建 dataset pipeline**，启动时间 < 5 秒。

## 完整真机示例

### UR3 单臂

```bash
# ===== 机器 A: GPU 服务器 (192.168.1.100, 有 GPU) =====
CUDA_VISIBLE_DEVICES=0 python -m fluxvla.remote.serve \
    --config configs/pi05/pi05_paligemma_ur3_full_finetune.py \
    --ckpt-path /data/checkpoints/ur3/step-003000.pt \
    --host 0.0.0.0 --port 5555 --dtype bf16
# 等待 "Server is ready and listening on tcp://0.0.0.0:5555"

# ===== 机器 B: 工控机 (192.168.1.50, 无 GPU, 连接 ROS) =====
python scripts/inference.py \
    --config configs/pi05/pi05_paligemma_ur3_full_finetune.py \
    --cfg-options inference.offload_inference.enabled=True \
                  inference.offload_inference.host=192.168.1.100 \
                  inference.offload_inference.port=5555
```

### ALOHA 双臂

```bash
# ===== 机器 A: GPU 服务器 (192.168.1.100) =====
CUDA_VISIBLE_DEVICES=0 python -m fluxvla.remote.serve \
    --config configs/pi05/pi05_paligemma_aloha_full_finetune.py \
    --ckpt-path /data/checkpoints/aloha/step-005000.pt \
    --host 0.0.0.0 --port 5555 --dtype bf16

# ===== 机器 B: 工控机 (192.168.1.50, 无 GPU) =====
python scripts/inference.py \
    --config configs/pi05/pi05_paligemma_aloha_full_finetune.py \
    --cfg-options inference.offload_inference.enabled=True \
                  inference.offload_inference.host=192.168.1.100 \
                  inference.offload_inference.port=5555
```

## 自定义 Client（不使用 Runner）

如果有自己的控制循环，可以直接使用 `RemoteVLAZmq`：

```python
from fluxvla.remote import RemoteVLAZmq
import numpy as np

# 连接 GPU 服务器（默认 msgpack 序列化）
vla = RemoteVLAZmq(host='192.168.1.100', port=5555, timeout_s=30.0)

# 或使用 protobuf 序列化（跨语言客户端推荐）
# vla = RemoteVLAZmq(host='192.168.1.100', port=5555, serializer='protobuf')

# 健康检查
assert vla.ping(), "Server unreachable!"

# 推理循环
while True:
    obs = {
        'cam_high': cam_high_img,              # np.uint8, (H, W, 3), BGR
        'cam_left_wrist': cam_left_img,        # np.uint8, (H, W, 3), BGR
        'qpos': np.array(joint_positions),     # np.float32, (n_joints,)
        'task_description': 'pick up the cup', # str
    }

    actions = vla.predict_action(**obs, unnorm_key='private')
    # actions: torch.Tensor, shape (1, action_chunk, action_dim)
    # 已反归一化，可直接执行

    robot.execute(actions[0, 0].cpu().numpy())

vla.close()
```

**关键点**：
- 图像直接传 camera 原始输出（`uint8`, BGR），server 负责 resize + normalize
- `qpos` 传 `float32` numpy array，server 负责 state normalize
- `task_description` 传字符串，server 负责 tokenize
- 返回的 `actions` 已经是反归一化后的机器人指令
- `unnorm_key` 告诉 server 用哪组统计量做反归一化

## Client 端对比

|                              | 本地推理 | Offload 远程推理             |
| ---------------------------- | -------- | ---------------------------- |
| 加载模型权重                 | Yes      | **No**                       |
| 加载 tokenizer               | Yes      | **No**                       |
| 构建 dataset pipeline        | Yes      | **No**                       |
| 加载 dataset_statistics.json | Yes      | **No**                       |
| Action 反归一化              | Client   | **Server**                   |
| 需要 `--ckpt-path`           | Yes      | **No**                       |
| 需要 GPU                     | Yes      | **No**                       |
| Client 依赖                  | 全部     | `zmq msgpack numpy opencv torch` |
| 启动时间                     | ~60s     | **< 5s**                     |

## 序列化格式

| 特性 | msgpack（默认） | protobuf |
|------|----------------|----------|
| Schema | 无，Python dict 直接序列化 | `.proto` 文件定义 |
| 跨语言 | 有库但靠约定 | 编译时类型安全 |
| 适用场景 | Python-Python，快速迭代 | C++/Rust 端侧对接 |
| 切换方式 | `RemoteVLAZmq(serializer='msgpack')` | `RemoteVLAZmq(serializer='protobuf')` |

Server 端**自动检测**客户端格式，无需配置。Proto 定义在 `proto/vla_service.proto`。

## Profiling

客户端每 50 次调用打印性能统计：

```
[RemoteVLAZmq profiling] calls=50  avg_total=95.2ms  avg_serialize=1.2ms
  avg_zmq_roundtrip=92.5ms  avg_server_infer=85.3ms  avg_deserialize=1.5ms
  avg_payload=85KB
```

Server 端每 50 次请求打印：

```
[ZMQ VLAServer] req=50  deserialize=0.8ms  preprocess=5.2ms  infer=82.1ms
  serialize=0.3ms  avg_infer=82.5ms
```

## 模块结构

```
fluxvla/remote/
    __init__.py           # 包导出
    policy.py             # BasePolicy 抽象基类
    server_client.py      # ZMQ REQ/REP 通信框架 + MsgSerializer + ObsSerializer
    serializers.py        # Protobuf 序列化器 + 格式检测
    vla_server.py         # Server 端推理封装 (VLAInferPipeline, VLAPolicy)
    remote_vla.py         # Client 端代理 (RemoteVLAZmq)
    serve.py              # Server 启动入口 (python -m fluxvla.remote.serve)
    proto/
        vla_service.proto # Protobuf schema 定义
        vla_service_pb2.py# 生成的 Python 代码
```

## 注意事项

1. **Server config 必须完整**：`--config` 中需包含 `dataset` 和 `denormalize_action` 配置，同时 `--ckpt-path` 同级目录需有 `dataset_statistics.json`
2. **图像 key 白名单**：`cam_high`, `cam_left_wrist`, `cam_right_wrist`, `agentview_image`, `robot0_eye_in_hand_image` 会 JPEG 压缩传输，其他 `uint8` 图像用无损 npy 格式
3. **线程安全**：`RemoteVLAZmq` 内部有锁保护，可从多线程安全调用
4. **防火墙**：确保 ZMQ 端口（默认 5555）对 Client 可达，`telnet <server_ip> 5555` 可验证
5. **网络要求**：Server 和 Client 需在同一局域网内，延迟 < 10ms 为佳
