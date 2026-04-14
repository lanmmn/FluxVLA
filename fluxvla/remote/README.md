# FluxVLA Remote Inference (ZMQ)

基于 ZeroMQ + msgpack 的远程 VLA 推理框架。模型推理、数据预处理和 action 反归一化全部部署在 GPU 服务器上，客户端只需发送 raw observation（JPEG 图像 + 关节状态），接收可直接执行的机器人指令。客户端无需加载 tokenizer、image processor、模型权重或 dataset_statistics.json。

支持的 Runner：

- `AlohaInferenceRunner` — ALOHA 双臂机器人
- `URInferenceRunner` — UR3/UR5 机械臂
- `AlohaRTCInferenceRunner` — ALOHA + RTC 实时控制
- `LiberoEvalRunner` — LIBERO 仿真评估

## 架构

```
┌───────────────────────┐   JPEG+msgpack   ┌────────────────────────────────┐
│  Client (机器人端)     │    ~100KB/req     │  Server (GPU)                  │
│                       │ ────────────────► │                                │
│  ROS/Env → raw obs    │                   │  ObsSerializer.from_bytes()    │
│  (numpy img + qpos    │                   │  ↓                             │
│   + task string)      │                   │  dataset(obs) → tensor batch   │
│                       │                   │  ↓                             │
│  RemoteVLAZmq         │                   │  VLAInferPipeline              │
│   .predict_action()   │                   │   .predict_action(**batch)     │
│                       │  denormed action  │  ↓                             │
│  直接执行              │ ◄──────────────── │  denormalize → serialize       │
└───────────────────────┘                   └────────────────────────────────┘
```

**预处理 + 后处理全在 server 端执行**：tokenize、image resize/normalize、state normalize、model inference、action denormalize 全部在 GPU 服务器上完成。客户端收到的是可直接执行的机器人指令，启动快，无依赖。

## 真机部署指南

### 网络拓扑

```
┌──────────────────────┐          TCP/ZMQ          ┌──────────────────────┐
│  机器人端 (Client)    │  ◄───── 局域网 ────────►  │  GPU 服务器 (Server)  │
│                      │     ~100KB/req             │                      │
│  - NUC / Jetson /    │     ~100ms RTT             │  - A100 / 4090 / ... │
│    树莓派 / 工控机    │                            │  - 模型 + 权重       │
│  - ROS + 相机 + 关节  │                            │  - 预处理 + 反归一化  │
│  - 无需 GPU          │                            │                      │
└──────────────────────┘                            └──────────────────────┘
```

**前提要求**：
- Server 和 Client 在同一局域网内，能 TCP 互通（默认端口 5555）
- Server 有 GPU，已安装 FluxVLA 全部依赖
- Client 只需 `pip install zmq msgpack numpy opencv-python torch`（无需模型相关依赖）

### Step 1: 启动 Server（GPU 机器）

```bash
# ALOHA 双臂
CUDA_VISIBLE_DEVICES=0 python zmq_msgpack/serve_vla_zmq.py \
    --config configs/pi05/pi05_paligemma_aloha_full_finetune.py \
    --ckpt-path /data/checkpoints/aloha/step-005000.pt \
    --host 0.0.0.0 --port 5555 --device cuda:0 --dtype bf16

# UR3 单臂
CUDA_VISIBLE_DEVICES=0 python zmq_msgpack/serve_vla_zmq.py \
    --config configs/pi05/pi05_paligemma_ur3_full_finetune.py \
    --ckpt-path /data/checkpoints/ur3/step-003000.pt \
    --host 0.0.0.0 --port 5555

# LIBERO 仿真评估
CUDA_VISIBLE_DEVICES=1 python zmq_msgpack/serve_vla_zmq.py \
    --config configs/pi05/pi05_paligemma_libero_10_full_finetune.py \
    --ckpt-path /data/checkpoints/libero/step-003172.pt \
    --host 0.0.0.0 --port 5555 --dataset-key eval
```

Server 启动后会依次打印：

```
[serve_vla_zmq] Building VLA model from config ...
[serve_vla_zmq] Loading checkpoint: /data/checkpoints/aloha/step-005000.pt
[serve_vla_zmq] Loaded norm_stats from /data/checkpoints/aloha/dataset_statistics.json
[serve_vla_zmq] Dataset pipeline built from cfg.inference.dataset
[serve_vla_zmq] Denormalize action transform built from cfg.inference.denormalize_action
[serve_vla_zmq] Model on cuda:0 (bf16), ready to serve.
Server is ready and listening on tcp://0.0.0.0:5555
```

看到 `Server is ready` 即可启动 Client。

**Server 参数：**

| 参数            | 默认值    | 说明                                                  |
| --------------- | --------- | ----------------------------------------------------- |
| `--config`      | (必填)    | mmengine config 文件路径                              |
| `--ckpt-path`   | (必填)    | 模型 checkpoint 路径（同目录需有 `dataset_statistics.json`）|
| `--host`        | `0.0.0.0` | 监听地址（`0.0.0.0` 监听所有网卡）                    |
| `--port`        | `5555`    | 监听端口                                              |
| `--device`      | `cuda:0`  | 推理设备                                              |
| `--dtype`       | `bf16`    | 精度类型（bf16/fp16/fp32）                            |
| `--dataset-key` | 自动检测  | 从哪个 config key 读 dataset（`inference` 或 `eval`） |

### Step 2: 配置 Client（机器人端）

在 config 文件的 `inference`（真机）或 `eval`（仿真）字段中添加 `offload_inference`：

#### ALOHA 双臂

```python
inference = dict(
    type='AlohaInferenceRunner',
    offload_inference=dict(
        enabled=True,
        backend='zmq',
        host='192.168.1.100',   # ← 改为 GPU 服务器的局域网 IP
        port=5555,
        timeout_s=30.0,
    ),
    seed=7,
    dataset=dict(...),          # config 中保留，但 client 端不会加载
    denormalize_action=dict(
        type='DenormalizePrivateAction',
        norm_type='min_max',
        action_dim=14,
    ),
    action_chunk=50,
    operator=dict(type='AlohaOperator', ...),
)
```

#### UR3/UR5 单臂

```python
inference = dict(
    type='URInferenceRunner',
    offload_inference=dict(
        enabled=True,
        backend='zmq',
        host='192.168.1.100',   # ← GPU 服务器 IP
        port=5555,
        timeout_s=30.0,
    ),
    seed=7,
    dataset=dict(...),
    denormalize_action=dict(
        type='DenormalizePrivateAction',
        norm_type='min_max',
        action_dim=7,
    ),
    action_chunk=50,
    operator=dict(type='UROperator', ...),
)
```

#### LIBERO 仿真评估

```python
eval = dict(
    type='LiberoEvalRunner',
    offload_inference=dict(
        enabled=True,
        backend='zmq',
        host='192.168.1.100',
        port=5555,
        timeout_s=30.0,
    ),
    task_suite_name='libero_10',
    model_family='pi0',
    eval_chunk_size=10,
    # ...
)
```

`offload_inference` 字段说明：

```python
offload_inference=dict(
    enabled=False,       # bool: 是否启用远程推理
    backend='zmq',       # str: 通信后端 ('zmq' 或 'grpc')
    host='localhost',    # str: GPU 服务器地址
    port=5555,           # int: 服务器端口
    timeout_s=30.0,      # float: 单次请求超时秒数
)
```

### Step 3: 启动 Client（机器人端）

Offload 模式下客户端**不需要 `--ckpt-path`**，不加载模型权重、不加载 tokenizer、不加载 dataset_statistics.json。

#### ALOHA / UR 真机推理

```bash
# 方式 1: config 中已写好 offload_inference
python scripts/inference.py \
    --config configs/pi05/pi05_paligemma_aloha_full_finetune.py

# 方式 2: 命令行覆盖 config（无需修改 config 文件）
python scripts/inference.py \
    --config configs/pi05/pi05_paligemma_aloha_full_finetune.py \
    --cfg-options inference.offload_inference.enabled=True \
                  inference.offload_inference.host=192.168.1.100 \
                  inference.offload_inference.port=5555
```

#### LIBERO 仿真评估

```bash
# LIBERO eval 需要 --ckpt-path 用于写日志文件（不加载权重）
CUDA_VISIBLE_DEVICES=0 torchrun --nnodes 1 --nproc_per_node 1 scripts/eval.py \
    --config configs/pi05/pi05_paligemma_libero_10_full_finetune.py \
    --ckpt-path /path/to/checkpoint.pt \
    --cfg-options eval.offload_inference.enabled=True \
                  eval.offload_inference.host=192.168.1.100 \
                  eval.offload_inference.port=5555
```

### 完整真机示例（ALOHA）

```bash
# ===== 机器 A: GPU 服务器 (192.168.1.100) =====
CUDA_VISIBLE_DEVICES=0 python zmq_msgpack/serve_vla_zmq.py \
    --config configs/pi05/pi05_paligemma_aloha_full_finetune.py \
    --ckpt-path /data/checkpoints/aloha/step-005000.pt \
    --host 0.0.0.0 --port 5555 --dtype bf16
# 等待打印 "Server is ready and listening on tcp://0.0.0.0:5555"

# ===== 机器 B: 机器人工控机 (192.168.1.50, 无 GPU) =====
python scripts/inference.py \
    --config configs/pi05/pi05_paligemma_aloha_full_finetune.py \
    --cfg-options inference.offload_inference.enabled=True \
                  inference.offload_inference.host=192.168.1.100 \
                  inference.offload_inference.port=5555
# Client 启动 < 5 秒，开始采集图像 → 发送 → 接收动作 → 执行
```

### 自定义 Client（不使用 Runner）

如果你有自己的控制循环，可以直接使用 `RemoteVLAZmq`：

```python
from fluxvla.remote import RemoteVLAZmq
import numpy as np

# 连接 GPU 服务器
vla = RemoteVLAZmq(host='192.168.1.100', port=5555, timeout_s=30.0)

# 健康检查
assert vla.ping(), "Server unreachable!"

# 推理循环
while True:
    # 从相机/ROS 获取观测（raw numpy，无需预处理）
    obs = {
        'cam_high': cam_high_img,              # np.uint8, (H, W, 3), BGR
        'cam_left_wrist': cam_left_img,        # np.uint8, (H, W, 3), BGR
        'qpos': np.array(joint_positions),     # np.float32, (n_joints,)
        'task_description': 'pick up the cup', # str
    }

    # 发送 raw obs → 接收 denormalized action（可直接执行）
    actions = vla.predict_action(**obs, unnorm_key='aloha')
    # actions: torch.Tensor, shape (1, action_chunk, action_dim)
    # 已经是机器人关节空间的真实指令，无需任何后处理

    # 执行第一帧动作
    robot.execute(actions[0, 0].cpu().numpy())

# 释放
vla.close()
```

**关键点**：
- 图像直接传 camera 原始输出（`uint8`, BGR），server 负责 resize + normalize
- `qpos` 传 `float32` numpy array，server 负责 state normalize
- `task_description` 传字符串，server 负责 tokenize
- 返回的 `actions` 已经是反归一化后的机器人指令
- `unnorm_key` 告诉 server 用哪组统计量做反归一化

## Client 端启动对比

|                              | 本地推理 | offload（当前） |
| ---------------------------- | -------- | --------------- |
| 加载模型权重                 | Yes      | **No**          |
| 加载 tokenizer               | Yes      | **No**          |
| 加载 image processor         | Yes      | **No**          |
| 构建 dataset pipeline        | Yes      | **No**          |
| 加载 dataset_statistics.json | Yes      | **No**          |
| Action 反归一化              | Client   | **Server**      |
| 需要 `--ckpt-path`          | Yes      | **No**（仅 LIBERO eval 日志路径需要）|
| 需要 GPU                    | Yes      | **No**          |
| Client 依赖                 | 全部     | `zmq msgpack numpy opencv torch` |
| 启动时间                     | ~60s     | **<5s**         |

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

## 测试

### 集成测试（mock model）

```bash
python -m fluxvla.remote.test_remote_inference
```

测试覆盖：ObsSerializer 序列化/反序列化、Aloha 3-cam 端到端、UR3 2-cam 端到端、Libero tuple-dataset 端到端。

### 端到端真实模型测试（Libero 10）

已验证 PI0.5 + Libero 10 全流程：

```bash
# 1. 启动 Server（GPU 1）
CUDA_VISIBLE_DEVICES=1 python zmq_msgpack/serve_vla_zmq.py \
    --config configs/pi05/pi05_paligemma_libero_10_full_finetune.py \
    --ckpt-path /path/to/step-003172-epoch-02-loss=0.0734.pt \
    --host 0.0.0.0 --port 5555 --device cuda:0 --dtype bf16 --dataset-key eval

# 2. 启动 Client（GPU 0）
CUDA_VISIBLE_DEVICES=0 torchrun --nnodes 1 --nproc_per_node 1 scripts/eval.py \
    --config configs/pi05/pi05_paligemma_libero_10_full_finetune.py \
    --ckpt-path /path/to/step-003172-epoch-02-loss=0.0734.pt \
    --cfg-options eval.offload_inference.enabled=True \
                  eval.offload_inference.host=127.0.0.1 \
                  eval.offload_inference.port=5555
```

实测结果（libero_10, 10 tasks x 1 trial, 251 steps）：

| 阶段 | Client 端 | Server 端 |
|------|-----------|-----------|
| data_preprocess | 0.0ms | 20.1ms |
| serialize | 1.3ms | — |
| network roundtrip | 3.9ms | — |
| model inference | — | 340.8ms |
| deserialize | 0.9ms | — |
| env_step | 234.5ms | — |
| **total per step** | **619.6ms** | — |
| payload | **40KB** | — |

## 注意事项

1. **Server config 必须完整**：`--config` 中需包含 `dataset` 和 `denormalize_action` 配置，server 负责完整的前处理 + 后处理。同时 `--ckpt-path` 同级目录需有 `dataset_statistics.json`
2. **图像 key 白名单**：只有 `cam_high`, `cam_left_wrist`, `cam_right_wrist`, `agentview_image`, `robot0_eye_in_hand_image` 会 JPEG 压缩传输，其他 `uint8` 图像（如深度图）用无损 npy 格式
3. **线程安全**：`RemoteVLAZmq` 内部有锁保护，可安全地从多线程调用 `predict_action()`
4. **gRPC 后端**：设 `backend='grpc'`、`port=50051` 可切换到 gRPC（需 `grpc/` 目录依赖）
5. **防火墙**：确保 Server 的 ZMQ 端口（默认 5555）对 Client 可达，`telnet <server_ip> 5555` 可快速验证
