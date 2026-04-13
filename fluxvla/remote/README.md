# FluxVLA Remote Inference (ZMQ)

基于 ZeroMQ + msgpack 的远程 VLA 推理框架。模型推理和数据预处理部署在 GPU 服务器上，客户端只需发送 raw observation（JPEG 图像 + 关节状态），无需加载 tokenizer 或 image processor。

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
│                       │   action tensor   │  ↓                             │
│  denormalize → 执行    │ ◄──────────────── │  serialize actions             │
└───────────────────────┘                   └────────────────────────────────┘
```

**预处理在 server 端执行**：tokenize、image resize/normalize、state normalize 全部在 GPU 服务器上完成。客户端启动快，传输量小。

## 快速开始

### 1. 启动 Server（GPU 机器）

```bash
# ALOHA
python zmq_msgpack/serve_vla_zmq.py \
    --config configs/pi05/pi05_paligemma_aloha_full_finetune.py \
    --ckpt-path /path/to/checkpoint.pt \
    --host 0.0.0.0 --port 5555 --device cuda:0 --dtype bf16

# UR3
python zmq_msgpack/serve_vla_zmq.py \
    --config configs/pi05/pi05_paligemma_ur3_full_finetune.py \
    --ckpt-path /path/to/checkpoint.pt \
    --host 0.0.0.0 --port 5555

# LIBERO 评估
python zmq_msgpack/serve_vla_zmq.py \
    --config configs/pi05/pi05_paligemma_libero_10_full_finetune.py \
    --ckpt-path /path/to/checkpoint.pt \
    --host 0.0.0.0 --port 5555 --dataset-key eval
```

Server 启动时会自动从 config 中构建 dataset 预处理管道（加载 tokenizer 等），并打印：

```
[serve_vla_zmq] Dataset pipeline built from cfg.inference.dataset
[serve_vla_zmq] Model on cuda:0 (bf16), ready to serve.
```

**参数说明：**

| 参数            | 默认值    | 说明                                                  |
| --------------- | --------- | ----------------------------------------------------- |
| `--config`      | (必填)    | mmengine config 文件路径                              |
| `--ckpt-path`   | (必填)    | 模型 checkpoint 路径                                  |
| `--host`        | `0.0.0.0` | 监听地址                                              |
| `--port`        | `5555`    | 监听端口                                              |
| `--device`      | `cuda:0`  | 推理设备                                              |
| `--dtype`       | `bf16`    | 精度类型（bf16/fp16/fp32）                            |
| `--dataset-key` | 自动检测  | 从哪个 config key 读 dataset（`inference` 或 `eval`） |

### 2. 配置 Client

在 config 的 `inference`（或 `eval`）字段中添加 `remote_inference`：

#### ALOHA

```python
inference = dict(
    type='AlohaInferenceRunner',
    remote_inference=dict(
        enabled=True,
        backend='zmq',
        host='192.168.1.100',   # GPU 服务器 IP
        port=5555,
        timeout_s=30.0,
    ),
    seed=7,
    dataset=dict(...),          # client 端不会加载，仅 server 用
    denormalize_action=dict(
        type='DenormalizePrivateAction',
        norm_type='min_max',
        action_dim=14,
    ),
    action_chunk=50,
    operator=dict(type='AlohaOperator', ...),
)
```

#### UR3

```python
inference = dict(
    type='URInferenceRunner',
    remote_inference=dict(
        enabled=True,
        backend='zmq',
        host='192.168.1.100',
        port=5555,
        timeout_s=30.0,
    ),
    seed=7,
    dataset=dict(...),          # client 端不会加载
    denormalize_action=dict(
        type='DenormalizePrivateAction',
        norm_type='min_max',
        action_dim=7,
    ),
    action_chunk=50,
    operator=dict(type='UROperator', ...),
)
```

#### LIBERO 评估

```python
eval = dict(
    type='LiberoEvalRunner',
    remote_inference=dict(
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

### 3. 启动 Client

#### ALOHA / UR 实时推理

```bash
python scripts/inference.py \
    --config configs/pi05/pi05_paligemma_aloha_full_finetune.py \
    --ckpt-path /path/to/checkpoint.pt \
    --cfg-options inference.remote_inference.enabled=True \
                  inference.remote_inference.host=192.168.1.100
```

#### LIBERO 评估

```bash
torchrun --nnodes 1 --nproc_per_node 2 scripts/train.py \
    --config configs/pi05/pi05_paligemma_libero_10_full_finetune.py \
    --work-dir /path/to/work_dir \
    --eval-after-train \
    --cfg-options eval.remote_inference.enabled=True \
                  eval.remote_inference.host=192.168.1.100
```

## Client 端启动对比

|                       | 无 remote | remote（旧） | remote（新） |
| --------------------- | --------- | ------------ | ------------ |
| 加载模型权重          | Yes       | No           | No           |
| 加载 tokenizer        | Yes       | Yes          | **No**       |
| 加载 image processor  | Yes       | Yes          | **No**       |
| 构建 dataset pipeline | Yes       | Yes          | **No**       |
| 启动时间              | ~60s      | ~30s         | **\<5s**     |

## Config 字段说明

```python
remote_inference=dict(
    enabled=False,       # bool: 是否启用远程推理
    backend='zmq',       # str: 通信后端 ('zmq' 或 'grpc')
    host='localhost',    # str: 服务端地址
    port=5555,           # int: 服务端端口
    timeout_s=30.0,      # float: 单次请求超时秒数
)
```

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

运行集成测试（无需 checkpoint，使用 mock model）：

```bash
python -m fluxvla.remote.test_remote_inference
```

测试覆盖：ObsSerializer 序列化/反序列化、Aloha 3-cam 端到端、UR3 2-cam 端到端、Libero tuple-dataset 端到端。

## 注意事项

1. **ckpt_path 仍需提供**：客户端用它定位 `dataset_statistics.json`（denormalize 需要），但不加载模型
2. **server config 需匹配**：server 的 `--config` 要包含与 client 相同的 dataset transforms
3. **JPEG 压缩**：图像以 JPEG quality=95 传输，适合 robot camera 图像，有极小精度差异
4. **gRPC 后端**：设 `backend='grpc'`、`port=50051` 可切换到 gRPC（需 `grpc/` 目录依赖）
