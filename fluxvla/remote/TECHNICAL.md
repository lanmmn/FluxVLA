# FluxVLA Remote Inference - Technical Documentation

## Architecture Overview

```
+------------------+          TCP/ZMQ           +------------------+
|   Eval Client    |  ========================  |   GPU Server     |
|                  |                            |                  |
| LiberoEvalRunner |                            | serve_vla_zmq.py |
|        |         |                            |        |         |
| RemoteVLAZmq     |  -- predict_action req --> | VLAPolicy        |
|   (drop-in proxy)|                            |   |              |
|        |         |                            | VLAInferPipeline |
|  predict_action()|  <-- action response ---   |   |              |
|                  |                            | PI05FlowMatching |
+------------------+          msgpack           +------------------+
```

## 模块结构

```
fluxvla/remote/
    __init__.py          # 包导出
    policy.py            # BasePolicy 抽象基类
    server_client.py     # ZMQ REQ/REP 通信框架 + MsgSerializer
    vla_server.py        # VLA 推理服务端（VLAInferPipeline, VLAPolicy, TensorSerializer）
    remote_vla.py        # VLA 客户端代理（RemoteVLAZmq）
```

## 通信协议

### 传输层

- **Socket 类型**：ZMQ REQ/REP（同步请求-响应）
- **传输协议**：TCP (`tcp://host:port`)
- **消息序列化**：msgpack（外层消息帧）+ torch.save/load（tensor payload）

### 请求格式

```python
{
    "endpoint": "predict_action",      # 端点名称
    "data": {
        "batch_data": bytes,           # torch.save 序列化的 tensor batch
        "unnorm_key": "libero_10",     # denormalization key
    },
}
```

### 响应格式

```python
# 成功
{
    "action_data": bytes,    # torch.save 序列化的 action tensor（已反归一化）
    "infer_time": 0.025,     # 服务端推理 + 预处理耗时（秒）
}

# 失败
{
    "error": "Error message"
}
```

> **注意**：action 已在 server 端完成反归一化，client 收到的是可直接执行的机器人指令。

## 核心类详解

### TensorSerializer (`vla_server.py`)

负责 torch tensor 的序列化/反序列化。使用 `torch.save/load` 而非 pickle，确保安全。

```python
class TensorSerializer:
    serialize_batch(batch: dict) -> bytes      # dict of tensors -> bytes
    deserialize_batch(data: bytes) -> dict     # bytes -> dict of tensors
    serialize_actions(actions: Tensor) -> bytes # action tensor -> bytes
    deserialize_actions(data: bytes) -> Tensor # bytes -> action tensor
```

**安全保障**：

- `torch.load(..., weights_only=True)` 禁止任意 pickle 反序列化
- 所有 tensor 先移到 CPU 再序列化，避免跨设备问题

### VLAInferPipeline (`vla_server.py`)

封装真实 VLA 模型，提供 GPU 推理能力。

```python
class VLAInferPipeline:
    def __init__(self, vla_model, device="cuda:0", mixed_precision_dtype=torch.bfloat16):
        # 将模型移到指定设备，设置 eval 模式

    @torch.no_grad()
    def predict_action(self, **batch) -> torch.Tensor:
        # 1. 将 batch 中的 tensor 移到 GPU
        # 2. 在 torch.autocast 下调用 vla.predict_action()
        # 3. 返回 action tensor
```

### VLAPolicy (`vla_server.py`)

将 `VLAInferPipeline` 包装为 `BasePolicy`，注册为 PolicyServer 的 endpoint。

```python
class VLAPolicy(BasePolicy):
    def __init__(self, pipeline, dataset=None,
                 denormalize_action=None, task_suite_name=""):
        # pipeline: VLAInferPipeline 推理管道
        # dataset: 可选的预处理管道（server-side preprocessing）
        # denormalize_action: 可选的反归一化 transform（server-side denormalization）
        # task_suite_name: denormalization 查找 key（如 "libero_10"）
```

**predict_action 流程**：

1. 反序列化 raw obs（JPEG 解码）
2. 预处理（dataset transforms: image resize/normalize, tokenize, state normalize）
3. 模型推理（GPU）
4. **action 反归一化**（如果配置了 `denormalize_action`）
5. 序列化 action 返回

**注册的端点**：

| Endpoint         | 功能                          | 需要输入 |
| ---------------- | ----------------------------- | -------- |
| `predict_action` | raw obs → 反归一化后的 action | Yes      |
| `get_status`     | 服务端状态查询                | No       |
| `ping`           | 健康检查                      | No       |
| `kill`           | 远程关闭服务                  | No       |

**性能统计**：每 50 次请求打印一次 deserialize/preprocess/infer/serialize 耗时。

### RemoteVLAZmq (`remote_vla.py`)

客户端代理，实现与本地 VLA 模型完全相同的接口（duck typing）。

**predict_action 调用流程**：

```
1. 分离 unnorm_key（非 tensor 字段）
2. 将所有 tensor 移到 CPU
3. torch.save() 序列化为 bytes
4. msgpack 打包请求，ZMQ 发送
5. ZMQ 接收响应，msgpack 解包
6. torch.load(map_location=device) 反序列化 action
7. 返回 action tensor（已在目标设备上）
```

**Duck typing 兼容**：

```python
vla.eval()                        # 返回 self
vla.cuda(device_id)               # 记录设备，返回 self
vla.freeze_vision_backbone = True  # 静默吸收 freeze_* 赋值
vla.norm_stats = {...}            # 支持设置 norm_stats
```

这使得 `LiberoEvalRunner.run_setup()` 不需要区分本地/远程模型。

## 数据流

### 完整推理流程

```
LiberoEvalRunner.run()
    |
    v
obs = env.step(action)                       # LIBERO 环境产生观测
    |
    +--[本地路径]-------+--[远程路径]----------+
    |                   |                      |
    v                   v                      |
 dataset(obs)        raw obs (dict)            |
    |                   |                      |
    v                   v                      |
 predict_action()   RemoteVLAZmq              |
 (本地 GPU)          .predict_action()         |
    |                   |                      |
    |                   v                      |
    |              ObsSerializer.to_bytes()    |
    |                   |  (~100KB JPEG)       |
    |                   v                      |
    |              ZMQ REQ send ──► ZMQ REP recv
    |                                  |
    |                   VLAPolicy.predict_action()  # 服务端
    |                   |-- ObsSerializer.from_bytes()
    |                   |-- dataset(obs) → tensor batch
    |                   |-- VLAInferPipeline.predict_action()  # GPU 推理
    |                   |-- denormalize_action()  # 反归一化
    |                   |-- TensorSerializer.serialize_actions()
    |                                  |
    |              ZMQ REP send ──► ZMQ REQ recv
    |                   |
    |                   v
    |              torch.load(action_data)
    |                   |
    v                   v
 denormalize()    actions: 已反归一化，可直接执行
    |                   |
    v                   v
 env.step(action)   env.step(action.tolist())
```

### Tensor 规格

**Server 内部 tensor（预处理后）**：

| 字段          | Shape          | Dtype   | 说明               |
| ------------- | -------------- | ------- | ------------------ |
| `images`      | (1, C, H, W)   | float32 | C = num_views * 3  |
| `img_masks`   | (1, num_views) | bool    | 视角掩码           |
| `lang_tokens` | (1, seq_len)   | int64   | tokenized 任务描述 |
| `lang_masks`  | (1, seq_len)   | bool    | token 掩码         |
| `states`      | (1, state_dim) | float32 | 本体感知状态       |

**返回给 client 的 action**：

| 字段               | Shape                           | Dtype   | 说明                             |
| ------------------ | ------------------------------- | ------- | -------------------------------- |
| `actions` (output) | (1, n_action_steps, action_dim) | float32 | 反归一化后的动作序列，可直接执行 |

## Config 集成方式

### Client 端

`LiberoEvalRunner.__init__()` 通过 `offload_inference` 参数决定使用本地还是远程模型：

```python
# fluxvla/engines/runners/libero_eval_runner.py

if self._use_offload:
    # 远程模式：不加载模型、不构建 dataset、不构建 denormalize_action
    # 不需要 dataset_statistics.json
    self.vla = RemoteVLAZmq(...)
    self.dataset = None
    self.denormalize_action = None
else:
    # 本地模式：完整加载
    self.vla = build_vla_from_cfg(cfg.model).eval()
    self.dataset = build_dataset_from_cfg(dataset)
    self.denormalize_action = build_transform_from_cfg(denormalize_action)
```

### Server 端

`serve_vla_zmq.py` 从 config 中读取 `denormalize_action` 配置并传给 `VLAPolicy`：

```python
# zmq_msgpack/serve_vla_zmq.py

# 从 config 构建反归一化 transform
denorm_cfg = dict(cfg.eval.denormalize_action)
denorm_cfg['norm_stats'] = data_stat_path
denormalize_action = build_transform_from_cfg(denorm_cfg)

# 传给 server
server = create_vla_server(
    pipeline=pipeline,
    dataset=dataset,
    denormalize_action=denormalize_action,
    task_suite_name=cfg.eval.task_suite_name,
)
```

## 性能 Profiling

### 客户端（RemoteVLAZmq）

每 50 次调用输出：

```
[RemoteVLAZmq profiling] calls=50  avg_total=45.2ms  avg_serialize=2.1ms
  avg_zmq_roundtrip=15.3ms  avg_server_infer=25.8ms  avg_deserialize=1.2ms
  avg_payload=512KB
```

### 服务端（VLAPolicy）

每 50 次请求输出：

```
[ZMQ VLAServer] req=50  deserialize=1.8ms  infer=23.5ms  serialize=0.9ms
  avg_infer=24.1ms
```

### 典型延迟分布

| 阶段               | 耗时      | 占比 |
| ------------------ | --------- | ---- |
| 客户端序列化       | ~2ms      | 4%   |
| 网络传输（局域网） | ~15ms     | 33%  |
| 服务端推理         | ~25ms     | 56%  |
| 客户端反序列化     | ~1ms      | 2%   |
| 其他开销           | ~2ms      | 5%   |
| **总计**           | **~45ms** | 100% |
