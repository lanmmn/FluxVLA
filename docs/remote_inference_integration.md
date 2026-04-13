# 远程推理集成：设计、实现与验证

## 1. 背景与目标

`LiberoEvalRunner` 原本只支持本地 GPU 推理——模型加载、前向推理、eval 循环全部运行在同一台机器上。在实际部署中，经常需要「本地跑环境 / 机器人控制 + 远程 GPU 服务器跑模型推理」的分离架构。

**目标**：在 config 中通过一个开关控制是否启用远程推理，启用后 eval 循环代码零改动，推理请求自动转发至 gRPC 服务器。

## 2. 方案设计

### 2.1 备选方案分析

| 方案 | 描述 | 优点 | 缺点 |
|------|------|------|------|
| A: 发送 JPEG + 服务端预处理 | Client 发送原始图片 bytes，Server 做 transform + 推理 | 传输量小 | Server 需完整 transform pipeline + tokenizer |
| B: 发送预处理后的 tensor (protobuf repeated float) | 把 tensor 数据展开成 float 数组放入 proto 字段 | proto 原生支持 | 丢失 dtype (bfloat16)、需手动管理 shape |
| C: pickle 序列化整个 batch | 用 pickle 序列化整个 batch dict | 简单 | 安全风险（任意代码执行） |
| **D: torch.save 序列化（采用）** | 用 `torch.save` / `torch.load` + `BytesIO` 序列化 batch dict | 原生支持所有 dtype、零额外依赖、shape 自动保持 | 依赖 PyTorch（双端都有） |
| E: 在 runner 层面 if-else | 在 eval 循环中插入 remote inference 分支 | 灵活 | 侵入性强，维护负担大 |

### 2.2 最终方案：RemoteVLA 代理类

采用 **方案 D** 的序列化方式，配合 **代理模式** 实现：

1. 新建 `RemoteVLA` 类，实现与真实 VLA 模型完全相同的 `predict_action(**batch)` 接口
2. 在 `LiberoEvalRunner.__init__` 中，当 `remote_inference.enabled=True` 时，用 `RemoteVLA` 替代 `self.vla`
3. Eval 循环调用 `self.vla.predict_action(**batch)` 时，自动通过 gRPC 转发

**核心数据流**：

```
Client:  dataset(obs) → batch dict → torch.save → bytes → gRPC →
Server:  → torch.load → GPU → vla.predict_action → torch.save → bytes → gRPC →
Client:  → torch.load → actions tensor → denormalize → env.step
```

### 2.3 数据量评估

以 PI05 模型、2 视角 224x224 为例：

| 字段 | 类型 | 大小 |
|------|------|------|
| `images` | float32, (1, 6, 224, 224) | ~2.4 MB |
| `lang_tokens` | int64, (1, ~48) | ~384 B |
| `lang_masks` | bool, (1, ~48) | ~48 B |
| `img_masks` | bool, (1, 2) | ~2 B |
| `states` | bfloat16, (1, 32) | ~64 B |
| **合计** | | **~2.4 MB / 请求** |

gRPC 默认 max message 4MB，配置后放宽至 64MB，无压力。

## 3. 实现步骤

### Step 1: 扩展 Proto 定义

在 `vla_service.proto` 中新增 2 个消息 + 1 个 RPC，不修改任何现有接口：

```protobuf
message TensorBatchInferRequest {
  bytes batch_data = 1;    // torch.save 序列化的 batch dict
  string unnorm_key = 2;   // 反归一化 key
}

message TensorBatchInferResponse {
  bytes action_data = 1;   // torch.save 序列化的 action tensor
  double infer_time = 2;   // 推理耗时
}

rpc TensorBatchInfer(TensorBatchInferRequest) returns (TensorBatchInferResponse);
```

### Step 2: RemoteVLA 客户端代理 (`grpc/remote_vla.py`)

关键设计点：
- `predict_action(**kwargs)`: 提取 `unnorm_key` 字符串字段 → tensor 移到 CPU → `torch.save` 序列化 → gRPC 发送 → 反序列化 → 移回原始 device
- Duck-typing: `eval()`, `cuda()`, `freeze_*` 属性赋值全部为 no-op，使 `run_setup()` 无需分支
- Channel options: `max_send/receive_message_length = 64MB`

### Step 3: Server 端 TensorBatchInfer (`grpc/grpc_server.py`)

- 新增 `VLAInferPipeline` 类：封装真实 VLA 模型 + `torch.no_grad()` + `torch.autocast`
- `TensorBatchInfer` handler: `torch.load` → 移入 GPU → `pipeline.predict_action` → `torch.save` 返回
- 未配置 `vla_pipeline` 时返回 `UNIMPLEMENTED`

### Step 4: Server 启动脚本 (`grpc/serve_vla.py`)

复用 mmengine Config 系统，自动完成：加载 config → 构建 VLA 模型 → 加载 checkpoint → 加载 norm_stats → 创建 VLAInferPipeline → 启动 gRPC server。

### Step 5: LiberoEvalRunner 集成

`__init__` 中新增 `remote_inference: Dict = None` 参数：
- `enabled=True` 时：跳过本地模型构建和 checkpoint 加载，创建 `RemoteVLA`
- `enabled=False`（默认）：行为完全不变
- `run_setup` 中：对 `RemoteVLA` 跳过 `.cuda()` 调用
- `run` 循环：零改动

### Step 6: Config 配置

```python
eval = dict(
    type='LiberoEvalRunner',
    remote_inference=dict(
        enabled=False,       # True 开启远程推理
        host='localhost',
        port=50051,
        timeout_s=30.0,
    ),
    # ...
)
```

## 4. 测试验证

### 4.1 单元测试

共 19 个端到端测试，全部通过：

```
$ cd grpc && python -m unittest test_grpc test_tensor_batch -v

Ran 19 tests in 4.470s
OK
```

**test_tensor_batch.py 测试内容**（8 个）：

| 测试 | 验证点 |
|------|--------|
| `TestTensorSerialization.test_round_trip_dtypes` | batch dict 序列化后 dtype/shape/值完全一致 |
| `TestTensorSerialization.test_bfloat16_preserved` | bfloat16 tensor 序列化后保持 dtype |
| `TestTensorBatchInferRPC.test_01_basic_infer` | 端到端 RPC，action shape 正确 |
| `TestTensorBatchInferRPC.test_02_unnorm_key_passthrough` | unnorm_key 透传 |
| `TestTensorBatchInferRPC.test_03_invalid_batch_data` | 损坏数据返回 INVALID_ARGUMENT |
| `TestRemoteVLA.test_01_predict_action` | RemoteVLA.predict_action 端到端 |
| `TestRemoteVLA.test_02_duck_typing` | eval()/cuda()/freeze_*/norm_stats 兼容 |
| `TestNoVLAPipeline.test_unimplemented` | 无 pipeline 时返回 UNIMPLEMENTED |

### 4.2 Libero Eval 端到端验证

**实验配置**：

| 项目 | 值 |
|------|-----|
| 模型 | PI05FlowMatching |
| Checkpoint | `step-012688-epoch-08-loss=0.0492.pt` (~40GB) |
| 任务集 | libero_10 (10 tasks x 2 trials = 20 episodes) |
| Server | GPU:0 (A100 80GB), bf16, gRPC port 50051 |
| Client | GPU:1,2 (torchrun 2 workers), 远程推理模式 |
| eval_chunk_size | 10 |

**启动命令**：

```bash
# Server (GPU:0)
CUDA_VISIBLE_DEVICES=0 python grpc/serve_vla.py \
  --config configs/pi05/pi05_paligemma_libero_10_full_finetune.py \
  --ckpt-path .../step-012688-epoch-08-loss=0.0492.pt \
  --host 0.0.0.0 --port 50051

# Client (GPU:1,2)
CUDA_VISIBLE_DEVICES=1,2 torchrun --nnodes 1 --nproc_per_node 2 scripts/eval.py \
  --config configs/pi05/pi05_paligemma_libero_10_full_finetune.py \
  --ckpt-path .../step-012688-epoch-08-loss=0.0492.pt \
  --cfg-options eval.num_trials_per_task=2 eval.remote_inference.enabled=True
```

**结果**：

```
Evaluation: 100%|██████████| 20/20 [05:18<00:00, 15.95s/it]
# episodes completed so far: 20
# successes: 20 (100.0%)
```

- **20/20 episodes 全部成功 (100.0%)**
- 总耗时 5 分 18 秒，平均每 episode ~16 秒
- Rollout 视频正常保存于 checkpoint 目录下

## 5. 文件清单

| 文件 | 类型 | 说明 |
|------|------|------|
| `grpc/vla_service.proto` | 修改 | 新增 TensorBatchInfer 消息和 RPC |
| `grpc/remote_vla.py` | 新建 | RemoteVLA 代理类 |
| `grpc/grpc_server.py` | 修改 | 新增 VLAInferPipeline + TensorBatchInfer handler |
| `grpc/serve_vla.py` | 新建 | Server 启动脚本 |
| `grpc/test_tensor_batch.py` | 新建 | 8 个 TensorBatch/RemoteVLA 测试 |
| `grpc/vla_service_pb2.py` | 重新生成 | protobuf 编译产物 |
| `grpc/vla_service_pb2_grpc.py` | 重新生成 | gRPC 编译产物 |
| `fluxvla/engines/runners/libero_eval_runner.py` | 修改 | 新增 remote_inference 参数 |
| `configs/pi05/pi05_paligemma_libero_10_full_finetune.py` | 修改 | 新增 remote_inference 配置项 |
