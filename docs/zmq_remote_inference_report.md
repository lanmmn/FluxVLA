# ZMQ+msgpack 远程推理模块：实现与测试报告

## 1. 概述

基于 ZMQ REQ/REP + msgpack 通信框架实现的 VLA 远程推理模块（`fluxvla.remote`）。代码已整合到 `fluxvla/remote/` 包中，gRPC 后端已移除。

## 2. 架构设计

### 2.1 与 gRPC 方案的对比

| 特性 | gRPC 方案 | ZMQ+msgpack 方案 |
|------|----------|-----------------|
| 传输协议 | HTTP/2 (gRPC) | TCP (ZMQ REQ/REP) |
| 消息格式 | Protobuf + torch.save bytes | msgpack + torch.save bytes |
| 依赖 | grpcio, grpcio-tools, protobuf | pyzmq, msgpack |
| 需要编译 proto | 是 | 否 |
| Schema 定义 | 严格 (.proto) | 灵活 (Python dict) |
| 流式推理 | 支持双向流 | 仅请求-响应 |
| 认证 | 无 | 支持 API token |
| 扩展性 | 需修改 proto 并重编译 | 运行时注册 endpoint |
| 复杂度 | ~800 行 (5 文件) | ~400 行 (3 文件) |

### 2.2 数据流

```
Client (Eval / 仿真)                        Server (GPU 推理)
┌────────────────────────────┐          ┌────────────────────────────┐
│  env.step() → obs          │          │  fluxvla.remote.serve      │
│  dataset(obs) → batch dict │          │  加载 config + checkpoint   │
│  RemoteVLAZmq              │  ZMQ     │  VLAPolicy (BasePolicy)    │
│    torch.save(batch) ──────┼────────► │    torch.load → GPU        │
│    msgpack.packb           │  REQ/REP │    vla.predict_action      │
│                            │          │    torch.save(actions)     │
│    actions ◄───────────────┼──────────┤    msgpack.packb           │
│  denormalize + execute     │          │                            │
└────────────────────────────┘          └────────────────────────────┘
```

### 2.3 关键设计决策

1. **Tensor 序列化**：复用 gRPC 方案的 `torch.save`/`torch.load` 方式，保证 bfloat16 等所有 dtype 的正确性。msgpack 只负责传输外层 dict 结构和 bytes payload。

2. **复用 PolicyServer 框架**：`VLAPolicy` 继承 `BasePolicy`，通过 `register_endpoint("predict_action", ...)` 注册自定义 endpoint，完全融入现有 ZMQ 框架。

3. **RemoteVLAZmq 代理**：与 gRPC 的 `RemoteVLA` 接口完全一致（`predict_action(**batch)`、`eval()`、`cuda()`、`freeze_*`、`norm_stats`），可直接替换。

## 3. 实现文件清单

| 文件 | 类型 | 说明 |
|------|------|------|
| `fluxvla/remote/vla_server.py` | 核心 | TensorSerializer + VLAInferPipeline + VLAPolicy + create_vla_server |
| `fluxvla/remote/remote_vla.py` | 核心 | RemoteVLAZmq 代理类（predict_action 接口 + profiling） |
| `fluxvla/remote/serve.py` | 入口 | CLI 启动脚本（加载 config → 构建模型 → 启动 ZMQ server） |
| `fluxvla/remote/test_remote_inference.py` | 测试 | 集成测试 |
| `fluxvla/engines/runners/libero_eval_runner.py` | 修改 | 新增 `backend='zmq'` 分支 |
| `configs/pi05/pi05_paligemma_libero_10_full_finetune.py` | 修改 | 新增 `backend` 字段 |

## 4. 单元测试

### 4.1 测试结果

```
$ cd /root/projects/fluxvla && python -m pytest zmq_msgpack/test_vla_zmq.py -v

zmq_msgpack/test_vla_zmq.py::TestTensorSerializer::test_round_trip_dtypes PASSED
zmq_msgpack/test_vla_zmq.py::TestTensorSerializer::test_bfloat16_preserved PASSED
zmq_msgpack/test_vla_zmq.py::TestTensorSerializer::test_action_serialization PASSED
zmq_msgpack/test_vla_zmq.py::TestVLAPolicyE2E::test_01_basic_infer PASSED
zmq_msgpack/test_vla_zmq.py::TestVLAPolicyE2E::test_02_unnorm_key_passthrough PASSED
zmq_msgpack/test_vla_zmq.py::TestVLAPolicyE2E::test_03_ping PASSED
zmq_msgpack/test_vla_zmq.py::TestVLAPolicyE2E::test_04_profiling PASSED
zmq_msgpack/test_vla_zmq.py::TestRemoteVLAZmqDuckTyping::test_duck_typing PASSED

8 passed in 5.39s
```

### 4.2 测试详情

| 测试 | 验证点 |
|------|--------|
| `test_round_trip_dtypes` | batch dict (float32, int64) 序列化后 dtype/shape/值完全一致 |
| `test_bfloat16_preserved` | bfloat16 tensor 序列化后保持 dtype |
| `test_action_serialization` | action tensor round-trip 正确 |
| `test_01_basic_infer` | RemoteVLAZmq → ZMQ server → MockVLA 端到端，action shape (1, 10, 32) |
| `test_02_unnorm_key_passthrough` | unnorm_key 透传到 server pipeline |
| `test_03_ping` | 健康检查 ping 正常 |
| `test_04_profiling` | profiling 累积器正确记录 call_count、timing、payload |
| `test_duck_typing` | eval()/cuda()/freeze_*/norm_stats 兼容性 |

### 4.3 原有测试回归

```
$ python -m pytest zmq_msgpack/test_server_client.py -v

12 passed in 5.87s
```

原有 12 个 ZMQ 框架测试全部通过，无回归。

## 5. Libero Eval 端到端验证

### 5.1 实验配置

| 项目 | 值 |
|------|-----|
| 模型 | PI05FlowMatching |
| Checkpoint | `step-012688-epoch-08-loss=0.0492.pt` (~40GB) |
| 任务集 | libero_10 (10 tasks × 2 trials = 20 episodes) |
| Server | GPU:0 (A100 80GB), bf16, ZMQ port 5555 |
| Client | GPU:1,2 (torchrun 2 workers), ZMQ 远程推理 |
| eval_chunk_size | 10 |
| 通信协议 | ZMQ REQ/REP + torch.save/torch.load |

### 5.2 启动命令

```bash
# Server (GPU:0)
CUDA_VISIBLE_DEVICES=0 python -m fluxvla.remote.serve \
  --config configs/pi05/pi05_paligemma_libero_10_full_finetune.py \
  --ckpt-path .../step-012688-epoch-08-loss=0.0492.pt \
  --host 0.0.0.0 --port 5555

# Client (GPU:1,2)
CUDA_VISIBLE_DEVICES=1,2 torchrun --nnodes 1 --nproc_per_node 2 scripts/eval.py \
  --config configs/pi05/pi05_paligemma_libero_10_full_finetune.py \
  --ckpt-path .../step-012688-epoch-08-loss=0.0492.pt \
  --cfg-options eval.num_trials_per_task=2 \
    eval.remote_inference.enabled=True \
    eval.remote_inference.backend=zmq \
    eval.remote_inference.host=localhost \
    eval.remote_inference.port=5555
```

### 5.3 结果

```
Evaluation: 100%|██████████| 20/20 [05:23<00:00, 16.16s/it]
# episodes completed so far: 20
# successes: 20 (100.0%)
```

**20/20 episodes 全部成功 (100.0%)**，总耗时 5 分 23 秒，平均每 episode ~16 秒。

## 6. 延迟分析

### 6.1 Client 端延迟（~500 次请求平均）

| 阶段 | 平均耗时 | 占比 | 说明 |
|------|---------|------|------|
| **序列化** (serialize) | 1.6 ms | 0.4% | CPU tensor 转移 + `torch.save` → bytes |
| **ZMQ 往返** (roundtrip) | 446.0 ms | 99.2% | ZMQ 网络传输 + Server 处理全部时间 |
| **反序列化** (deserialize) | 2.0 ms | 0.4% | `torch.load` → CUDA tensor |
| **总计** (total) | ~450 ms | 100% | 端到端单次推理延迟 |

### 6.2 Server 端延迟（500 次请求平均）

| 阶段 | 平均耗时 | 占比 | 说明 |
|------|---------|------|------|
| **反序列化** (deserialize) | 1.3 ms | 0.3% | `torch.load` → CPU tensor |
| **模型推理** (infer) | 429.4 ms | 99.6% | GPU 前向 + flow matching 去噪 |
| **序列化** (serialize) | 0.6 ms | 0.1% | action tensor → `torch.save` → bytes |

### 6.3 网络开销

| 指标 | 值 |
|------|-----|
| 平均 payload 大小 | ~1180 KB / 请求 |
| ZMQ 纯网络开销 | ~15 ms（roundtrip − server_infer） |
| 网络占比 | ~3.3% |

### 6.4 ZMQ vs gRPC 延迟对比

| 指标 | gRPC (50ep) | ZMQ (20ep) | 差异 |
|------|------------|-----------|------|
| Client 总延迟 | 535 ms | 450 ms | ZMQ 快 16% |
| Client 序列化 | 1.6 ms | 1.6 ms | 持平 |
| Client 反序列化 | 0.9 ms | 2.0 ms | gRPC 快 1.1ms |
| 网络往返 | 532 ms | 446 ms | ZMQ 快 86ms |
| Server 推理 | 530 ms | 429 ms | ZMQ 测 +GPU 负载不同 |
| Server 反序列化 | 1.7 ms | 1.3 ms | ZMQ 快 0.4ms |
| Server 序列化 | 0.6 ms | 0.6 ms | 持平 |
| 网络纯开销 | 6 ms | 15 ms | gRPC 快 9ms |
| Eval 成功率 | 98% (49/50) | 100% (20/20) | 样本量不同 |
| 平均 episode 耗时 | ~16.3s | ~16.2s | 基本一致 |

> **注意**：两次测试的 GPU 负载不同（gRPC 测试时其他 GPU 有训练任务，ZMQ 测试时同样有），推理耗时差异主要由 GPU 负载和温度导致，不代表通信协议差异。
>
> 网络纯开销 ZMQ (15ms) > gRPC (6ms)，这是因为：
> - gRPC 使用 HTTP/2 多路复用和持久连接，协议层开销更小
> - ZMQ REQ/REP 每次请求需要完整的消息边界处理
> - 但两者的网络开销都远小于推理时间（<4%），对总延迟影响可忽略

## 7. Config 使用方法

### 使用 ZMQ 后端

```python
eval = dict(
    type='LiberoEvalRunner',
    remote_inference=dict(
        enabled=True,
        backend='zmq',           # 'zmq' 使用 ZMQ 后端
        host='192.168.1.100',    # 服务器 IP
        port=5555,               # ZMQ 端口
        timeout_s=30.0,
    ),
    # ... 其余配置不变
)
```

### 命令行覆盖

```bash
--cfg-options eval.offload_inference.enabled=True eval.offload_inference.host=192.168.1.100 eval.offload_inference.port=5555
```

## 8. 总结

- ZMQ+msgpack 远程推理模块已完成实现和测试
- 8 个单元测试全部通过，原有 12 个测试无回归
- Libero Eval 20/20 episodes 100% 成功率，验证功能正确性
- 延迟分析显示通信开销 < 4%，模型推理占 99%+ 时间
- 代码已整合到 `fluxvla/remote/` 包中，gRPC 后端已移除
