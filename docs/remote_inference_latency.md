# 远程推理延迟分析报告

## 1. 实验配置

| 项目 | 值 |
|------|-----|
| 模型 | PI05FlowMatching (mini-diffuser) |
| Checkpoint | `step-012688-epoch-08-loss=0.0492.pt` (~40GB) |
| 任务集 | libero_10 (10 tasks × 5 trials = 50 episodes) |
| Server | GPU:0 (A100 80GB), bf16, gRPC port 50051 |
| Client | GPU:1,2 (torchrun 2 workers), 远程推理模式 |
| eval_chunk_size | 10 |
| 通信协议 | gRPC + torch.save/torch.load 序列化 |
| 总请求数 | ~1350 次（Server 端统计） |
| 每 Worker 请求数 | ~700 次 |
| 总耗时 | 13 分 35 秒 |
| 成功率 | 49/50 episodes (98%) |

## 2. 延迟分解

### 2.1 Client 端（每次 `predict_action` 调用）

| 阶段 | 平均耗时 | 占比 | 说明 |
|------|---------|------|------|
| **序列化** (serialize) | 1.6 ms | 0.3% | CPU tensor 转移 + `torch.save` → bytes |
| **gRPC 往返** (roundtrip) | 532 ms | 99.4% | 网络传输 + Server 处理全部时间 |
| **反序列化** (deserialize) | 0.9 ms | 0.2% | `torch.load` → CUDA tensor |
| **总计** (total) | ~535 ms | 100% | 端到端单次推理延迟 |

### 2.2 Server 端（每次 `TensorBatchInfer` 处理）

| 阶段 | 平均耗时 | 占比 | 说明 |
|------|---------|------|------|
| **反序列化** (deserialize) | 1.7 ms | 0.3% | `torch.load` → CPU tensor |
| **模型推理** (infer) | 526–533 ms | 99.1% | GPU 前向 + flow matching 去噪 |
| **序列化** (serialize) | 0.6 ms | 0.1% | action tensor → `torch.save` → bytes |

### 2.3 网络开销

| 指标 | 值 |
|------|-----|
| 平均 payload 大小 | ~1180 KB / 请求 |
| 网络纯开销 | ~6 ms（gRPC roundtrip − server infer） |
| 网络占比 | ~1.1% |

## 3. 延迟瀑布图

```
Client                                              Server
  │                                                    │
  ├─ serialize ──────── 1.6ms ─┐                       │
  │                            │                       │
  │                        gRPC request (~1180KB)      │
  │                            ├──── ~3ms ────────────►│
  │                            │                       ├─ deserialize ── 1.7ms
  │                            │                       ├─ GPU infer ──── 530ms
  │                            │                       ├─ serialize ──── 0.6ms
  │                        gRPC response               │
  │                    ◄────── ~3ms ───────────────────┤
  ├─ deserialize ───── 0.9ms ─┘                        │
  │                                                    │
  ├─────────── total: ~535ms ──────────────────────────┤
```

## 4. 分析

### 4.1 瓶颈分析

**模型推理占绝对主导地位**（~99%），序列化/反序列化和网络传输开销可忽略：

| 分类 | 耗时 | 占比 |
|------|------|------|
| GPU 推理 | ~530 ms | 99.1% |
| 网络传输 | ~6 ms | 1.1% |
| 序列化 + 反序列化（双端） | ~4.8 ms | 0.9% |

### 4.2 关键结论

1. **远程推理 overhead 极小**：相比本地推理（~530ms），远程推理仅增加 ~5ms（序列化 + 网络），不到 1% 的额外开销。
2. **序列化效率高**：`torch.save`/`torch.load` 在 ~1.2MB payload 下序列化耗时 < 2ms，无需考虑压缩或更复杂的序列化方案。
3. **网络不是瓶颈**：同机通信（localhost）~6ms 往返。跨机器内网（千兆+）预计 < 10ms，对总延迟影响仍然很小。
4. **优化方向应聚焦于模型推理本身**：如 mini-diffuser 减少去噪步数、模型量化、TensorRT 编译等。

### 4.3 与本地推理对比

| 模式 | 单次推理延迟 | 额外开销 | 适用场景 |
|------|------------|---------|---------|
| 本地推理 | ~530 ms | — | 单机多 GPU |
| 远程推理（同机） | ~535 ms | +5 ms (~1%) | 仿真/eval 与 GPU 分离 |
| 远程推理（同内网） | ~540 ms (估计) | +10 ms (~2%) | 机器人控制 + GPU 服务器 |

## 5. 数据采集方法

### Client 端
`RemoteVLA` 内置 profiling，使用 `time.perf_counter()` 分别记录 serialize、gRPC roundtrip、deserialize 三段耗时，每 50 次调用打印累计平均值。

### Server 端
`TensorBatchInfer` handler 内置 profiling，分别记录 deserialize、infer、serialize 三段耗时，每 50 次请求打印累计平均值。

### 原始日志示例

**Client (Worker 0, 第 700 次调用)**:
```
[RemoteVLA profiling] calls=700  avg_total=535.2ms  avg_serialize=1.6ms  avg_grpc_roundtrip=532.4ms  avg_server_infer=526.1ms  avg_deserialize=0.9ms  avg_payload=1180KB
```

**Server (第 1350 次请求)**:
```
[TensorBatchInfer] req=1350  deserialize=1.7ms  infer=533.2ms  serialize=0.6ms  avg_infer=530.5ms
```
