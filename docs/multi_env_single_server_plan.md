# 多仿真环境 + 单模型服务器架构方案

## 1. 背景与动机

当前 Libero Eval 使用 torchrun 启动多个 worker，每个 worker 运行一个 env + 一个模型副本（或通过 RemoteVLA 共享一个远程模型）。但存在一个关键问题：

**模型推理是瓶颈（~530ms/次），而 env.step 相对很快（~10-20ms）。**

如果只有一个 GPU 能用于推理，当前架构下多个 env worker 的请求是串行排队的——一个 worker 等推理时，其他 worker 也在等。理想情况下，应该让多个 env 并行运行，在等待推理结果时继续执行非推理工作（env reset、数据预处理、结果记录等）。

### 核心问题

> 仿真场景下，是否可以只挂载一个主模型，启动多个 Libero env 去 query 它？

**答案：可以。** gRPC server 天然支持并发请求，配合异步客户端或多线程 env 即可实现。

## 2. 当前架构 vs 目标架构

### 2.1 当前架构（torchrun 多进程）

```
┌─────────────────┐  ┌─────────────────┐
│  Worker 0       │  │  Worker 1       │
│  (GPU:1)        │  │  (GPU:2)        │
│  env + dataset  │  │  env + dataset  │
│  RemoteVLA ─────┼──┼─ RemoteVLA ─────┐
└─────────────────┘  └─────────────────┘│
                                         │ gRPC
                     ┌───────────────────▼──┐
                     │  Model Server (GPU:0) │
                     │  VLAInferPipeline     │
                     └──────────────────────┘
```

- 每个 worker 是独立进程（torchrun 启动）
- 每个 worker 串行执行：obs → preprocess → **predict_action (等 ~530ms)** → denorm → env.step → 记录
- Worker 之间的推理请求到达 server 时由 gRPC ThreadPool 处理，但 GPU 推理本身是串行的
- **问题**：Worker 0 在等推理时，Worker 1 也在等推理，两者的 env 都闲置着

### 2.2 目标架构（多 env + 单 server + 异步推理）

```
                    ┌──────────────────────────────────┐
                    │         Orchestrator             │
                    │                                  │
                    │  ┌──────┐ ┌──────┐ ┌──────┐     │
                    │  │Env 0 │ │Env 1 │ │Env N │     │
                    │  └──┬───┘ └──┬───┘ └──┬───┘     │
                    │     │        │        │          │
                    │  ┌──▼────────▼────────▼──┐      │
                    │  │   Async Request Queue  │      │
                    │  └────────────┬───────────┘      │
                    │               │ gRPC (batched)   │
                    └───────────────┼──────────────────┘
                                    │
                    ┌───────────────▼──────────────────┐
                    │     Model Server (GPU:0)          │
                    │     VLAInferPipeline              │
                    │     支持 dynamic batching         │
                    └──────────────────────────────────┘
```

## 3. 方案设计

### 3.1 方案 A：多线程 Env + 异步 gRPC（推荐）

**核心思路**：在单个进程内用线程池管理多个 env，每个 env 的推理请求通过异步 gRPC 发出，等待期间线程让出 CPU 给其他 env。

```python
import concurrent.futures
import threading

class MultiEnvEvalRunner:
    def __init__(self, num_envs, remote_inference_config, ...):
        self.num_envs = num_envs
        self.envs = [create_libero_env(task) for task in tasks]
        self.vla = RemoteVLA(**remote_inference_config)  # 共享一个连接
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_envs)

    def run_single_env(self, env_id, task_id, trial_id):
        """单个 env 的 rollout 循环"""
        env = self.envs[env_id]
        obs = env.reset()
        for step in range(max_steps):
            batch = self.dataset.transform(obs)
            # 推理（阻塞当前线程，但不阻塞其他线程）
            actions = self.vla.predict_action(**batch)
            action = denormalize(actions)
            obs, reward, done, info = env.step(action)
            if done:
                break
        return success

    def run(self):
        """并发运行多个 env"""
        futures = []
        for task_id, trial_id in all_trials:
            env_id = allocate_env(task_id)
            future = self.executor.submit(
                self.run_single_env, env_id, task_id, trial_id)
            futures.append(future)

        results = [f.result() for f in concurrent.futures.as_completed(futures)]
```

**优点**：
- 实现简单，复用现有 `RemoteVLA`
- gRPC channel 天然线程安全，多线程共享一个连接
- env 等推理时线程阻塞，其他 env 线程可以继续执行
- 无需修改 server 端

**缺点**：
- Python GIL 限制 CPU 密集型并行（但 env.step 大部分时间在 C 扩展中）
- Libero env (MuJoCo) 是否线程安全需要验证

**GIL 影响评估**：
- `env.step()` → MuJoCo C 库，释放 GIL ✓
- `torch.save/load` → C 扩展，释放 GIL ✓
- gRPC 等待 → I/O 等待，释放 GIL ✓
- 数据预处理（transforms）→ 主要是 numpy/torch 操作，大部分释放 GIL ✓
- **结论**：GIL 对此场景影响很小

### 3.2 方案 B：多进程 Env + 共享 gRPC

**核心思路**：每个 env 在独立进程中运行（避免 GIL 和线程安全问题），通过各自的 gRPC channel 查询同一个 server。

```python
import torch.multiprocessing as mp

def env_worker(env_id, task_queue, result_queue, remote_config):
    """独立进程：运行 env + 查询远程模型"""
    vla = RemoteVLA(**remote_config)  # 每个进程独立连接
    while True:
        task = task_queue.get()
        if task is None:
            break
        result = run_single_episode(env_id, vla, task)
        result_queue.put(result)

class MultiProcessEvalRunner:
    def __init__(self, num_workers, remote_config):
        self.task_queue = mp.Queue()
        self.result_queue = mp.Queue()
        self.workers = []
        for i in range(num_workers):
            p = mp.Process(target=env_worker,
                           args=(i, self.task_queue, self.result_queue, remote_config))
            p.start()
            self.workers.append(p)
```

**优点**：
- 完全避免 GIL 和线程安全问题
- 每个进程的 env 完全隔离
- 与当前 torchrun 架构类似，改造成本低

**缺点**：
- 每个进程需要独立的 gRPC 连接（开销小）
- 进程间通信比线程间通信慢
- 内存占用更高（每个进程独立的 env + dataset）

### 3.3 方案 C：异步 Env + Server 端 Batching（高级）

**核心思路**：Client 端异步发送多个 env 的推理请求，Server 端收集一定时间窗口内的请求组成 batch，一次 GPU forward 处理多个请求。

```
Client:
  Env 0 → request 0 ──┐
  Env 1 → request 1 ──┼──► Server: collect & batch
  Env 2 → request 2 ──┘         ↓
                           GPU forward (batch_size=3)
                                 ↓
                           split & return
  Env 0 ← response 0 ◄──┤
  Env 1 ← response 1 ◄──┤
  Env 2 ← response 2 ◄──┘
```

**优点**：
- GPU 利用率最高（batch inference 比串行 inference 快得多）
- 吞吐量显著提升

**缺点**：
- 实现复杂度高
- 需要修改 server 端（batch 收集、超时机制、请求路由）
- 不同 env 的输入 shape 需要一致或支持 padding
- 延迟可能增加（需等待凑够 batch 或超时）

## 4. 推荐方案与实施路线

### 第一阶段：方案 B（多进程，快速验证）

最小改动，快速验证多 env 并发查询的可行性和吞吐量提升。

**改动点**：
1. 新建 `MultiEnvEvalRunner`（或修改 `LiberoEvalRunner`）
2. 用 `mp.Process` 启动 N 个 env worker，每个创建自己的 `RemoteVLA` 连接
3. 主进程分配任务，收集结果

**预期收益**：
- 当 env 数量 ≤ GPU 推理并发能力时，吞吐量线性提升
- 例如：2 个 env，推理 530ms，env.step 20ms → 理论利用率从 50% 提升到 ~95%

### 第二阶段：方案 A（多线程，降低资源开销）

验证 Libero env 线程安全性后，切换到多线程方案以降低内存开销。

**改动点**：
1. 将 `mp.Process` 替换为 `ThreadPoolExecutor`
2. 共享一个 `RemoteVLA` 连接
3. 添加线程安全的结果收集

### 第三阶段：方案 C（Server 端 Batching，最大化吞吐）

当 env 数量 > 4 时，串行推理成为瓶颈，此时添加 server 端 dynamic batching。

**改动点**：
1. Server 端新增 `BatchCollector`：收集时间窗口内的请求
2. 新增 `BatchTensorBatchInfer` RPC（或复用现有 RPC + server 内部 batching）
3. Client 端改用 async gRPC（`grpc.aio`）

## 5. 吞吐量估算

### 假设
- 模型推理：530ms（串行，不支持 batch）
- env.step + 预处理 + 序列化：30ms
- 网络 RTT：6ms

### 单 env（当前）
```
Timeline:  [infer 530ms][step 30ms][infer 530ms][step 30ms]...
Throughput: 1 / 560ms ≈ 1.8 steps/s
GPU utilization: 530/560 ≈ 95%
```

### 2 env（方案 A/B，无 server batching）
```
Timeline:
  Env 0: [infer 530ms][step 30ms][infer 530ms     ][step 30ms]...
  Env 1:       [step 30ms][infer 530ms][step 30ms ][infer 530ms]...
  GPU:   [req0 530][req1 530][req0 530][req1 530]...

Throughput: 2 / 1060ms ≈ 1.9 steps/s (每 env)
           总计 ≈ 3.8 steps/s
GPU utilization: ~100% (推理之间几乎无间隔)
```

**注意**：因为 GPU 推理是串行的（单 GPU 无法同时跑两个 forward），2 个 env 的总吞吐并不是 2 倍。真正的收益在于：
- GPU 利用率从 95% → ~100%（消除了 env.step 期间的 GPU 空闲）
- **eval 总时间缩短**：每个 env 的 episode 可以重叠执行

### N env + Server Batching（方案 C）
```
假设 batch_size=4 的推理时间 ≈ 600ms（比 4x530 快很多）

Throughput: 4 / 600ms ≈ 6.7 steps/s
GPU utilization: ~100%
加速比: 6.7 / 1.8 ≈ 3.7x
```

## 6. 实施计划

### 6.1 Phase 1 具体步骤

| 步骤 | 内容 | 文件 |
|------|------|------|
| 1 | 新建 `MultiEnvLiberoEvalRunner` 类 | `fluxvla/engines/runners/multi_env_libero_eval_runner.py` |
| 2 | 实现 env worker 进程 | 同上 |
| 3 | 实现任务分配器（round-robin） | 同上 |
| 4 | 添加 config 参数 `num_envs` | config |
| 5 | 验证：2 env 查询同一 server | libero eval |
| 6 | 验证：4 env 查询同一 server | libero eval |
| 7 | 对比 throughput 和成功率 | 性能报告 |

### 6.2 Config 扩展

```python
eval = dict(
    type='MultiEnvLiberoEvalRunner',   # 或 'LiberoEvalRunner'
    num_envs=4,                         # 并发 env 数量
    remote_inference=dict(
        enabled=True,
        host='192.168.1.100',
        port=50051,
        timeout_s=30.0,
    ),
    # ... 其余配置不变
)
```

### 6.3 Server 端 gRPC 配置建议

| env 数量 | 建议 `max_workers` | 说明 |
|----------|-------------------|------|
| 1-2 | 4 (默认) | 足够处理并发请求 |
| 3-4 | 8 | 避免请求排队 |
| 5+ | 考虑 server batching | 串行推理成为瓶颈 |

## 7. 风险与注意事项

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| Libero env 不线程安全 | 方案 A 不可用 | 使用方案 B（多进程） |
| GPU 显存不足（多请求排队） | OOM | Server 端限制并发推理数（信号量） |
| env.step 比预期慢 | 吞吐提升有限 | 增加 env 数量以保证 GPU 始终有请求 |
| gRPC ThreadPool 饱和 | 请求超时 | 增加 `max_workers`，使用异步 gRPC |
| 不同 env 的 episode 长度不同 | 负载不均 | 任务队列动态分配 |

## 8. 总结

| 阶段 | 方案 | 复杂度 | 预期收益 | 适用场景 |
|------|------|--------|---------|---------|
| Phase 1 | 多进程 env | 低 | GPU 利用率 95% → 100% | 2-4 个 env |
| Phase 2 | 多线程 env | 中 | 降低内存开销 | 验证线程安全后 |
| Phase 3 | Server batching | 高 | 吞吐量 3-4x 提升 | 5+ 个 env |

**建议从 Phase 1 开始**，因为改动最小且完全复用现有 `RemoteVLA` + gRPC server 基础设施。
