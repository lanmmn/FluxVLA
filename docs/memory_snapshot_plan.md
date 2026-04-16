# PI0.5 显存 Snapshot 方案

## 1. 目标

在 PI0.5 训练过程中使用 `torch.cuda.memory._snapshot()` 记录显存分配时间线，
导出 pickle 文件后上传到 [PyTorch Memory Viz](https://pytorch.org/memory_viz) 可视化，
定位哪些 tensor 占显存最多、为什么 B=8 会压满 80GB。

## 2. 方案设计

### 2.1 采集策略

只在训练稳态采集，避免 warmup 阶段的 JIT 编译和初始化干扰：

```
Step 0-2:    warmup（不采集）
Step 3:      开始记录 memory history
Step 5:      导出 snapshot（覆盖一个完整的 forward + backward + optimizer_step）
Step 5+:     停止记录，恢复正常训练
```

为什么只采 2-3 步：
- `memory._record_memory_history()` 开启后每个 allocation 都会记录调用栈
- 记录本身有额外开销（~10-20% 显存和速度）
- 2-3 步足以覆盖 fwd+bwd+opt 完整周期

### 2.2 插入位置

**文件：`fluxvla/engines/runners/base_train_runner.py`**

在 `_run_step_based` 和 `_run_epoch_based` 的训练循环中，于 `_training_step` 调用前后控制 snapshot：

```python
# _run_step_based 中（现有代码结构）:
while self.metric.global_step < self.max_steps:
    # ... 取 batch ...

    # ---- Memory Snapshot 控制 ----
    step = self.metric.global_step
    if step == 3:
        torch.cuda.memory._record_memory_history(
            max_entries=100000,
            stacks="python",          # 记录 Python 调用栈
        )
    if step == 5:
        snapshot = torch.cuda.memory._snapshot()
        from pickle import dump
        snapshot_path = os.path.join(self.args.work_dir, "memory_snapshot.pickle")
        dump(snapshot, open(snapshot_path, "wb"))
        torch.cuda.memory._record_memory_history(enabled=None)
        overwatch.info(f"Memory snapshot saved to {snapshot_path}")
    # ---- END Memory Snapshot ----

    loss = self._training_step(batch)
```

不在 `_training_step` 内部插入，因为：
- 记录粒度是 step 级别，不需要在 fwd/bwd/opt 内部再嵌套
- 避免对训练代码的侵入

### 2.3 采集参数

```python
torch.cuda.memory._record_memory_history(
    max_entries=100000,    # 最多记录 10 万个 allocation 事件
    stacks="python",       # 记录 Python 层调用栈（比 "c++" 更易读）
)
```

- `stacks="python"` 而非 `"all"`（包含 C++）：Python 栈已够定位，C++ 栈会让 pickle 文件大很多
- `max_entries=100000`：2-3 步的 allocation 事件约 5-10 万条，10 万条上限足够

### 2.4 多卡场景

只有 rank 0 采集，避免多卡同时写文件：

```python
if overwatch.is_rank_zero():
    if step == 3:
        torch.cuda.memory._record_memory_history(...)
    if step == 5:
        snapshot = torch.cuda.memory._snapshot()
        dump(snapshot, open(...))
        torch.cuda.memory._record_memory_history(enabled=None)
```

### 2.5 环境变量控制

通过环境变量 `FLUXVLA_MEMORY_SNAPSHOT` 控制是否开启，默认不开启：

```python
MEMORY_SNAPSHOT_ENABLED = os.environ.get("FLUXVLA_MEMORY_SNAPSHOT", "0") == "1"
MEMORY_SNAPSHOT_START = int(os.environ.get("FLUXVLA_MEMORY_SNAPSHOT_START", "3"))
MEMORY_SNAPSHOT_END = int(os.environ.get("FLUXVLA_MEMORY_SNAPSHOT_END", "5"))
```

这样不需要改代码就能开关：

```bash
# 开启显存 snapshot
FLUXVLA_MEMORY_SNAPSHOT=1 python scripts/train.py --config ...

# 自定义采集区间
FLUXVLA_MEMORY_SNAPSHOT=1 FLUXVLA_MEMORY_SNAPSHOT_START=5 FLUXVLA_MEMORY_SNAPSHOT_END=8 \
  python scripts/train.py --config ...
```

## 3. 改动清单

| 文件 | 改动 |
|------|------|
| `fluxvla/engines/runners/base_train_runner.py` | 在 `__init__` 中读取环境变量；在 `_run_step_based` 和 `_run_epoch_based` 中插入 snapshot 控制逻辑 |

不需要改模型代码、配置文件或其他文件。

## 4. 查看结果

### 4.1 下载 pickle 文件

```bash
scp user@server:/path/to/work_dir/memory_snapshot.pickle ./
```

### 4.2 打开可视化

浏览器访问 https://pytorch.org/memory_viz ，选择 "GPU Memory Visualizer"，
拖入 `memory_snapshot.pickle` 文件。

### 4.3 看什么

| 视图 | 关注点 |
|------|--------|
| **Active Memory Timeline** | 时间线上哪些 tensor 一直占着显存（长条 = 常驻，短条 = 临时） |
| **Allocation Stack** | 点击某个分配块，看调用栈，定位是哪行代码创建的 |
| **OOM 前的最后分配** | 如果发生 OOM，看 crash 前最后几个大块分配是什么 |
| **按大小排序** | 找最大的常驻 tensor，通常是优化器状态、模型参数、激活值 |

### 4.4 预期会看到的显存分布

```
时间线示意（一个 step）:

──────────────────────────────────────────────────→ 时间
███ AdamW optimizer states ████████████████████████  ← 常驻 ~39GB
███ Model params (BF16) ███████████████████████████  ← 常驻 ~6.5GB
███ Gradients (BF16) ██████████████████████████████  ← 常驻 ~6.5GB
     ▓▓▓▓▓ forward activations ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓     ← 临时 ~20GB
     ▓▓▓▓ backward ▓▓▓▓▓▓▓▓▓▓▓▓▓▓                  ← 临时
              ▒▒ optimizer step ▒▒                   ← 临时
```

## 5. 注意事项

1. **开启 snapshot 后训练会变慢**（约 10-20%），因为每个 allocation 都要记录调用栈
2. **pickle 文件可能较大**（2-3 步约 50-200MB），确保磁盘空间足够
3. **snapshot 只记录 CUDA 显存**，不包含 CPU 内存
4. **采集完自动停止记录**，后续 step 不再有额外开销
5. **与 NVTX 标记不冲突**，两者可以同时使用
