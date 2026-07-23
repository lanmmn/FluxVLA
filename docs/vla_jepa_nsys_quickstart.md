# VLA-JEPA 训练 NVTX / Nsight Systems 简明说明

本文简要记录 VLA-JEPA 训练性能采集新增了什么、如何运行，以及如何查看
Nsight Systems 分析结果。完整原理和手工命令见
[vla_jepa_nsys_nvtx_profiling.md](vla_jepa_nsys_nvtx_profiling.md)。

## 新增内容

### 1. 训练阶段 NVTX 标记

训练 runner 会为以下阶段添加 NVTX range：

- optimizer step：`OPTIMIZER_STEP_XXXXXX`
- 每个梯度累积 micro step：`MICRO_XX/DATA_WAIT`、`MICRO_XX/TRAIN_SYNC_X`
- 训练子阶段：`H2D_AND_PREPARE`、`FORWARD`、`BACKWARD`、`GRAD_CLIP`
- 优化器阶段：`ADAMW_STEP`、`LR_SCHEDULER`、`ZERO_GRAD`

runner 通过以下环境变量控制 `cudaProfilerStart/Stop`：

- `FLUXVLA_NSYS_START_STEP`：从哪个 optimizer step 开始采集
- `FLUXVLA_NSYS_NUM_STEPS`：采集多少个 optimizer step
- `FLUXVLA_NSYS_ALL_RANKS`：是否让所有 rank 触发 profiler，默认只由 rank 0 触发

### 2. 一体化运行脚本

新增 `scripts/profile_vla_jepa_nsys.sh`，支持三种模式：

| 模式 | 作用 |
| --- | --- |
| `all` | 执行短训练采集，并自动分析生成的报告 |
| `profile` | 只执行训练采集 |
| `analyze` | 分析已有 `.nsys-rep` 文件或目录 |

脚本会自动完成：

- 将 `FLUXVLA_NSYS_*` 变量传入 `torchrun` 训练进程；
- 使用 `nsys profile` 跟踪 CUDA、NVTX、NCCL 和 OS Runtime；
- 生成 `.nsys-rep` 与 `.sqlite`；
- 生成 CUDA/NVTX/NCCL CSV 汇总；
- 执行 GPU 利用率、GPU gap、CUDA 同步和 memcpy 诊断；
- 生成便于快速查看的 `overview.txt`；
- 复用已有 SQLite、CSV 和规则结果，支持中断后继续分析。

## 快速使用

### 一键采集并分析

```bash
cd /root/projects/fluxvla
bash scripts/profile_vla_jepa_nsys.sh all
```

默认行为：

- 使用 8 张 GPU；
- 每卡 batch size 为 32；
- gradient accumulation 为 1；
- 先运行 10 个 optimizer step 进行 warm-up；
- 采集 optimizer step 10、11、12；
- 再运行 1 个收尾 step 后退出；
- 关闭 W&B；
- 输出到 `/tmp/fluxvla_nsys_vla_jepa`。

### 调整采集参数

```bash
PER_DEVICE_BATCH_SIZE=8 \
GRAD_ACCUMULATION_STEPS=4 \
START_STEP=10 \
NUM_STEPS=2 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash scripts/profile_vla_jepa_nsys.sh all \
  train_dataloader.per_device_num_workers=8
```

环境变量用于控制 profiling，脚本末尾参数会直接追加到训练命令的
`--cfg-options`。

### 仅采集

```bash
bash scripts/profile_vla_jepa_nsys.sh profile
```

### 分析已有报告

```bash
bash scripts/profile_vla_jepa_nsys.sh analyze /path/to/run.nsys-rep
```

传入目录时，会分析该目录第一层的所有 `.nsys-rep`：

```bash
bash scripts/profile_vla_jepa_nsys.sh analyze /path/to/report-directory
```

如果只需要 CSV，不运行耗时较长的自动诊断规则：

```bash
RUN_ANALYZE_RULES=0 \
bash scripts/profile_vla_jepa_nsys.sh analyze /path/to/run.nsys-rep
```

### 只检查命令、不启动训练

```bash
DRY_RUN=1 bash scripts/profile_vla_jepa_nsys.sh profile
```

## CPU 和 I/O 记录范围

默认模式不只是记录 GPU，还会记录：

- `OPTIMIZER_STEP` 及各训练子阶段的 CPU-side wall time；
- 主训练进程等待下一批数据的 `DATA_WAIT`；
- batch 搬运与准备的 `H2D_AND_PREPARE`；
- `osrt_sum.csv` 中的 `read`、`poll`、`sem_wait` 等 OS Runtime 调用；
- CUDA API 中的 stream/event synchronization；
- GPU kernel、NCCL、GPU gap 和 GPU 利用率。

默认模式为了控制开销，关闭了 CPU sampling、context-switch trace 和文件路径
记录。因此它能判断“主进程是否在等数据、文件 API 总时间是否明显”，但不能定位
具体 CPU 热函数、阻塞线程调用栈或读取了哪个文件。

需要深入检查 CPU、DataLoader worker 或文件 I/O 时，建议只采集 1-2 个 step：

```bash
CPU_IO_TRACE=1 \
NUM_STEPS=1 \
bash scripts/profile_vla_jepa_nsys.sh all
```

该模式额外启用：

- native CPU sampling：`--sample=process-tree`；
- CPU context switch：`--cpuctxsw=process-tree`；
- Python sampling（默认 100 Hz）；
- CUDA API Python backtrace，用于定位同步调用来源；
- 文件访问路径：`--osrt-file-access=true`。
- process-tree syscall trace：`--syscall=process-tree`。

深度模式会增加运行开销和报告大小，应保持较短 capture window，并优先在 Nsight
Systems GUI 中查看 CPU sampling、线程状态、文件路径和 GPU timeline。若还需要
PyTorch autograd 级 NVTX，可额外设置 `NSYS_PYTORCH_TRACE=autograd-nvtx`，但会
显著增大报告。

## 常用参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `OUTPUT_DIR` | `/tmp/fluxvla_nsys_vla_jepa` | Nsight 报告输出目录 |
| `WORK_DIR` | `/tmp/vla_jepa_nsys_profile_run` | 短训练工作目录 |
| `NPROC_PER_NODE` | `8` | GPU/rank 数量 |
| `PER_DEVICE_BATCH_SIZE` | `32` | 每张 GPU 的 batch size |
| `GRAD_ACCUMULATION_STEPS` | `1` | 梯度累积次数 |
| `START_STEP` | `10` | capture 起始 optimizer step |
| `NUM_STEPS` | `3` | capture 的 optimizer step 数量 |
| `ALL_RANKS` | `0` | 是否由所有 rank 调用 profiler API |
| `TOP_N` | `20` | `overview.txt` 中每类结果的最大行数 |
| `FORCE_EXPORT` | `0` | 强制重新导出 SQLite |
| `FORCE_STATS` | `0` | 强制重新生成 CSV 和规则结果 |
| `RUN_ANALYZE_RULES` | `1` | 是否运行自动诊断规则 |
| `CPU_IO_TRACE` | `0` | 是否启用 CPU sampling、context switch 和文件路径记录 |
| `NSYS_PYTORCH_TRACE` | `none` | 可选 PyTorch autograd/functions NVTX 跟踪 |

## 输出结果

每个 `.nsys-rep` 会对应一个 `<报告名>_analysis` 目录，重点文件包括：

| 文件 | 用途 |
| --- | --- |
| `overview.txt` | 汇总主要统计和自动诊断，建议首先查看 |
| `optimizer_step_breakdown.txt` | 每个 optimizer step 的 CPU-side 阶段拆分 |
| `nvtx_training_phases_cpu.csv` | 各训练阶段的 CPU-side NVTX 时间 |
| `nvtx_training_phases_gpu.csv` | NVTX range 对应的 GPU projection |
| `cuda_gpu_kern_sum.csv` | CUDA kernel 时间汇总 |
| `cuda_gpu_kern_by_nvtx.csv` | 按最内层 NVTX range 分组的 kernel |
| `cuda_kern_exec_sum.csv` | CUDA API、queue 和 kernel 执行时间 |
| `nccl_kernels.csv` | NCCL kernel 汇总 |
| `osrt_sum.csv` | OS Runtime API 汇总 |
| `syscall_sum.csv` | 深度模式下的 Linux syscall 汇总 |
| `osrt_file_io.csv` | 文件读取、映射、关闭等 API 汇总 |
| `osrt_blocking.csv` | poll、semaphore、condition 等阻塞 API 汇总 |
| `rule_gpu_gaps.txt` | 长 GPU 空闲区间诊断 |
| `rule_gpu_time_util.txt` | 低 GPU 利用率区间诊断 |
| `rule_cuda_api_sync.txt` | 阻塞式 CUDA 同步调用 |

## 查看结果时的注意事项

1. `nvtx_pushpop_sum` 是 CPU-side range 时间；嵌套 range 会重复计时，不能直接相加。
2. `nvtx_gpu_proj_sum` 只统计在 NVTX range 内发起的 GPU 操作。PyTorch
   autograd/FSDP 工作线程不一定继承 Python 线程上的 `BACKWARD` range，因此
   `BACKWARD` projection 很小不代表反向 GPU 开销很小，应结合
   `OPTIMIZER_STEP`、kernel 汇总和时间线判断。
3. NCCL kernel 可能与计算重叠，NCCL 累计时间不能直接当作纯通信损失。
4. 如果出现 `No reports were generated`，优先确认训练进程是否收到
   `FLUXVLA_NSYS_START_STEP` 和 `FLUXVLA_NSYS_NUM_STEPS`，以及训练是否执行到
   capture 结束 step。

## `OPTIMIZER_STEP` 拆分与梯度累积对比

这里的 `OPTIMIZER_STEP_XXXXXX` 不是单独的 `optimizer.step()`，而是一个完整的
梯度更新窗口。使用 `GRAD_ACCUMULATION_STEPS=4` 时，它包含：

```text
4 × (DATA_WAIT + H2D_AND_PREPARE + FORWARD + BACKWARD)
+ GRAD_CLIP + ADAMW_STEP + LR_SCHEDULER + ZERO_GRAD
```

此前 `per_device_batch_size=16, grad_accumulation_steps=4` 的 8 卡报告中，
有效 batch 为 512，CPU-side 平均拆分如下（每个 rank-step）：

| 阶段 | 时间 | 占 step wall time |
| --- | ---: | ---: |
| 4 次 `FORWARD` | 5222.3 ms | 48.07% |
| 4 次 `BACKWARD` | 5465.7 ms | 50.30% |
| 4 次 `H2D_AND_PREPARE` | 16.2 ms | 0.15% |
| 4 次 `DATA_WAIT` | 2.8 ms | 0.03% |
| `GRAD_CLIP` | 19.7 ms | 0.18% |
| 真正的 `ADAMW_STEP` | 5.9 ms | 0.05% |
| 其他 Python/range 间隙 | 131.7 ms | 1.21% |
| **完整 `OPTIMIZER_STEP`** | **10865.1 ms** | **100%** |

因此旧基线约 10.9 秒不是 optimizer 本身慢，也不是 DataLoader/磁盘 I/O 慢，主要是
4 次 forward/backward。当前报告中所有进程的 `read` API 聚合约 76.7 ms，且
`DATA_WAIT` 每步仅约 2.8 ms；GPU kernel 汇总中 NCCL AllGather 约占 40.9%，
说明 FSDP 通信和计算是更值得继续分析的方向。注意 OS Runtime 时间会跨 rank 和
线程重叠，不能直接与单个 step wall time相加。

### 新报告：batch 32，梯度累积 1

新报告使用 `per_device_batch_size=32, grad_accumulation_steps=1`，8 卡有效 batch
为 256，与正式配置的 `4 × 8 × 8 = 256` 相同。CPU-side 平均拆分为：

| 阶段 | 时间 | 占 step wall time |
| --- | ---: | ---: |
| 1 次 `FORWARD` | 2244.9 ms | 53.78% |
| 1 次 `BACKWARD` | 1814.3 ms | 43.47% |
| `H2D_AND_PREPARE` | 5.6 ms | 0.13% |
| `DATA_WAIT` | 0.5 ms | 0.01% |
| `GRAD_CLIP` | 21.9 ms | 0.53% |
| 真正的 `ADAMW_STEP` | 7.1 ms | 0.17% |
| 其他 Python/range 间隙 | 78.9 ms | 1.89% |
| **完整 `OPTIMIZER_STEP`** | **4174.0 ms** | **100%** |

capture 边界漏记了 step 10 的 1 个外层 `OPTIMIZER_STEP` range，但内部 forward /
backward 有完整的 24 个 rank-step 实例。Analyzer 会按 8 rank × 3 step 修正归一化，
并在 breakdown 中显示 warning。

两组报告的吞吐对比：

| 设置 | 有效 batch | step 时间 | 估算吞吐 |
| --- | ---: | ---: | ---: |
| batch 16 × 8 卡 × 累积 4 | 512 | 10.865 s | 47.1 samples/s |
| batch 32 × 8 卡 × 累积 1 | 256 | 4.174 s | 61.3 samples/s |

新配置吞吐约提高 30%。处理同样 512 个样本时，新配置需要两个 step，约 8.35 秒，
仍比旧基线的 10.87 秒短约 23%。由于新配置有效 batch 与正式训练配置一致，且本次
显存能够容纳 batch 32，它是更值得继续做长训练验证的设置。

## 如何理解 H2D、AllGather 和 CUDA stream synchronization

### H2D

H2D 是 Host-to-Device，即 CPU 内存到 GPU 显存的数据搬运。当前代码中的
`H2D_AND_PREPARE` 还包括浮点 tensor 转为训练 dtype，以及递归处理 batch 中的
嵌套 tensor。新报告平均每个 optimizer step 约 5.6 ms，仅占 0.13%，不是主要
瓶颈。

### NCCL AllGather

当前训练使用 FSDP `FULL_SHARD`：每个 rank 平时只持有一部分参数，在执行被 FSDP
包装模块的 forward/backward 前，需要通过 AllGather 从其他 rank 收集参数分片，
临时恢复完整参数。梯度随后主要通过 ReduceScatter 重新分片。

AllGather 百分比表示该 kernel 占所有 GPU kernel **累计活跃时间之和**的比例，
不是单个 optimizer step wall time 的比例。不同 GPU、stream 和计算/通信可能
并行，因此不能把该百分比直接乘以 step 时间或与其他百分比当作串行开销相加。

新旧报告对比：

| 通信指标 | batch 16 / 累积 4 | batch 32 / 累积 1 |
| --- | ---: | ---: |
| AllGather 实例 | 10,272 | 2,568 |
| AllGather kernel 占比 | 40.9% | 26.2% |
| ReduceScatter 实例 | 1,008 | 1,008 |
| ReduceScatter kernel 占比 | 5.7% | 14.6% |

累积 1 将 AllGather 次数降低约 75%，因为每个 optimizer step 只执行一个
micro-batch。ReduceScatter 次数不变，是因为累积 4 时前三个 micro-step 使用
`no_sync()`，原本也只在最后一个 micro-step 同步梯度。新报告通信占比仍高，
AllGather + ReduceScatter 合计约 40.8%，但参数 AllGather 压力明显低于旧基线。

### CUDA stream synchronization

`cudaStreamSynchronize` 表示 CPU 线程主动等待某条 CUDA stream 上此前提交的工作
完成。新报告最长等待约 574.6 ms，旧基线为 422.9 ms。等待内容可能包括计算
kernel、NCCL 通信或其他 GPU 工作。

它不表示同步 API 自己执行了 574.6 ms，也不一定意味着删除同步就能节省 574.6
ms；同步点经常只是把此前异步提交的 GPU 工作暴露成 CPU 等待。只有当时间线显示
GPU 同时空闲，或者同步阻止了本来可以重叠的 CPU/GPU 工作时，才可判断为可优化的
同步瓶颈。

新报告还检测到 GPU 5 在 capture 开始附近有一次 503.4 ms 的 GPU gap，并有若干
417-1670 ms、平均利用率低于 50% 的窗口。这说明部分时段存在 rank/stream 不平衡，
但唯一超过 500 ms 的 gap 位于 capture 初始阶段，可能受到 profiler 边界影响，
不能直接当作稳态训练瓶颈。

默认报告没有同步调用的 Python 栈，不能准确断言其来源。可用短窗口深度采集定位：

```bash
CPU_IO_TRACE=1 \
NUM_STEPS=1 \
bash scripts/profile_vla_jepa_nsys.sh all
```

深度模式会记录 CUDA API Python backtrace、CPU sampling 和 context switch，可在
Nsight Systems GUI 中从长同步事件回溯调用栈，并检查同一时间 GPU/NCCL
timeline。

## 验证情况

该流程已在 8 张 A100 上完成端到端验证：成功捕获 optimizer step 10-12，
生成 `.nsys-rep`、SQLite、训练阶段 CSV、NCCL 汇总、自动诊断和
`overview.txt`，稳定版本 analyzer 退出码为 0。
