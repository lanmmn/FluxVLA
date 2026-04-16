# PI0.5 NVTX Profiling Guide

## 1. 改动概览

在 3 个文件中插入了 NVTX (NVIDIA Tools Extension) 标记，使 Nsight Systems 时间线能按训练阶段分组显示，而非只有一堆未分类的 CUDA kernel。

| 文件 | 改动 |
|------|------|
| `fluxvla/models/vlas/pi0_flowmatching.py` | 模型 forward 路径的细粒度标记 |
| `fluxvla/engines/utils/model_utils.py` | attention kernel 标记 |
| `fluxvla/engines/runners/base_train_runner.py` | 训练循环的 data_load / forward / backward / optimizer 标记 |

所有 NVTX 调用都有 fallback（无 CUDA 环境时自动变成 no-op），**不影响正常训练**。

---

## 2. NVTX 标记层级

### 2.1 训练循环 (`base_train_runner.py`)

```
training_loop
  ├─ data_load              ← DataLoader 取 batch
  └─ _training_step
       ├─ forward           ← 含 autocast
       ├─ backward          ← loss.backward()
       └─ optimizer_step    ← grad clip + optimizer.step() + zero_grad()
```

### 2.2 模型 Forward (`pi0_flowmatching.py`)

```
PI0_forward (range_push/pop 包裹整个 forward)
  │
  ├─ embed_prefix
  │    ├─ vision_backbone+projector   ← SigLIP + projector
  │    └─ llm_embed_tokens            ← Gemma embed_tokens
  │
  ├─ embed_suffix
  │    ├─ state_proj                  ← 状态投影
  │    ├─ time_embedding              ← 时间步正弦编码
  │    └─ action_proj+time_mlp        ← action 投影 + time MLP 融合
  │
  ├─ mask_prepare                     ← 2D/4D mask 或 BlockMask 构建
  │
  ├─ forward_model                    ← 整个 transformer 前向
  │    ├─ transformer_layer_0
  │    │    ├─ layernorm+qkv_proj[0]  ← prefix 侧 (backbone)
  │    │    ├─ layernorm+qkv_proj[1]  ← suffix 侧 (expert)
  │    │    ├─ rope_layer_0           ← RoPE 位置编码
  │    │    ├─ attention_layer_0       ← eager_attention / flex_attention
  │    │    ├─ o_proj+ffn[0]_layer_0  ← prefix O_proj + FFN
  │    │    └─ o_proj+ffn[1]_layer_0  ← suffix O_proj + FFN
  │    ├─ transformer_layer_1
  │    │    └─ ... (同上)
  │    └─ transformer_layer_17
  │
  ├─ action_out_proj                  ← action 输出投影
  └─ loss_compute                     ← MSE loss
```

### 2.3 Attention Kernel (`model_utils.py`)

```
attention_layer_i
  └─ eager_attention          ← 当前使用的 attention 实现
     或 flex_attention        ← 如果切换到 flex
```

---

## 3. 如何 Profiling

### 3.1 单卡采集（推荐首次使用）

```bash
nsys profile \
  -t cuda,nvtx,osruntime \
  -s none \
  -o pi05_train_profile \
  --force-overwrite=true \
  --cuda-memory-usage=true \
  --duration=60 \
  python scripts/train.py \
    --config configs/pi05/pi05_paligemma_libero_10_full_finetune.py
```

参数说明：

| 参数 | 含义 |
|------|------|
| `-t cuda,nvtx,osruntime` | 跟踪 CUDA kernel、NVTX 标记、OS runtime |
| `-s none` | 不采样 CPU（减小文件体积） |
| `-o pi05_train_profile` | 输出文件名，自动加 `.nsys-rep` 后缀 |
| `--duration=60` | 只采集 60 秒 |
| `--cuda-memory-usage=true` | 记录 GPU 内存使用 |

### 3.2 跳过 Warmup，只采集稳态

```bash
nsys profile \
  -t cuda,nvtx,osruntime \
  -s none \
  -o pi05_steady_state \
  --force-overwrite=true \
  --cuda-memory-usage=true \
  --delay=30 \
  --duration=30 \
  python scripts/train.py \
    --config configs/pi05/pi05_paligemma_libero_10_full_finetune.py
```

`--delay=30`：等 30 秒让 warmup 和 JIT 编译完成
`--duration=30`：采集 30 秒稳态数据

### 3.3 多卡 FSDP 采集

```bash
nsys profile \
  -t cuda,nvtx,osruntime \
  -s none \
  -o pi05_rank%q{RANK} \
  --force-overwrite=true \
  --cuda-memory-usage=true \
  --duration=60 \
  torchrun --nproc_per_node=8 scripts/train.py \
    --config configs/pi05/pi05_paligemma_libero_10_full_finetune.py
```

每个 rank 生成独立文件：`pi05_rank0.nsys-rep` ~ `pi05_rank7.nsys-rep`

### 3.4 代码中控制采集区间（更精确）

如果想精确控制从第几步开始采集、采集几步，可以用 `torch.cuda.profiler` API：

```python
# 在 base_train_runner.py 的 _run_step_based 中
for step in range(max_steps):
    batch = next(dataloader_iter)

    if step == 5:   # warmup 5 步后开始
        torch.cuda.profiler.start()
    if step == 15:  # 采集 10 步后停止
        torch.cuda.profiler.stop()

    loss = self._training_step(batch)
```

配合 nsys 启动时加 `-s cudaProfilerApi`：

```bash
nsys profile -t cuda,nvtx -s cudaProfilerApi -o pi05_selected_steps \
  python scripts/train.py --config ...
```

---

## 4. 下载到本地查看

### 4.1 从服务器下载

```bash
# 在本地机器执行
scp user@server:/path/to/project/pi05_train_profile.nsys-rep ./
```

如果是多卡：

```bash
scp user@server:'/path/to/project/pi05_rank*.nsys-rep' ./
```

### 4.2 安装 Nsight Systems GUI

下载地址：https://developer.nvidia.com/nsight-systems

支持 Windows / Linux / macOS（查看端不需要 GPU）。

### 4.3 打开文件

```bash
nsys-ui pi05_train_profile.nsys-rep
```

或在 GUI 中 File → Open。

---

## 5. 时间线阅读指南

### 5.1 关键行

| 行 | 看什么 |
|----|--------|
| **NVTX** | 带 label 的彩色区间，就是你插入的标记 |
| **CUDA HW** | GPU kernel 执行时间 |
| **CUDA API** | CPU 侧发起 kernel 调用 |
| **OS Runtime** | 线程/锁/IO 等待 |

### 5.2 诊断 Checklist

| 现象 | 诊断 |
|------|------|
| NVTX 标记间有大段 GPU 空白 | CPU 瓶颈或数据加载慢 |
| `data_load` 很长 | DataLoader IO/预处理慢，考虑加 `num_workers` |
| `attention_layer` 占 forward 大头 | Attention 是瓶颈，验证 eager vs SDPA 差异 |
| `backward` 比 `forward` 长很多 | 正常（backward 计算量通常 > forward） |
| `o_proj+ffn[0]` 远长于 `o_proj+ffn[1]` | prefix 692 token vs suffix 11 token，符合预期 |
| `mask_prepare` 显眼 | 4D mask 构建耗时，FlexAttention 的 BlockMask 可消除 |
| `vision_backbone+projector` 很长 | SigLIP 冻结但仍占 forward 时间 |

### 5.3 量化某段耗时

在 GUI 中：
1. 右键 NVTX 区间 → "Show in Events View"
2. 或拖拽选区 → 底部显示选中时长

---

## 6. 对比实验

可通过切换 `attention_implementation` 对比不同 attention 后端：

```bash
# Eager (当前默认)
nsys profile -t cuda,nvtx -s none -o pi05_eager --duration=30 \
  python scripts/train.py --config configs/pi05/pi05_paligemma_libero_10_full_finetune.py

# Flex (需配置 attention_implementation='flex')
nsys profile -t cuda,nvtx -s none -o pi05_flex --duration=30 \
  python scripts/train.py --config configs/pi05/pi05_paligemma_libero_10_full_finetune.py \
  --cfg-options model.attention_implementation=flex
```

然后在 Nsight Systems 中对比 `attention_layer` 区间在不同后端下的耗时。
