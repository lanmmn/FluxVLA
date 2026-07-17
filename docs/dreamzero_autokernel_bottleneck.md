# DreamZero 推理瓶颈分析

本文记录基于 AutoKernel 对 DreamZero 推理热路径做的算子瓶颈和图融合分析。

## 分析范围

这次没有完成完整端到端 DreamZero 推理 profile，因为当时 8 张 GPU 都被另一个训练/调试任务占用，每张卡只剩约 1.6 GB 空闲显存，无法容纳完整 40 层 DreamZero 模型。

因此本次分析改为 profile 一个生产形状的 `CausalWanAttentionBlock`。这个 block 是 DreamZero DiT denoise 的重复热路径。正常推理中，它大约会执行：

```text
num_inference_steps * num_layers = 16 * 40 = 640 次
```

profile 使用的形状和当前生产配置一致：

```text
frame_seqlen = 128
num_frame_per_block = 2
num_action_per_block = 10
num_state_per_block = 1
block sequence length = 2 * 128 + 10 + 1 = 267
hidden dim = 5120
ffn dim = 13824
num heads = 40
head dim = 128
attention backend = FA2
```

AutoKernel wrapper：

```text
/root/code/autokernel/models/dreamzero_dit_block.py
```

主要 profile 产物：

```text
/tmp/dreamzero_dit_block_fa2_profile.json
/root/code/autokernel/workspace/trace.json
```

## 算子瓶颈

从 Chrome trace 中只聚合 CUDA kernel 事件后，瓶颈分布如下：

```text
matmul / GEMM        62.4%
elementwise          14.6%
copy / cat           11.5%
reduce                4.9%
flash attention       4.5%
layernorm             1.1%
other                 1.0%
```

所以当前瓶颈不是 FA2 attention kernel 本身。主要耗时集中在 attention 周边的大量 dense projection 和 FFN：

- self-attention 的 q/k/v/o linear projection。
- I2V cross-attention 的 q/k/v 以及 image k/v projection。
- FFN `5120 -> 13824 -> 5120`。
- modulation、residual、norm、RoPE、dtype/copy 等周边小 kernel。
- blockwise attention 构造 K/V context 时反复 `torch.cat` 和 copy。

Top CUDA kernel 主要是 BF16 GEMM，其次才是 FA2、reduce、cat-copy 和 elementwise。FA2 kernel 能看到，但在这个 block 形状下只占很小一部分。

## 图融合结论

`torch.compile` 在这条路径上遇到两个关键限制：

```text
Graph break due to unsupported builtin flash_attn_2_cuda.PyCapsule.varlen_fwd
Torchinductor does not support code generation for complex operators
```

第一个问题说明 `flash_attn.flash_attn_varlen_func` 是第三方 C++/CUDA PyCapsule 调用。Dynamo 不能把它保留在 compiled graph 里，所以 Inductor 无法优化或融合 FA2 kernel 本身。

第二个问题来自当前 complex-number RoPE 路径。Inductor 仍然可以编译部分周边区域，但 complex operator 会限制 codegen 质量，并产生额外 graph boundary 或较慢的生成代码。

在较小形状 `frame_seqlen=32` 上，单 block compile 后 AutoKernel 报告的 GPU time 从：

```text
eager   : 13.579 ms
compile : 10.376 ms
```

约等于 isolated small block 上 1.31x 的改善。收益主要来自周边 elementwise、reduce、copy 的部分融合。生产形状 `frame_seqlen=128` 的 compiled block 没能在当前机器状态下完成 profile，因为 Inductor Triton autotuning 触发了 OOM。

## 解释

当前 DreamZero 推理主要受以下因素限制：

1. DiT block 内 GEMM-heavy 的 projection 和 FFN。
2. 碎片化的 elementwise、norm、modulation kernel。
3. 通过重复 `cat`/copy 构造 K/V context。
4. FA2 varlen 和 complex RoPE 限制了 Inductor 的图融合能力。

默认 attention 已经走 FA2，而且 FA2 kernel 本身不是当前 profile 中的主瓶颈。单独优化 attention kernel 不太可能显著降低端到端延迟，除非同时减少周边 K/V packing、copy 和 projection 开销。

## 优化优先级

### 1. 启用 cross-attention K/V cache

代码里已经有 `crossattn_cache` 概念，但当前 inference block 路径没有把它继续传给 `self.cross_attn(...)`。

因为 text 和 image context 在一个 episode 内基本固定，每个 denoise step、每个 block 都重复计算 cross-attention K/V projection 是浪费的。

预期收益：

- 减少重复 cross-attention context K/V GEMM。
- 减少一部分 norm 和 elementwise 开销。
- 收益会随 `num_inference_steps * num_layers` 放大。

### 2. 减少 K/V context 的 `cat` 和 copy

blockwise attention 路径会反复用 `torch.cat` 构造 K/V context。CUDA trace 中 copy/cat 成本比较明显。

可选方向：

- 预分配 packed K/V buffer。
- 写入固定 slice，避免每次拼接新 tensor。
- 如果 backend 支持，改成 varlen/paged-style attention metadata。
- 对相同 context range 避免跨 denoise step 重复构造。

### 3. 把 complex RoPE 替换成 real cos/sin RoPE

当前 complex polar RoPE 会阻碍 Inductor codegen。real-valued cos/sin 实现更容易被 Inductor 或自定义 Triton kernel 融合。

预期收益：

- 消除 complex operator warning。
- 提升 compile 友好度。
- 让 RoPE、reshape、copy 更容易融合或替换成自定义 kernel。

### 4. 优先做局部 fusion，而不是整 block compile

整 block `torch.compile` 比较脆弱，因为 FA2 varlen 会 graph break，而且 Inductor autotuning 需要额外显存。更现实的方向是对稳定子图做局部 fusion：

- `norm + modulation + add/mul`。
- FFN 中的 `bias + GELU`。
- real-valued RoPE transform。
- residual add 链。
- K/V packing。

这样不依赖 Inductor 理解 FA2 extension，但仍然能减少 FA2 周围的大量小 kernel。

## 建议的下一步

最高 ROI 的实现目标是 DreamZero inference 的 cross-attention K/V cache，其次是清理 K/V context packing。它们直接对应本次 profile 中测到的 GEMM 和 copy/cat 成本，而且不需要改 FA2 本身。
