# Transformer 基础组件与变体学习指南

## 1. 基础组件概览

### 1.1 Linear Layer

最基本的 `y = Wx + b`。在 Transformer 中无处不在：

- Q/K/V Projection（`q_proj`, `k_proj`, `v_proj`）
- Output Projection（`o_proj`）
- FFN 的 up/down/gate projection
- 各种 head（action_out_proj 等）

变体较少，核心是理解它在不同位置的**语义角色**。

______________________________________________________________________

### 1.2 Normalization

| 变体          | 典型用途               | 核心公式 / 区别                                  | 本 repo 对应                          |
| ------------- | ---------------------- | ------------------------------------------------ | ------------------------------------- |
| **LayerNorm** | Transformer 标配       | `(x - mean) / std * γ + β`，对最后一维归一化     | `nn.LayerNorm`                        |
| **RMSNorm**   | Llama / Gemma          | 去掉 mean centering，只做 `x / rms(x) * γ`，更快 | `GemmaRMSNorm`                        |
| **AdaLN**     | DiT (Gr00t)            | LayerNorm + 外部条件（timestep）生成 scale/shift | `cross_attention_dit.py:AdaLayerNorm` |
| **AdaRMS**    | Pi0.5 expert 侧        | RMSNorm + 外部条件化                             | `condition_gemma.py`                  |
| **GroupNorm** | CNN / UNet / Diffusion | 按 channel group 归一化                          | —                                     |
| **BatchNorm** | 传统 CNN               | 按 batch 维归一化，训练/推理行为不一致           | —                                     |

**为什么 LLM 时代从 LayerNorm 转向 RMSNorm？**

- RMSNorm 省去了 mean 计算，训练速度快 ~5-10%
- 实验表明去掉 centering 对性能几乎无影响
- Llama 率先采用后成为事实标准

______________________________________________________________________

### 1.3 Attention

| 变体                              | 核心区别                                           | 典型用途                    |
| --------------------------------- | -------------------------------------------------- | --------------------------- |
| **Self-Attention**                | Q/K/V 来自同一序列                                 | Transformer encoder/decoder |
| **Cross-Attention**               | Q 来自一边，K/V 来自另一边                         | Gr00t DiT 的偶数层          |
| **Multi-Head Attention (MHA)**    | 标准多头，Q/K/V head 数相同                        | GPT-2、BERT                 |
| **Multi-Query Attention (MQA)**   | K/V 只有 1 个 head，Q 多个 head                    | PaLM、推理场景              |
| **Grouped-Query Attention (GQA)** | K/V 共享 group heads，介于 MHA 和 MQA 之间         | Llama 2/3、Gemma            |
| **Joint Attention**               | 多个模态拼接后在同一矩阵中 attend，mask 控制可见性 | Pi0.5                       |

**本 repo 中的 attention 实现：**

```
eager_attention_forward  → 朴素实现，model_utils.py
  Q @ K^T / sqrt(d) + mask → softmax → @ V

FlexAttention (计划中)   → PyTorch 原生，支持 mask_mod
FlashAttention 2         → 不适用（Pi0.5 mask 不是纯 causal）
Triton kernels           → 自定义推理加速，attention_triton_ops.py
```

**GQA 在 Gemma 中的体现：**

```
Gemma: num_attention_heads=8, num_key_value_heads=1
含义：8 个 Q head 共享 1 个 K/V head
效果：KV cache 缩小 8x，推理显存大幅降低
```

______________________________________________________________________

### 1.4 Position Encoding

| 变体                     | 核心思想                                      | 用在哪                           |
| ------------------------ | --------------------------------------------- | -------------------------------- |
| **Sinusoidal (Vaswani)** | 固定 sin/cos，不可学习                        | 原始 Transformer、Pi0.5 timestep |
| **Learned Embedding**    | `nn.Embedding(max_len, dim)`                  | GPT-2、Gr00t action pos          |
| **RoPE**                 | 旋转位置编码，作用在 Q/K 上                   | Llama、Gemma、Pi0.5              |
| **ALiBi**                | 不加 pos emb，直接在 attention score 上加偏置 | MPT、BLOOM                       |

**RoPE 为什么成为主流？**

- 天然支持相对位置
- 推理时可外推到训练时未见过的长度
- 不需要额外参数

**本 repo 中：**

```python
# model_utils.py
cos, sin = self.llm_backbone.rotary_emb(dummy_tensor, position_ids)
query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
```

______________________________________________________________________

### 1.5 Activation Function

| 变体             | 公式                                  | 说明                        |
| ---------------- | ------------------------------------- | --------------------------- |
| **ReLU**         | `max(0, x)`                           | 经典，但有 dead neuron 问题 |
| **GELU**         | `x * Φ(x)`（Φ 是标准正态 CDF）        | BERT / GPT 标配             |
| **SiLU (Swish)** | `x * sigmoid(x)`                      | Llama / Gemma / Pi0.5       |
| **GeGLU**        | `GELU(xW₁) ⊙ xW₂` (Gated Linear Unit) | PaLM / Gemma FFN            |

**Gemma FFN 结构（GeGLU 变体）：**

```python
# 典型 Gemma MLP
gate = self.gate_proj(x)      # W_gate
up   = self.up_proj(x)        # W_up
x    = F.silu(gate) * up      # SiLU-gated
x    = self.down_proj(x)      # W_down
```

这是 3 个 Linear 组成的 FFN，比原始 Transformer 的 2 层 FFN 更强。

______________________________________________________________________

### 1.6 Embedding

| 变体                   | 说明                                                                            |
| ---------------------- | ------------------------------------------------------------------------------- |
| **Token Embedding**    | `nn.Embedding(vocab_size, dim)`，离散 token 到连续向量                          |
| **Patch Embedding**    | `nn.Conv2d(3, dim, patch_size, stride=patch_size)`，ViT 把图像切 patch          |
| **Timestep Embedding** | Sinusoidal → MLP，把连续 timestep 映射到高维，Diffusion 用                      |
| **Learnable Tokens**   | `nn.Embedding(N, dim).weight`，可学习的"工作区"token，如 Gr00t 的 future_tokens |

______________________________________________________________________

## 2. 推荐学习路线

### 第一阶段：打地基（1-2 周）

**从零手写 Transformer** 是最有效的学习方式。

#### 核心资源

1. **Andrej Karpathy - "Let's build GPT from scratch"**

   - YouTube 搜 "Andrej Karpathy GPT"，约 2 小时
   - 从零写 self-attention、multi-head、layernorm、FFN
   - 一个视频理清 80% 的基础组件

2. **The Illustrated Transformer** (Jay Alammar)

   - 搜 "jalammar illustrated transformer"
   - 图解每一步数据流，建立直觉

3. **Attention Is All You Need** (Vaswani et al., 2017)

   - 原始论文，定义了 Transformer 的一切基础
   - 重点看 Section 3 (Model Architecture)

#### 学完应该能回答

- 为什么要除以 `sqrt(d_k)`？
- Multi-Head 的每个 head 在做什么？
- FFN 在 Transformer 中扮演什么角色？
- Layer Norm 为什么放在 attention 前面（Pre-LN）而不是后面（Post-LN）？

______________________________________________________________________

### 第二阶段：读源码理解变体（1-2 周）

**直接读 HuggingFace transformers 的模型实现**，它们注释详尽，结构清晰。

#### 推荐阅读顺序

```
1. GPT-2
   文件: transformers/models/gpt2/modeling_gpt2.py
   学到: 标准 MHA + LayerNorm + GELU FFN
   约 800 行，最经典的 decoder-only 实现

2. Llama
   文件: transformers/models/llama/modeling_llama.py
   学到: RMSNorm + RoPE + GQA + SiLU + GeGLU FFN
   现代 LLM 的事实标准架构

3. Gemma
   文件: transformers/models/gemma/modeling_gemma.py
   学到: 和 Llama 类似但有细节差异（如 embedding 缩放）
   本 repo Pi0.5 的 backbone

4. PaliGemma
   学到: Vision encoder (SigLIP) + LLM 如何拼接
   理解多模态 token 拼接的方式
```

#### 阅读技巧

- 重点看 `XxxAttention` 类和 `XxxMLP` 类
- 忽略 `XxxForCausalLM` 外层 wrapper，聚焦 `XxxModel` 和 `XxxDecoderLayer`
- 对比 GPT-2 和 Llama 的差异，就能理解所有"现代化改进"

______________________________________________________________________

### 第三阶段：理解 Diffusion / Flow Matching 变体（按需）

#### DiT（Gr00t 的理论基础）

- **论文**: "Scalable Diffusion Models with Transformers" (Peebles & Xie, 2023)
- 解释了为什么 AdaLN-Zero 比 cross-attention 更适合 diffusion
- 本 repo `cross_attention_dit.py` 是其实现

#### Flow Matching（Pi0 / Pi0.5 / Gr00t 都用）

- **论文**: "Flow Matching for Generative Modeling" (Lipman et al., 2023)
- 理解 velocity field `v_t`、Euler 积分、和 DDPM 的区别
- 本 repo 的 loss: `MSE(predicted_velocity, noise - actions)`

#### Pi0 Joint Attention

- **论文**: arXiv:2410.24164
- 解释 joint attention + flow matching 的设计动机
- 为什么选择 segment-based mask 而不是 cross-attention

______________________________________________________________________

### 第四阶段：进阶优化机制（按需）

#### FlashAttention

- **论文**: "FlashAttention: Fast and Memory-Efficient Exact Attention" (Dao et al., 2022)
- 核心: IO-aware tiling，不存完整 N×N attention 矩阵
- 数学上完全等价，只是计算顺序不同
- 本 repo 之所以不用 FA2：Pi0.5 的 mask 不是纯 causal，FA2 不支持任意 mask

#### FlexAttention

- **PyTorch 官方博客**: 搜 "PyTorch FlexAttention"
- 支持自定义 `mask_mod` 函数，适配 Pi0.5 的 segment mask
- 本 repo 计划方案: `docs/flexattention_pi05_v1.md`

#### FSDP (Fully Sharded Data Parallel)

- **PyTorch FSDP Tutorial**: 搜 "PyTorch FSDP getting started"
- 理解参数分片、all-gather、reduce-scatter
- 本 repo 训练用 FSDP

______________________________________________________________________

## 3. 快速查阅方法

遇到代码中不认识的组件，按优先级：

1. **PyTorch 官方文档** — `torch.nn.LayerNorm` 等，有严格数学公式
2. **读本 repo 的实现** — 如 `model_utils.py` 的 `eager_attention_forward`，朴素实现最易理解
3. **搜索 "组件名 + explained"** — 如 "GQA attention explained"，通常有高质量博客
4. **搜索 "组件名 + paper"** — 找到原始论文的 Method 章节

______________________________________________________________________

## 4. 本 repo 组件速查表

| 代码中看到的                      | 它是什么                            | 首选学习资源                                               |
| --------------------------------- | ----------------------------------- | ---------------------------------------------------------- |
| `GemmaRMSNorm`                    | RMSNorm 归一化                      | Llama 论文 / HF Llama 源码                                 |
| `apply_rotary_pos_emb`            | RoPE 旋转位置编码                   | "RoFormer" 论文 (Su et al.)                                |
| `eager_attention_forward`         | 朴素 scaled dot-product attention   | Karpathy GPT 视频                                          |
| `make_att_2d_masks` (cumsum)      | Segment-based attention mask        | Pi0 论文 / `docs/flexattention_pi05_v1.md`                 |
| `AdaLayerNorm`                    | DiT 条件化 LayerNorm                | DiT 论文 (Peebles & Xie)                                   |
| `SiLU` / `F.silu`                 | Swish 激活函数                      | "Searching for Activation Functions" (Ramachandran et al.) |
| `CategorySpecificLinear`          | 按 embodiment 切换权重的 Linear     | Gr00t 论文 / 直接读源码                                    |
| `GQA` (1 KV head vs 8 Q head)     | Grouped-Query Attention             | "GQA" 论文 (Ainslie et al.)                                |
| `create_sinusoidal_pos_embedding` | Sinusoidal timestep embedding       | Vaswani 原始论文 / DDPM                                    |
| `SigLIP` (via PaliGemma)          | 对比学习 vision encoder             | SigLIP 论文 (Zhai et al.)                                  |
| `flow matching loss`              | `MSE(v_predicted, noise - actions)` | Flow Matching 论文 (Lipman et al.)                         |
| `Euler 积分` (predict_action)     | ODE 求解器                          | Flow Matching 论文                                         |
| `FSDP`                            | 分布式参数分片训练                  | PyTorch FSDP 官方教程                                      |
| `gradient_checkpointing`          | 用时间换显存                        | "Training Deep Nets with Sublinear Memory"                 |

______________________________________________________________________

## 5. 论文精选列表

按重要性排序，覆盖本 repo 涉及的所有关键技术：

| #   | 论文                                              | 年份 | 和本 repo 的关系                        |
| --- | ------------------------------------------------- | ---- | --------------------------------------- |
| 1   | Attention Is All You Need                         | 2017 | 一切的基础                              |
| 2   | Llama 2                                           | 2023 | RMSNorm + RoPE + GQA，现代 LLM 标准架构 |
| 3   | Flow Matching for Generative Modeling             | 2023 | Pi0 / Gr00t 的 action 生成方法          |
| 4   | Scalable Diffusion Models with Transformers (DiT) | 2023 | Gr00t head 的理论基础                   |
| 5   | Pi0: A Vision-Language-Action Flow Model          | 2024 | Pi0.5 的前身，joint attention 设计      |
| 6   | Pi0.5                                             | 2025 | 本 repo 主模型架构                      |
| 7   | FlashAttention                                    | 2022 | 理解 attention 优化的基础               |
| 8   | RoFormer (RoPE)                                   | 2021 | 位置编码方案                            |
| 9   | GQA                                               | 2023 | KV cache 优化                           |
| 10  | PaliGemma                                         | 2024 | Pi0.5 的视觉语言 backbone               |
