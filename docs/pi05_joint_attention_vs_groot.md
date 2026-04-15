# Pi0.5 Joint Attention vs Gr00t Cross-Attention DiT 架构对比

## 1. 概述

本文档对比分析 Pi0.5 和 Gr00t 两种 Vision-Language-Action (VLA) 模型中 attention 机制的设计差异。两者都基于 Flow Matching 生成 action trajectory，但在视觉语言信息与 action 的融合方式上采用了截然不同的策略。

| 模型  | 论文                                                 | 核心代码                                                                                      |
| ----- | ---------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| Pi0.5 | [arXiv:2504.16054](https://arxiv.org/abs/2504.16054) | `fluxvla/models/vlas/pi05_flowmatching.py`                                                    |
| Gr00t | GR00T-N1.5                                           | `fluxvla/models/heads/flow_matching_head.py` + `fluxvla/models/blocks/cross_attention_dit.py` |

______________________________________________________________________

## 2. Pi0.5 Joint Attention 详解

### 2.1 核心思想

将所有模态的 token（image、language、state、action）拼接成**一个长序列**，在同一个 transformer 中通过精心设计的 attention mask 实现联合推理。

### 2.2 序列结构

```
[img₁ img₂ ... img₅₁₂ | lang₁ ... lang₁₈₀ | state | act₁ | act₂ ... act₁₀]
 ←────── prefix (692 tokens) ──────→          ←──── suffix (11 tokens) ────→
                                    total seq_len = 703
```

- **Prefix**：由 PaliGemma (SigLIP + Gemma) 提取的图像嵌入 (512 tokens, 2 views × 256 patches) + 语言嵌入 (180 tokens, 固定 pad)
- **Suffix**：state 嵌入 (1 token) + noisy action 嵌入 (10 tokens, 融合了 timestep 信息)

### 2.3 Attention Mask 构建

#### Step 1: 1D att_masks 标记

在 `embed_prefix` 和 `embed_suffix` 中分别构建 1D 标记：

```python
# embed_prefix: 所有 prefix token 标记为 0（同一 segment，双向 attention）
att_masks = [0]*512 + [0]*180

# embed_suffix: state 和 act₁ 标记为 1（开始新 segment），其余 action 为 0
att_masks += [1, 1, 0, 0, ..., 0]  # state=1, act₁=1, act₂..₁₀=0×9
```

拼接后得到完整的 1D att_masks (shape: `(B, 703)`)。

#### Step 2: cumsum 转换为 2D mask

`make_att_2d_masks` (`fluxvla/engines/utils/model_utils.py:76-94`) 通过 `cumsum` 技巧一步生成 2D 可见性矩阵：

```python
cumsum = torch.cumsum(att_masks, dim=1)
# cumsum: [0,0,...,0,  0,0,...,0,  1,  2,  2,2,...,2]
#          ── img ──   ── lang ──  st  a₁  ── a₂..₁₀ ──
#          segment 0   segment 0   s1  s2    segment 2

att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
# 含义：token i 能看到 token j，当且仅当 segment(j) <= segment(i)
```

#### Step 3: 可见性矩阵

|  Q (行) \\ KV (列)   | img (seg=0) | lang (seg=0) | state (seg=1) | act₁ (seg=2) | act₂..₁₀ (seg=2) |
| :------------------: | :---------: | :----------: | :-----------: | :----------: | :--------------: |
|   **img (seg=0)**    |     Bi      |      Bi      |       x       |      x       |        x         |
|   **lang (seg=0)**   |     Bi      |      Bi      |       x       |      x       |        x         |
|  **state (seg=1)**   |      Y      |      Y       |       Y       |      x       |        x         |
|   **act₁ (seg=2)**   |      Y      |      Y       |       Y       |      Y       |        x         |
| **act₂..₁₀ (seg=2)** |      Y      |      Y       |       Y       |      Y       |        Bi        |

- **Bi** = 双向可见; **Y** = 单向可见 (Q 可看 KV); **x** = 不可见
- Prefix 内部：完全双向
- Prefix -> Suffix：不可见（img/lang 看不到 state/action）
- Suffix -> Prefix：可见（state/action 可以看到 img/lang）
- Suffix 内部：高 segment 可看低 segment，同 segment 双向

### 2.4 双模型联合 Transformer

Pi0.5 的 "joint" 不仅指序列拼接，还包括**两套独立参数在同一个 attention 中协作**：

```
_forward_transformer_layers (pi0_flowmatching.py:330-457):

每一层 Transformer:
  ┌─────────────────────────────────────────────┐
  │ 1. prefix tokens → llm_backbone 的 Q/K/V proj │
  │ 2. suffix tokens → llm_expert 的 Q/K/V proj   │  (独立参数!)
  │                                                │
  │ 3. Q = cat([Q_prefix, Q_suffix])               │
  │    K = cat([K_prefix, K_suffix])               │
  │    V = cat([V_prefix, V_suffix])               │
  │                                                │
  │ 4. RoPE → 统一 attention(mask) → 分拆输出      │
  │                                                │
  │ 5. prefix → backbone 的 O_proj + FFN           │
  │    suffix → expert 的 O_proj + FFN             │
  └─────────────────────────────────────────────┘
```

- `llm_backbone`：Gemma 权重，处理 prefix（VL token）
- `llm_expert`：独立的 Gemma 权重，处理 suffix（state + action token）
- 两者层数必须相同，**逐层共享一个 attention 矩阵**

### 2.5 时间步注入

Pi0.5 通过 **AdaRMS (Adaptive RMS Norm)** 将 timestep 信息注入 suffix 侧：

```python
# pi05_flowmatching.py embed_suffix
time_emb = sinusoidal_embedding(timestep, dim=proj_width)  # (B, D)
time_emb = SiLU(time_mlp_in(time_emb))
time_emb = SiLU(time_mlp_out(time_emb))
# 返回为 adarms_cond，在 llm_expert 的 LayerNorm 中做条件化
```

注意：Pi0.5 的 timestep 是**标量** `(B,)`，所有 action token 共享同一个 time conditioning。Action embedding 不直接与 time 融合（区别于 Pi0 原版）。

### 2.6 推理流程

```
1. embed_prefix → prefix KV cache（只算一次）
2. 10 步 Euler 积分:
   for t in [1.0, 0.9, ..., 0.1]:
     embed_suffix(state, x_t, t) → suffix tokens
     suffix attend to prefix KV cache → v_t
     x_t += dt * v_t
3. action_out_proj(x_t) → predicted actions
```

______________________________________________________________________

## 3. Gr00t Cross-Attention DiT 详解

### 3.1 核心思想

**两阶段**架构：先用 VLM backbone 提取 vision-language features，再用独立的 DiT (Diffusion Transformer) head 通过 cross-attention 消费这些 features 来生成 action。

### 3.2 整体流程

```
阶段 1: Eagle VLM backbone
         images + lang_tokens → VL features (B, S, 2048)
                                      ↓ LayerNorm (VLLN)
                                      ↓ SelfAttentionTransformer (4 层)
                                      ↓
                               refined VL features

阶段 2: DiT head
         state + future_tokens + noisy_actions → hidden_states (B, 43, 1536)
         VL features 作为 encoder_hidden_states → cross-attention
         timestep → AdaLN 调制
                    ↓
         predicted velocity → action decode
```

### 3.3 DiT 的交替 Attention 结构

`DiT` (`fluxvla/models/blocks/cross_attention_dit.py:240-444`) 使用 16 层 `BasicTransformerBlock`，交替执行：

```python
for idx, block in enumerate(transformer_blocks):   # 16 layers
    if idx % 2 == 1:   # 奇数层: Self-Attention
        block(hidden_states, encoder_hidden_states=None)
        # action tokens 内部互相 attend
    else:               # 偶数层: Cross-Attention
        block(hidden_states, encoder_hidden_states=vl_features)
        # action tokens attend to VL features
```

每个 `BasicTransformerBlock` 内部结构：

```
input → AdaLN(timestep) → Attention → Residual → LayerNorm → FFN → Residual → output
```

- Cross-Attention 层：Q 来自 action side，K/V 来自 VL features
- Self-Attention 层：Q/K/V 全部来自 action side

### 3.4 DiT 输入序列

```
hidden_states = [state(1) | future_tokens(32) | action(10)]  → (B, 43, 1536)
encoder_hidden_states = VL features                           → (B, S, 2048)
```

- **future_tokens**：32 个可学习的 nn.Embedding，作为模型的"工作区"来建模未来状态
- 这两个序列**完全分离**，仅通过 cross-attention 桥接

### 3.5 时间步注入

Gr00t 使用 **AdaLN (Adaptive Layer Normalization)**：

```python
# TimestepEncoder: timestep → embedding
temb = self.time_proj(timesteps)           # Sinusoidal projection
temb = self.timestep_embedder(temb)        # MLP

# AdaLayerNorm: 在每层 Transformer 中
scale, shift = linear(silu(temb)).chunk(2)
x = LayerNorm(x) * (1 + scale) + shift
```

### 3.6 多 Embodiment 支持

Gr00t 使用 `CategorySpecificLinear`/`CategorySpecificMLP` 实现多机器人支持：

```python
class CategorySpecificLinear:
    def __init__(self, num_categories, input_dim, hidden_dim):
        self.W = nn.Parameter(0.02 * randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(zeros(num_categories, hidden_dim))

    def forward(self, x, cat_ids):
        selected_W = self.W[cat_ids]  # 按 embodiment_id 选择权重
        return bmm(x, selected_W) + selected_b
```

State encoder、action encoder、action decoder 都是 category-specific 的。

### 3.7 推理流程

```
1. VLM backbone → VL features（只算一次）
2. VLLN + SelfAttention → refined VL features
3. state_encoder(state) → state features
4. 4 步 Euler 积分:
   for t in [0, 0.25, 0.5, 0.75]:
     action_encoder(x_t, t) → action features
     cat(state, future_tokens, action) → DiT input
     DiT(input, VL_features, timestep) → velocity
     x_t += dt * velocity
5. action_decoder(x_t) → predicted actions
```

______________________________________________________________________

## 4. 核心区别对比

|                维度                |                       Pi0.5 (Joint Attention)                        |          Gr00t (Cross-Attention DiT)          |
| :--------------------------------: | :------------------------------------------------------------------: | :-------------------------------------------: |
|            **架构范式**            |          所有 token 在同一 transformer 中做 joint attention          |     VLM 提取特征 → 独立 DiT 做 denoising      |
|     **VL 与 Action 交互方式**      |              共享同一个 attention 矩阵，mask 控制可见性              |    两个独立序列，通过 cross-attention 桥接    |
|            **参数结构**            | prefix 用 Gemma backbone，suffix 用 Gemma expert，层级共享 attention |            VLM 和 DiT 参数完全独立            |
| **VL 特征是否在 denoising 中更新** |                **是** — prefix 在每层都参与计算并更新                | **否** — VL features 只在 backbone 中计算一次 |
|        **Attention 复杂度**        |                    O((prefix+suffix)²) = O(703²)                     |    O(action × VL) cross + O(action²) self     |
|           **时间步注入**           |                AdaRMS Norm（仅 suffix 侧 LayerNorm）                 |     AdaLN（条件化 DiT 每一层 LayerNorm）      |
|           **时间步粒度**           |                     标量 (B,)，所有 action 共享                      |          可支持 per-position (B, T)           |
|         **Denoising 步数**         |                                10 步                                 |                     4 步                      |
|         **多 Embodiment**          |                    不支持（单一 embodiment 设计）                    |       支持（CategorySpecific 权重切换）       |
|           **预训练基底**           |                    PaliGemma (SigLIP + Gemma 2B)                     |                 Eagle 3B VLM                  |
|         **推理时 VL 复用**         |                            KV Cache 机制                             |            特征提取一次后重复使用             |
|         **Future Tokens**          |                                  无                                  |        32 个可学习 token 作为"工作区"         |

______________________________________________________________________

## 5. 设计哲学差异

### 5.1 Pi0.5: 深度融合，重表达力

Pi0.5 让 VL 和 action token **在每一层 transformer 中都直接交互**。suffix token 的 Q 可以 attend 到 prefix 的 K/V，获取最新的、经过多层更新的视觉语言表示。这意味着：

- Action 生成能"看到"更丰富的上下文
- VL 表示本身也在 joint attention 中被间接更新（虽然 prefix 看不到 suffix）
- 代价：attention 矩阵较大 (703²)，推理需要 KV cache 优化

### 5.2 Gr00t: 模块化，重效率

Gr00t 将 VLM 和 action head 完全解耦：

- VLM backbone 只算一次，denoising 循环中只跑轻量的 DiT
- DiT 的 hidden_states 只有 43 个 token，attention 计算非常高效
- 更容易支持多 embodiment（通过 category-specific 权重）
- 代价：VL features 在 denoising 循环中是"冻结"的，信息流是单向的

### 5.3 Mask 设计的精妙之处

Pi0.5 的 cumsum mask 用一行代码 `cumsum[:, None, :] <= cumsum[:, :, None]` 就表达了复杂的 segment 级可见性。这比 Gr00t 的 cross-attention 更灵活：

- 可以轻松控制任意 segment 间的可见性
- 天然支持 FlexAttention 的 `mask_mod` 接口（参见 `docs/flexattention_pi05_v1.md`）
- 同一个 mask 格式可以适配 eager / SDPA / FlexAttention 等多种后端

______________________________________________________________________

## 6. 数据流对比图

### Pi0.5

```
images (B,2,3,224,224)
  │
  ↓
SigLIP ──→ (B,2,256,1152) ──→ projector ──→ img_embs (B,512,2048) ─┐
                                                                      │
                                                          prefix_embs │ (B,512+seq_len,2048)
                                                                      │  seq_len≈180
lang_tokens (B,seq_len) ──→ Gemma embed ──→ lang_embs (B,seq_len,2048)┘
                                                                       │
state (B,32)                                                           │
  │                                                                    │
  ↓                                                                    │
state_proj ──→ state_emb (B,1,1024) ─────────┐                        │
                                               │                        │
noisy_actions (B,10,32)                        │  suffix_embs            │
  │                                            │  (B,11,1024)           │
  ↓                                            │                        │
action_in_proj ──→ (B,10,1024) ──┐            │                        │
                                  ├─ cat ─────┘                        │
timestep (B,)                     │                                     │
  │                               │                                     │
  ↓                               │                                     │
sin_emb → MLP ──→ time_emb (B,1024)                                   │
  │                                                                     │
  │  adarms_cond                                                        │
  ↓                                          ↓                          ↓
┌──────────────────────────────────────────────────────────────────────┐
│              Joint Attention (per layer × N layers)                   │
│                                                                       │
│  prefix (B,692,2048) → Gemma backbone Q/K/V proj → (B,692,2048)     │
│  suffix (B,11,1024)  → Gemma expert   Q/K/V proj → (B,11,1024)      │
│                                                                       │
│  Q = cat → (B, H, 703, head_dim)    ← 统一 attention                │
│  K = cat → (B, H, 703, head_dim)                                     │
│  V = cat → (B, H, 703, head_dim)                                     │
│  attention_mask: (B, 703, 703)  cumsum-based segment mask            │
│                                                                       │
│  output → split → prefix_out (B,692,2048) → backbone O_proj + FFN   │
│                  → suffix_out (B,11,1024)  → expert   O_proj + FFN   │
└──────────────────────────────────────────────────────────────────────┘
                     │
                     ↓
              suffix_out[:, 1:, :] (B,10,1024)
                     │
                     ↓
              action_out_proj → v_t (B,10,32)  (predicted velocity)
```

### Gr00t

```
images + lang_tokens
  │
  ↓
Eagle VLM backbone ──→ VL features (B, S, 2048)     S = vision + lang tokens
  │
  ↓
LayerNorm (VLLN) ──→ (B, S, 2048)
  │
  ↓
SelfAttentionTransformer (4 layers) ──→ refined VL features (B, S, 2048) ──┐
                                                                             │
                                           encoder_hidden_states             │
                                                                             │
state (B, 64)                                                                │
  │                                                                          │
  ↓                                                                          │
CategorySpecificMLP ──→ state_emb (B, 1, 1536) ──────────┐                  │
                                                           │                  │
future_tokens (learnable) ──→ (B, 32, 1536) ─────────────┤                  │
                                                           │   cat            │
noisy_actions (B, 10, 32)                                  │   ↓             │
  │                                                        │ hidden_states   │
  ↓                              timestep (B,)             │ (B,43,1536)     │
ActionEncoder ──────────────────────┘                      │                  │
  │    action_emb (B, 10, 1536) ──→ + temb ──────────────┘                  │
  │                                                                          │
  ↓                                                                          ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│  DiT (16 layers, alternating cross/self attention)                          │
│                                                                              │
│  hidden_states:           (B, 43, 1536)                                     │
│  encoder_hidden_states:   (B, S,  2048)                                     │
│  timestep_emb:            (B, 1536)                                         │
│                                                                              │
│  Even layers (0,2,...,14): Cross-Attention                                  │
│    Q ← hidden  (B, H, 43, head_dim)                                        │
│    K,V ← VL    (B, H, S,  head_dim)                                        │
│    AdaLN(timestep) → scale, shift                                           │
│                                                                              │
│  Odd layers  (1,3,...,15): Self-Attention                                   │
│    Q,K,V ← hidden (B, H, 43, head_dim)                                     │
│    AdaLN(timestep) → scale, shift                                           │
│                                                                              │
│  output: (B, 43, 1536) → proj → (B, 43, 1024)                              │
└─────────────────────────────────────────────────────────────────────────────┘
             │
             ↓
      取 action 部分: (B, 10, 1024)
             │
             ↓
      CategorySpecificMLP → predicted velocity (B, 10, 32) → denorm → (B, 10, 7)
```

______________________________________________________________________

## 7. 相关文件索引

| 文件                                           | 说明                                           |
| ---------------------------------------------- | ---------------------------------------------- |
| `fluxvla/models/vlas/pi0_flowmatching.py`      | Pi0 基类，joint attention 核心实现             |
| `fluxvla/models/vlas/pi05_flowmatching.py`     | Pi0.5 子类，override embed_suffix              |
| `fluxvla/engines/utils/model_utils.py`         | `make_att_2d_masks`、`eager_attention_forward` |
| `fluxvla/models/heads/flow_matching_head.py`   | Gr00t FlowMatchingHead                         |
| `fluxvla/models/blocks/cross_attention_dit.py` | DiT、BasicTransformerBlock、AdaLN              |
| `docs/flexattention_pi05_v1.md`                | Pi0.5 FlexAttention 加速方案                   |
| `configs/pi05/`                                | Pi0.5 训练配置                                 |
| `configs/gr00t/`                               | Gr00t 训练配置                                 |
