# FlexAttention Implementation Plan for Pi0.5 Joint Attention

## 1. 为什么 FlexAttention 是 Pi0.5 的最佳方案

上一轮分析中我们发现：

| 方案 | Pi0.5 适配性 | 原因 |
|------|------------|------|
| Eager | 精确但慢 | 标准 matmul，O(N²) 显存 |
| SDPA | 中等 | 支持 bool mask，但传入自定义 mask 时可能退化为 math 后端 |
| FA2 | **不适合** | 只有 causal/non-causal，无法表达 Pi0.5 的混合 mask |
| **FlexAttention** | **最适合** | 可将 Pi0.5 的 cumsum mask 编译为高效 Triton kernel |

FlexAttention 的核心优势：**你定义 mask 逻辑，编译器帮你生成优化 kernel**——既保持精确语义，又获得接近 FlashAttention 的性能。

---

## 2. FlexAttention 技术原理

### 2.1 核心思想

传统方式：先生成完整的 N×N mask 矩阵 → 传给 attention kernel

FlexAttention：**只定义一个 mask 函数** → 编译器自动：
1. 在 block 级别（128×128）评估哪些 block 全空（skip）、全满（无需 mask）、部分有效（apply mask）
2. 生成 fused Triton kernel，将 mask 判断内联到 attention 计算中
3. 不再需要 O(N²) 的显存存储完整 mask

```
┌──────────────────────────────────────────┐
│              传统 Attention                │
│  1. 构造 N×N mask 矩阵（O(N²) 显存）       │
│  2. Q @ K^T                               │
│  3. 加上 mask                              │
│  4. softmax                               │
│  5. @ V                                   │
│  每步独立 kernel，频繁访问 HBM              │
└──────────────────────────────────────────┘

┌──────────────────────────────────────────┐
│            FlexAttention                  │
│  1. 编译 mask_mod → BlockMask             │
│     在 block 级别确定 skip/full/partial    │
│  2. Fused Triton Kernel:                  │
│     对每个 non-skip block:                │
│       Q_tile @ K_tile                     │
│       内联调用 score_mod（含 mask 逻辑）    │
│       online softmax                      │
│       @ V_tile                            │
│     全部在 SRAM 中完成                     │
│  显存 O(N)，单次 kernel launch             │
└──────────────────────────────────────────┘
```

### 2.2 两个核心 API

#### `mask_mod(b, h, q_idx, kv_idx) → bool`

定义注意力的可见性规则：返回 `True` 表示 q_idx 位置可以看到 kv_idx 位置。

```python
# 标准 causal mask
def causal(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

# Prefix LM（prefix 双向 + suffix 因果）
def prefix_lm(b, h, q_idx, kv_idx):
    return (kv_idx < prefix_len) | (q_idx >= kv_idx)
```

#### `score_mod(score, b, h, q_idx, kv_idx) → float`

修改 softmax 前的注意力分数，可用于 ALiBi、relative position bias 等。

```python
# ALiBi
def alibi(score, b, h, q_idx, kv_idx):
    return score - slopes[h] * abs(q_idx - kv_idx)
```

### 2.3 BlockMask：block 级别稀疏

`create_block_mask` 在 block 粒度（默认 128×128）评估 mask_mod：

```python
from torch.nn.attention.flex_attention import create_block_mask

block_mask = create_block_mask(
    mask_mod=my_mask_fn,
    B=batch_size,      # batch 维度
    H=num_heads,       # 或 None（所有 head 共享）
    Q_LEN=seq_len,     # query 序列长度
    KV_LEN=seq_len,    # key/value 序列长度
    device="cuda",
)
```

对于 Pi0.5 的 mask 模式，BlockMask 会自动识别：
- **Prefix-Prefix block**：全满（skip masking）
- **Action-Action block**：全满（skip masking）
- **Prefix-Action block**（上三角区域）：全空（skip computation）
- **边界 block**：部分有效（apply mask_mod）

这种 block sparsity 可以跳过大量无效计算。

### 2.4 与 torch.compile 的关系

FlexAttention **需要 `torch.compile`** 才能触发 Triton kernel 生成：

```python
from torch.nn.attention.flex_attention import flex_attention

# 必须 compile
flex_attention_compiled = torch.compile(flex_attention)

# 调用
output = flex_attention_compiled(Q, K, V, block_mask=block_mask)
```

不 compile 也能跑（走 eager fallback），但没有性能优势。

---

## 3. Pi0.5 的 Mask 模式分析

### 3.1 当前 mask 构建方式

Pi0.5 用 `att_masks`（1D）通过 `cumsum` 构建 2D mask：

```python
# att_masks 值:
# [0, 0, ..., 0 (img), 0, 0, ..., 0 (lang), 1 (state), 1 (act₁), 0, 0, ..., 0 (act₂..ₙ)]

cumsum = torch.cumsum(att_masks, dim=1)
# cumsum:
# [0, 0, ..., 0,       0, 0, ..., 0,       1,          2,         2, 2, ..., 2]

# 2D mask: cumsum[q] >= cumsum[kv]
att_2d = cumsum[:, None, :] <= cumsum[:, :, None]
```

### 3.2 可视化 mask 模式

```
序列:    [img₁ img₂ ... imgₘ | lang₁ ... langₙ | state | act₁ | act₂ ... actₖ]
cumsum:  [0    0         0   | 0          0     | 1     | 2    | 2         2   ]

                   img    lang   state  act₁  act₂..ₖ
        img      [  ✓       ✓      ✗     ✗      ✗   ]   cumsum=0, 只看 ≤0
        lang     [  ✓       ✓      ✗     ✗      ✗   ]   cumsum=0, 只看 ≤0
        state    [  ✓       ✓      ✓     ✗      ✗   ]   cumsum=1, 看 ≤1
        act₁     [  ✓       ✓      ✓     ✓      ✗   ]   cumsum=2, 看 ≤2（但 act₂ 也是 2）
        act₂..ₖ  [  ✓       ✓      ✓     ✓      ✓   ]   cumsum=2, 看 ≤2
```

这是一个 **3 段式 block mask**：
- **Block A（prefix 内部）**：完全双向
- **Block B（suffix 看 prefix）**：完全可见（下三角的非对角区域）
- **Block C（suffix 内部）**：近似因果（state→act₁→act₂..ₖ 递增关系，但 act₂..ₖ 彼此双向）
- **Block D（prefix 看 suffix）**：完全不可见

### 3.3 转化为 FlexAttention mask_mod

核心思路：将 cumsum 值作为 **segment id** 传入 mask_mod 函数。

```python
# segment_ids: 每个 token 的 cumsum 值，shape (B, L)
# 通过 torch.cumsum(att_masks, dim=1) 得到

def pi05_mask_mod(b, h, q_idx, kv_idx):
    q_segment = segment_ids[b, q_idx]   # query token 的 segment id
    kv_segment = segment_ids[b, kv_idx]  # key token 的 segment id
    return q_segment >= kv_segment        # 与原始 cumsum 逻辑完全一致
```

**这就是全部！** Pi0.5 的 cumsum mask 逻辑天然适合 FlexAttention 的 mask_mod 形式。

---

## 4. 详细实现方案

### 4.1 新增 `flex_attention_forward` 函数

**位置**：`fluxvla/engines/utils/model_utils.py`

```python
import torch
from torch.nn.attention.flex_attention import (
    flex_attention,
    create_block_mask,
)

# compile flex_attention 以触发 Triton kernel
_flex_attention_compiled = torch.compile(flex_attention)


def create_pi05_block_mask(att_masks, pad_masks, num_heads, device):
    """从 Pi0.5 的 att_masks 构建 FlexAttention BlockMask。

    Args:
        att_masks: (B, L) bool tensor, 原始 1D attention masks
        pad_masks: (B, L) bool tensor, padding masks
        num_heads: query head 数量（用于 H 参数）
        device: CUDA device

    Returns:
        BlockMask 对象
    """
    B, L = att_masks.shape

    # 计算 segment ids（与 make_att_2d_masks 中的 cumsum 逻辑一致）
    segment_ids = torch.cumsum(att_masks.int(), dim=1)  # (B, L)

    def mask_mod(b, h, q_idx, kv_idx):
        # 1. segment 可见性：q_segment >= kv_segment
        q_seg = segment_ids[b, q_idx]
        kv_seg = segment_ids[b, kv_idx]
        segment_ok = q_seg >= kv_seg

        # 2. padding 可见性：两个位置都不是 padding
        q_pad = pad_masks[b, q_idx]
        kv_pad = pad_masks[b, kv_idx]
        pad_ok = q_pad & kv_pad

        return segment_ok & pad_ok

    block_mask = create_block_mask(
        mask_mod,
        B=B,
        H=None,    # 所有 head 共享同一 mask（广播）
        Q_LEN=L,
        KV_LEN=L,
        device=device,
    )
    return block_mask


def flex_attention_forward(
    module,           # nn.Module (GemmaAttention layer)
    query,            # (B, Hq, L, D) = (B, 8, L, 256)
    key,              # (B, Hkv, L, D) = (B, 1, L, 256)
    value,            # (B, Hkv, L, D) = (B, 1, L, 256)
    attention_mask,   # BlockMask 对象（不再是 4D float tensor）
    scaling,          # float, head_dim ** -0.5
    dropout=0.0,
    **kwargs,
):
    """FlexAttention 注意力计算。

    注意：attention_mask 参数此时应传入 BlockMask 对象，
    而非传统的 4D additive mask。需要在调用前将 mask 构建逻辑
    从 _prepare_attention_masks_4d 切换到 create_pi05_block_mask。
    """
    # flex_attention 输入格式: (B, H, L, D)（与 eager 相同，无需转置）
    # 原生支持 GQA (enable_gqa=True)
    attn_output = _flex_attention_compiled(
        query, key, value,
        block_mask=attention_mask,  # BlockMask 对象
        scale=scaling,
        enable_gqa=True,
    )
    # 输出: (B, Hq, L, D)

    # 转置为 (B, L, Hq, D) 匹配 eager 输出格式
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, None
```

### 4.2 修改 mask 构建流程

**核心变更**：当 `attention_implementation='flex'` 时，不走 `_prepare_attention_masks_4d`（生成 4D float mask），而是走 `create_pi05_block_mask`（生成 BlockMask 对象）。

**位置**：`fluxvla/models/vlas/pi0_flowmatching.py`

需要修改 `forward()` 方法中的 mask 构建部分（约 line 630-637）：

```python
# 原始代码:
attention_masks = make_att_2d_masks(pad_masks, att_masks)
att_2d_masks_4d = self._prepare_attention_masks_4d(attention_masks)

# 改为:
if self.attention_implementation == 'flex':
    # 直接从 1D masks 构建 BlockMask，跳过 O(N²) 的 2D mask 构建
    attention_masks = create_pi05_block_mask(
        att_masks, pad_masks,
        num_heads=self.llm_backbone.config.num_attention_heads,
        device=att_masks.device,
    )
else:
    attention_masks = make_att_2d_masks(pad_masks, att_masks)
    attention_masks = self._prepare_attention_masks_4d(attention_masks)
```

**注意**：PI05FlowMatching 的 `forward()` 也需要同样的修改（如果它 override 了 mask 构建部分）。

### 4.3 更新 `get_attention_interface`

**位置**：`fluxvla/models/vlas/pi0_flowmatching.py` lines 192-209

```python
def get_attention_interface(self):
    if self.attention_implementation == 'fa2':
        raise NotImplementedError('FA2 is not suitable for Pi0.5 (mask incompatible)')
    elif self.attention_implementation == 'flex':
        attention_interface = flex_attention_forward  # 替换 NotImplementedError
    elif self.attention_implementation == 'eager':
        attention_interface = eager_attention_forward
    elif self.attention_implementation == 'sdpa':
        attention_interface = sdpa_attention_forward
    else:
        raise ValueError(...)
    return attention_interface
```

### 4.4 处理 torch.compile 与 gradient checkpointing 的交互

FlexAttention 需要 `torch.compile`，而训练使用 gradient checkpointing。需确保：

```python
# flex_attention 本身是 compiled 的
_flex_attention_compiled = torch.compile(flex_attention)

# gradient checkpointing 包裹的是外层 transformer layer，不影响内部 compiled kernel
# PyTorch 2.6 支持 compiled function 在 checkpoint 内运行
```

如果遇到 compile + checkpoint 兼容性问题，备选方案：
```python
# 使用 torch.compile 的 fullgraph=False 模式（允许 graph break）
_flex_attention_compiled = torch.compile(flex_attention, fullgraph=False)
```

---

## 5. 性能分析

### 5.1 Block Sparsity 收益

Pi0.5 的 mask 模式在 block 级别的稀疏分布：

```
假设 seq_len=300, block_size=128

Block 布局 (128×128):
    KV →   [block_0: 0-127]  [block_1: 128-255]  [block_2: 256-299]
Q ↓
[block_0]     FULL (✓)          PARTIAL             EMPTY
[block_1]     FULL (✓)          FULL (✓)            PARTIAL
[block_2]     FULL (✓)          FULL (✓)            FULL (✓)
```

- **EMPTY blocks**：完全跳过计算（prefix 看不到 suffix）
- **FULL blocks**：不需要 apply mask（纯 attention）
- **PARTIAL blocks**：apply mask_mod

约 1/9 的 blocks 可完全跳过，2/3 的 blocks 无需 mask 操作。

### 5.2 预估加速

| 后端 | 相对 eager 加速比 | 说明 |
|------|-----------------|------|
| Eager | 1x | 基线 |
| SDPA (bool mask) | ~5-10x | 可能退化为 math 后端 |
| **FlexAttention** | **~10-30x** | Triton kernel + block sparsity |
| FA2 (causal) | ~20-75x | 但 mask 不精确 |

FlexAttention 的加速比介于 SDPA 和 FA2 之间，**但保持精确 mask 语义**。

### 5.3 额外收益：节省 mask 构建显存

当前方式：
```python
attention_masks = make_att_2d_masks(pad_masks, att_masks)  # (B, L, L) bool
att_2d_masks_4d = self._prepare_attention_masks_4d(...)     # (B, 1, L, L) float
```
显存：O(B × L²)，对于 B=16, L=400 约 **~100MB**

FlexAttention：
```python
block_mask = create_pi05_block_mask(att_masks, pad_masks, ...)
```
显存：O(B × (L/128)²) block 级别元数据，约 **<1MB**

---

## 6. 完整改动范围

| 文件 | 改动类型 | 说明 |
|------|---------|------|
| `fluxvla/engines/utils/model_utils.py` | **新增** | `create_pi05_block_mask` + `flex_attention_forward` |
| `fluxvla/models/vlas/pi0_flowmatching.py` | **修改** | `get_attention_interface` 添加 flex 分支；`forward` 中 mask 构建分支 |
| `fluxvla/models/vlas/pi05_flowmatching.py` | **修改** | `forward` 中如有 mask 构建，同样添加分支 |
| `test/test_ops/test_flashattn.py` | **新增** | FlexAttention 数值一致性测试 |

**不需要改动**：
- `_forward_transformer_layers` —— 函数指针切换，零结构变更
- `make_att_2d_masks` —— 保留给 eager/sdpa 后端使用
- `condition_gemma.py` —— 独立路径
- 训练配置 —— 只需改 `attention_implementation: 'flex'`

---

## 7. 风险和注意事项

### 7.1 torch.compile 要求

FlexAttention 必须在 `torch.compile` 下运行才有性能优势。需确认：
- 当前训练 pipeline 是否已启用 `torch.compile`
- 如未启用，`flex_attention` 仍能运行（eager fallback），但无加速
- `torch.compile` + FSDP + gradient checkpointing 的三方兼容性需验证

### 7.2 BlockMask 动态性

`create_block_mask` 需要在每个 forward 中调用（因为 padding 不同导致 mask 不同）。PyTorch 提供了 `_compile=True` 参数加速 BlockMask 创建：

```python
block_mask = create_block_mask(mask_mod, B, H, Q_LEN, KV_LEN, _compile=True)
```

### 7.3 head_dim=256 支持

FlexAttention 底层是 Triton kernel，对 head_dim=256 的支持取决于 shared memory 大小。H100 的 shared memory (228KB) 足以支持 head_dim=256，但 A100 (164KB) 可能需要减小 block_size。

### 7.4 GQA 支持

FlexAttention 原生支持 GQA（`enable_gqa=True`），Hq=8, Hkv=1 的配置无问题。

### 7.5 回退策略

如果 FlexAttention 在特定环境下不可用（PyTorch 版本、GPU 架构），应自动回退到 SDPA 或 eager：

```python
def get_attention_interface(self):
    if self.attention_implementation == 'flex':
        try:
            from torch.nn.attention.flex_attention import flex_attention
            attention_interface = flex_attention_forward
        except ImportError:
            warnings.warn('FlexAttention not available, falling back to eager')
            attention_interface = eager_attention_forward
    ...
```

---

## 8. 测试方案

### 8.1 数值一致性

```python
def test_flex_matches_eager():
    """验证 FlexAttention 与 eager 在 bf16 容差内数值一致。"""
    # 构造 Pi0.5 配置: B=2, L=64, Hq=8, Hkv=1, D=256
    # 用真实的 att_masks 构建 BlockMask 和 4D additive mask
    # 对比两种后端输出
    # assert torch.allclose(out_flex, out_eager, atol=1e-2, rtol=1e-2)
```

### 8.2 BlockMask 正确性

```python
def test_block_mask_matches_2d_mask():
    """验证 BlockMask 与 make_att_2d_masks 的语义一致。"""
    # 构造 att_masks 和 pad_masks
    # 分别通过 make_att_2d_masks 和 create_pi05_block_mask 构建 mask
    # 逐元素对比可见性
```

### 8.3 Gradient Checkpointing 兼容性

```python
def test_flex_with_gradient_checkpointing():
    """验证 FlexAttention 在 gradient checkpointing 下可反向传播。"""
    # 用 torch.utils.checkpoint.checkpoint 包裹
    # 验证 loss.backward() 正常
```

### 8.4 端到端训练 Smoke Test

```bash
# 修改配置 attention_implementation='flex'，跑 10 步训练
# 验证 loss 正常下降，无 NaN
```

---

## 9. 总结：为什么 FlexAttention 是 Pi0.5 的最优解

| 维度 | Eager | SDPA | FA2 | **FlexAttention** |
|------|-------|------|-----|-------------------|
| Mask 保真度 | 精确 | 精确 | 近似 | **精确** |
| 加速比 | 1x | 5-10x | 20-75x | **10-30x** |
| 显存（mask） | O(N²) | O(N²) | 0 | **O(1) block 级** |
| head_dim=256 | ✓ | ✓ | 受限 | **✓** |
| GQA 支持 | repeat_kv | 原生 | 原生 | **原生** |
| Block sparsity | ✗ | ✗ | ✗ | **✓（跳过空 block）** |
| 需要 compile | ✗ | ✗ | ✗ | **✓** |

**FlexAttention 在保持精确 mask 语义的前提下，获得最优性能**，是 Pi0.5 这种非标准 mask 模式的理想选择。

---

## References

- [PyTorch FlexAttention Official Docs](https://docs.pytorch.org/docs/stable/nn.attention.flex_attention.html)
- [FlexAttention for Inference Blog](https://pytorch.org/blog/flexattention-for-inference/)
- [FlexAttention Paper (arXiv)](https://arxiv.org/html/2412.05496v1)
- [Attention Gym (GitHub)](https://github.com/meta-pytorch/attention-gym)
- [Colfax Research FlexAttention Guide](https://research.colfax-intl.com/a-users-guide-to-flexattention-in-flash-attention-cute-dsl/)
