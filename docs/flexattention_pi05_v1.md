# FlexAttention for Pi0.5 Joint Attention — Implementation Plan v1

## 1. 目标

将 Pi0.5 训练中的 eager attention 替换为 FlexAttention，在**保持精确 mask 语义**的前提下获得显著加速。

## 2. 为什么可行

### 2.1 seq_len 完全固定，compile 零重编译

```
img_tokens:   224/14 × 224/14 × 2 views = 512  (固定)
lang_tokens:  max_len = 180                     (固定 pad)
state:        1                                  (固定)
actions:      n_action_steps = 10                (固定)
─────────────────────────────────────────────────
total seq_len = 703                              (每个 batch 完全相同)
```

`torch.compile` 只编译一次，之后所有 batch 命中缓存。

### 2.2 只 compile 一个无状态函数，不影响训练 pipeline

```python
# 编译范围：仅 flex_attention 这一个算子
_flex_attn = torch.compile(flex_attention)

# 不涉及：
# - FSDP 的参数 all-gather/reduce-scatter（外层逻辑）
# - gradient checkpointing 的 save/recompute（外层逻辑）
# - 优化器状态管理
```

### 2.3 Pi0.5 mask 天然适配 mask_mod

Pi0.5 的 cumsum mask 可以直接表达为 FlexAttention 的 `mask_mod`：

```
att_masks:  [0×512 (img), 0×180 (lang), 1 (state), 1 (act₁), 0×9 (act₂..₁₀)]
cumsum:     [0×512,        0×180,        1,         2,         2×9            ]

mask_mod: q_segment >= kv_segment  ← 一行逻辑，完全等价于 make_att_2d_masks
```

生成的 block mask 模式：

```
703×703 attention matrix, block_size=128:

         KV →  [blk0: 0-127]  [blk1: 128-255]  ... [blk5: 640-703]
Q ↓
[blk0]          FULL            FULL (prefix内)      EMPTY
[blk1]          FULL            FULL                  EMPTY
 ...
[blk5]          FULL            FULL                  FULL/PARTIAL
```

- prefix→suffix 区域 EMPTY（跳过计算）
- prefix 内部 FULL（跳过 mask 判断）
- 只有 suffix 边界 block 需要 apply mask_mod

______________________________________________________________________

## 3. 改动清单

### 3.1 `fluxvla/engines/utils/model_utils.py` — 新增两个函数

#### `create_pi05_block_mask(att_masks, pad_masks, device)`

从 Pi0.5 的 1D att_masks + pad_masks 构建 BlockMask 对象。

```python
import torch
from torch.nn.attention.flex_attention import create_block_mask

def create_pi05_block_mask(att_masks, pad_masks, device):
    """从 Pi0.5 的 1D masks 构建 FlexAttention BlockMask。

    Args:
        att_masks: (B, L) bool tensor
        pad_masks: (B, L) bool tensor
        device: CUDA device

    Returns:
        BlockMask 对象
    """
    B, L = att_masks.shape
    segment_ids = torch.cumsum(att_masks.int(), dim=1)  # (B, L)

    def mask_mod(b, h, q_idx, kv_idx):
        q_seg = segment_ids[b, q_idx]
        kv_seg = segment_ids[b, kv_idx]
        segment_ok = q_seg >= kv_seg
        pad_ok = pad_masks[b, q_idx] & pad_masks[b, kv_idx]
        return segment_ok & pad_ok

    return create_block_mask(
        mask_mod,
        B=B,
        H=None,       # 所有 head 共享
        Q_LEN=L,
        KV_LEN=L,
        device=device,
    )
```

#### `flex_attention_forward(module, query, key, value, attention_mask, scaling, ...)`

与 `eager_attention_forward` 同签名的 FlexAttention 实现。

```python
from torch.nn.attention.flex_attention import flex_attention

_flex_attention_compiled = torch.compile(flex_attention)

def flex_attention_forward(
    module,
    query,            # (B, Hq, L, D)
    key,              # (B, Hkv, L, D)
    value,            # (B, Hkv, L, D)
    attention_mask,   # BlockMask 对象
    scaling,
    dropout=0.0,
    **kwargs,
):
    attn_output = _flex_attention_compiled(
        query, key, value,
        block_mask=attention_mask,
        scale=scaling,
        enable_gqa=True,
    )
    # (B, Hq, L, D) → (B, L, Hq, D)，匹配 eager 输出格式
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, None
```

### 3.2 `fluxvla/models/vlas/pi0_flowmatching.py` — 两处修改

#### (a) `get_attention_interface()` (line 192)

```python
def get_attention_interface(self):
    if self.attention_implementation == 'flex':
        from fluxvla.engines.utils.model_utils import flex_attention_forward
        attention_interface = flex_attention_forward
    elif self.attention_implementation == 'eager':
        attention_interface = eager_attention_forward
    # ... 其余保持不变
    return attention_interface
```

#### (b) `forward()` 中的 mask 构建 (line 633-637)

```python
# 现有代码:
attention_masks = make_att_2d_masks(pad_masks, att_masks)        # O(N²) 2D mask
att_2d_masks_4d = self._prepare_attention_masks_4d(attention_masks)  # O(N²) 4D float

# 改为:
if self.attention_implementation == 'flex':
    from fluxvla.engines.utils.model_utils import create_pi05_block_mask
    att_2d_masks_4d = create_pi05_block_mask(
        att_masks, pad_masks, device=att_masks.device)
else:
    attention_masks = make_att_2d_masks(pad_masks, att_masks)
    att_2d_masks_4d = self._prepare_attention_masks_4d(attention_masks)
```

### 3.3 配置文件 — 一行改动

```python
# configs/pi05/pi05_paligemma_libero_10_full_finetune.py
model = dict(
    type='PI05FlowMatching',
    attention_implementation='flex',   # 新增/修改
    ...
)
```

### 3.4 不需要改动的部分

| 组件                            | 原因                                                 |
| ------------------------------- | ---------------------------------------------------- |
| `_forward_transformer_layers`   | 通过 `self.attention_interface` 函数指针切换，零改动 |
| `make_att_2d_masks`             | 保留给 eager 后端使用                                |
| `_prepare_attention_masks_4d`   | 同上                                                 |
| `PI05FlowMatching` 子类         | 不再 override forward，继承父类改动                  |
| `condition_gemma.py`            | 独立 attention 路径                                  |
| `embed_prefix` / `embed_suffix` | mask 值构建不变                                      |

______________________________________________________________________

## 4. 数据流对比

### 4.1 Eager（现有）

```
att_masks (B, 703)  ──→  cumsum  ──→  make_att_2d_masks  ──→  (B, 703, 703) bool
                                              │
                                    _prepare_attention_masks_4d
                                              │
                                    (B, 1, 703, 703) float  ──→  eager_attention_forward
                                                                   │
                                                        Q @ K^T + mask → softmax → @ V
                                                        全 N² 计算，不跳过任何位置
```

显存开销：`B × 703² × 4 bytes (float32)` ≈ B × 1.9 MB

### 4.2 FlexAttention（目标）

```
att_masks (B, 703)  ──→  cumsum  ──→  create_pi05_block_mask  ──→  BlockMask 对象
                                              │                     (block 级元数据 ~KB)
                                              │
                                    flex_attention_forward (compiled Triton kernel)
                                              │
                                    按 block 处理：
                                      EMPTY block → skip
                                      FULL block  → pure attention (无 mask 开销)
                                      PARTIAL     → inline mask_mod
```

显存开销：BlockMask 元数据 ≈ 几 KB（`ceil(703/128)² = 36` 个 block 的元数据）

______________________________________________________________________

## 5. 预期收益

### 5.1 性能

| 指标         | Eager              | FlexAttention       | 说明                    |
| ------------ | ------------------ | ------------------- | ----------------------- |
| 注意力显存   | O(N²)              | O(N)                | 不存储完整 703×703 矩阵 |
| Mask 显存    | ~1.9 MB/样本       | ~KB                 | BlockMask 元数据        |
| 计算量       | 100% blocks        | ~80% blocks         | ~20% EMPTY blocks 跳过  |
| Kernel 效率  | PyTorch matmul × 3 | Fused Triton kernel | IO-aware tiling         |
| **预估加速** | **1x**             | **10-30x**          | attention 部分          |

### 5.2 训练吞吐预估

attention 占整个 forward 的比例约 30-40%（其余是 SigLIP、FFN、embedding 等），假设 attention 加速 15x：

```
整体加速 ≈ 1 / (0.65 + 0.35/15) ≈ 1.46x
```

即训练吞吐提升约 **40-50%**。

______________________________________________________________________

## 6. 风险评估

| 风险                            | 概率   | 影响                                                         | 应对                                      |
| ------------------------------- | ------ | ------------------------------------------------------------ | ----------------------------------------- |
| compile 与 FSDP 冲突            | 低     | 只 compile 一个无状态函数，不涉及参数管理                    | 回退到 eager                              |
| compile 与 grad checkpoint 冲突 | 低     | recompute 时重新调用 compiled kernel，PyTorch 2.6 支持       | 回退到 eager                              |
| 重编译                          | **无** | seq_len=703 固定                                             | N/A                                       |
| head_dim=256 Triton kernel 效率 | 中     | FlexAttention 的 Triton kernel 需要足够 shared memory        | H100 228KB 足够；A100 可能需调 block_size |
| 数值差异                        | 低     | FlexAttention 与 eager 的计算顺序不同，bf16 下可能有微小差异 | 验证 loss 曲线一致                        |

### 回退策略

```python
# 配置切换，一行回退
attention_implementation='eager'   # 回退到原始实现
attention_implementation='flex'    # FlexAttention
```

______________________________________________________________________

## 7. 测试方案

### 7.1 数值一致性（单元测试）

```python
def test_flex_matches_eager():
    """flex vs eager 输出在 bf16 容差内一致"""
    B, L, Hq, Hkv, D = 2, 703, 8, 1, 256
    q = torch.randn(B, Hq, L, D, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(B, Hkv, L, D, device='cuda', dtype=torch.bfloat16)
    v = torch.randn(B, Hkv, L, D, device='cuda', dtype=torch.bfloat16)

    # 构造真实 Pi0.5 mask
    att_masks = torch.zeros(B, L, dtype=torch.bool, device='cuda')
    att_masks[:, 692] = True   # state
    att_masks[:, 693] = True   # act₁
    pad_masks = torch.ones(B, L, dtype=torch.bool, device='cuda')

    # eager
    mask_2d = make_att_2d_masks(pad_masks, att_masks)
    mask_4d = torch.where(mask_2d[:, None], 0.0, -2.3819763e38)
    out_eager, _ = eager_attention_forward(module, q, k, v, mask_4d, scaling)

    # flex
    block_mask = create_pi05_block_mask(att_masks, pad_masks, 'cuda')
    out_flex, _ = flex_attention_forward(module, q, k, v, block_mask, scaling)

    assert torch.allclose(out_eager, out_flex, atol=1e-2, rtol=1e-2)
```

### 7.2 BlockMask 正确性

```python
def test_block_mask_equivalence():
    """BlockMask 可见性与 make_att_2d_masks 逐元素一致"""
    # 构造各种 att_masks pattern
    # 对比 BlockMask 的 mask_mod 与 make_att_2d_masks 的输出
```

### 7.3 端到端训练 Smoke Test

```bash
# 用 flex 跑 10 步训练，验证 loss 正常下降
python -m torch.distributed.launch --nproc_per_node=1 \
  train.py configs/pi05/pi05_paligemma_libero_10_full_finetune.py
```

### 7.4 性能对比

```python
# 对比单步训练时间
for impl in ['eager', 'flex']:
    # 修改配置 attention_implementation=impl
    # 跑 20 步，取后 10 步平均时间
    # 记录 GPU 显存峰值
```

______________________________________________________________________

## 8. 改动量统计

| 文件                                      | 改动                                                     | 行数        |
| ----------------------------------------- | -------------------------------------------------------- | ----------- |
| `fluxvla/engines/utils/model_utils.py`    | 新增 `create_pi05_block_mask` + `flex_attention_forward` | ~50 行      |
| `fluxvla/models/vlas/pi0_flowmatching.py` | 修改 `get_attention_interface` + `forward` mask 分支     | ~15 行      |
| `configs/pi05/*.py`                       | 改 `attention_implementation`                            | 1 行        |
| `test/test_ops/test_flashattn.py`         | 新增测试                                                 | ~60 行      |
| **总计**                                  |                                                          | **~126 行** |
