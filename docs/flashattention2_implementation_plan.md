# FlashAttention-2 Implementation Plan for Pi0.5 Joint Attention

## 1. 问题背景

Pi0.5 训练速度约为 GR00T 的 **1/3**，其中一个关键瓶颈是 Joint Attention 使用 **eager attention**（标准 PyTorch `torch.matmul`），没有利用任何硬件优化的注意力算子。

当前代码（`pi0_flowmatching.py:192-209`）已预留了 `'fa2'` 入口，但直接抛出 `NotImplementedError`：

```python
def get_attention_interface(self):
    if self.attention_implementation == 'fa2':
        raise NotImplementedError('FA2 is not implemented (yet)')
    elif self.attention_implementation == 'eager':
        attention_interface = eager_attention_forward  # ← 当前唯一可用
```

---

## 2. 技术原理

### 2.1 Eager Attention 的问题

标准注意力计算：

```
Attention(Q, K, V) = softmax(Q @ K^T / √d) @ V
```

**计算复杂度**：O(N² · d)  
**显存复杂度**：O(N²)（需要存储完整的 N×N 注意力矩阵）

对于 Pi0.5，Joint Attention 的序列长度 N = prefix_len + suffix_len ≈ 300-500 tokens，加上 GQA expand（1→8 heads），生成的注意力矩阵 shape 为 `(B, 8, N, N)`，在 bf16 下约占 **N² × 8 × 2 bytes**。

关键问题：
1. **HBM 带宽瓶颈**：Q @ K^T 的中间结果需要写入 HBM，再读回来做 softmax，再写入 HBM，再读回来做 @ V —— **3 次 HBM 读写**
2. **显存浪费**：完整的 N×N 注意力矩阵常驻显存
3. **无法利用 Tensor Core 的 tiling 优势**

### 2.2 FlashAttention-2 的核心思想

FlashAttention-2 的核心优化是 **IO-Aware Tiling**：

```
┌─────────────────────────────────┐
│          Standard Attention      │
│  Q,K,V (HBM) → Attn Matrix     │
│  (HBM) → Softmax (HBM) → Out   │
│  3次 HBM 读写                    │
└─────────────────────────────────┘

┌─────────────────────────────────┐
│        FlashAttention-2          │
│  将 Q,K,V 分块 (Tiles)           │
│  每个 Tile 在 SRAM 中完成:       │
│    Q_tile @ K_tile → 局部 softmax │
│    → @ V_tile → 累加到输出        │
│  只需 1次 HBM 读写               │
└─────────────────────────────────┘
```

**关键技术**：
- **Online Softmax**：在 tiling 过程中增量计算 softmax，无需全局 attention 矩阵
- **Kernel Fusion**：matmul + softmax + matmul 融合为单个 CUDA kernel
- **SRAM Tiling**：利用 GPU SRAM（共享内存，带宽 ~19 TB/s）替代 HBM（带宽 ~3 TB/s）

**效果**：
- 显存：O(N²) → **O(N)**（不再存储完整注意力矩阵）
- 速度：减少 HBM 访问次数，**2-4x 加速**（IO-bound 场景更显著）

### 2.3 PyTorch SDPA 的角色

PyTorch 2.x 提供了 `F.scaled_dot_product_attention`（SDPA），它是一个**调度器**，自动选择最优后端：

```
F.scaled_dot_product_attention(Q, K, V, attn_mask)
    ├── FlashAttention 后端（无自定义 mask 时）
    ├── Memory-Efficient Attention 后端（xformers 算法）
    └── Math 后端（标准 PyTorch，等同 eager）
```

优势：支持 **任意 bool 类型 attention mask**，同时仍能获得优化后端的加速。

---

## 3. Pi0.5 Joint Attention 的特殊挑战

### 3.1 非标准 Mask 模式

Pi0.5 的注意力 mask 通过 `make_att_2d_masks` 的 cumsum 机制构建，**不是标准 causal mask**：

```
Token 类型:     [img₁ img₂ ... lang₁ lang₂ ... | state | act₁ act₂ ... actₙ]
att_mask值:     [0    0        0     0          | 1     | 1    0    ... 0    ]
cumsum:         [0    0        0     0          | 1     | 2    2    ... 2    ]

生成的 2D mask（1=可见, 0=屏蔽）:

                img  lang  state  act₁  act₂..actₙ
img             ✓     ✓     ✗      ✗      ✗
lang            ✓     ✓     ✗      ✗      ✗
state           ✓     ✓     ✓      ✗      ✗
act₁            ✓     ✓     ✓      ✓      ✗
act₂..actₙ     ✓     ✓     ✓      ✓      ✓  (彼此双向)
```

关键特征：
- **Prefix 块**：完全双向（image 和 language tokens 互相可见）
- **State**：因果边界（能看 prefix，不能看 actions）
- **Action 块**：除 act₁ 外，**彼此双向**（能看到所有 prefix+state+actions）

### 3.2 为什么 `causal=True` 是近似而非精确

标准 causal mask（下三角矩阵）：

```
                img₁  img₂  lang  state  act₁  act₂
img₁             ✓     ✗     ✗     ✗      ✗     ✗
img₂             ✓     ✓     ✗     ✗      ✗     ✗    ← img₂ 看不到后面的 lang!
lang             ✓     ✓     ✓     ✗      ✗     ✗
state            ✓     ✓     ✓     ✓      ✗     ✗
act₁             ✓     ✓     ✓     ✓      ✓     ✗
act₂             ✓     ✓     ✓     ✓      ✓     ✓
```

**差异**：
1. Prefix 内 image tokens 变成单向（img₁ 看不到 img₂），破坏了 image-language fusion
2. Action tokens 变成严格因果（act₁ 看不到 act₂），破坏了 action 间双向注意力

### 3.3 GQA (Grouped Query Attention) 配置

```
num_attention_heads = 8    (Q heads)
num_key_value_heads = 1    (K/V heads)
num_key_value_groups = 8   (每个 KV head 被 8 个 Q head 共享)
head_dim = 256
```

- Eager 实现：通过 `repeat_kv` 将 KV 从 1 head 扩展到 8 heads
- SDPA：通过 `enable_gqa=True` 原生支持，无需扩展
- flash_attn：原生支持不同 Q/KV head 数

---

## 4. 方案设计：双后端策略

### 4.1 总体架构

```
attention_implementation config
    │
    ├── 'eager'  → eager_attention_forward()     [现有，保持不变]
    ├── 'sdpa'   → sdpa_attention_forward()      [新增，推荐默认]
    └── 'fa2'    → flash_attention_forward()     [新增，极速模式]
```

### 4.2 性能对比预估

| 后端 | API | Mask 支持 | 预估加速 | Mask 保真度 | 推荐场景 |
|------|-----|----------|---------|------------|---------|
| **eager** | `torch.matmul` | 任意 4D mask | 1x (baseline) | 精确 | 调试 |
| **sdpa** | `F.scaled_dot_product_attention` | 任意 bool mask | **~5-10x** | **精确** | **训练默认** |
| **fa2** | `flash_attn_func` | 仅 causal | **~20-75x** | 近似 | 推理/实验 |

### 4.3 推荐配置

```python
# 训练（保持精确语义）
model = dict(
    type='PI05FlowMatching',
    attention_implementation='sdpa',  # 推荐
    ...
)

# 需要极致速度时（接受 causal 近似）
model = dict(
    type='PI05FlowMatching',
    attention_implementation='fa2',  # opt-in
    ...
)
```

---

## 5. 详细实现方案

### 5.1 新增 `sdpa_attention_forward`

**位置**：`fluxvla/engines/utils/model_utils.py`（在 `eager_attention_forward` 之后）

```python
def sdpa_attention_forward(
    module: nn.Module,
    query: torch.Tensor,       # (B, Hq, L, D) = (B, 8, L, 256)
    key: torch.Tensor,         # (B, Hkv, L, D) = (B, 1, L, 256)
    value: torch.Tensor,       # (B, Hkv, L, D) = (B, 1, L, 256)
    attention_mask: Optional[torch.Tensor],  # (B, 1, L, L), 0.0/-2.38e38
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    """使用 PyTorch SDPA 的注意力计算，支持任意 mask 和 GQA。"""

    # 1. 将 additive mask 转为 bool mask（SDPA 对 bool mask 有更好的 kernel 支持）
    #    原始 mask: 0.0 = 可见, -2.38e38 = 屏蔽
    #    bool mask: True = 可见, False = 屏蔽
    if attention_mask is not None:
        attention_mask = attention_mask[:, :, :, :key.shape[-2]]
        bool_mask = (attention_mask > -1.0)  # 0.0 → True, -2.38e38 → False
    else:
        bool_mask = None

    # 2. 调用 SDPA（原生 GQA 支持，无需 repeat_kv）
    #    输入 shape: Q (B, Hq, L, D), K/V (B, Hkv, L, D)
    attn_output = F.scaled_dot_product_attention(
        query, key, value,
        attn_mask=bool_mask,
        dropout_p=dropout if module.training else 0.0,
        scale=scaling,
        enable_gqa=True,  # 允许 Hq != Hkv
    )
    # 输出 shape: (B, Hq, L, D)

    # 3. 转置为 (B, L, Hq, D) 以匹配 eager 的输出格式
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, None  # 不返回 attn_weights（SDPA 不支持）
```

**关键设计决策**：
- 使用 **bool mask** 而非 additive mask：SDPA 对 bool mask 可以走 FlashAttention 或 memory-efficient 后端
- `enable_gqa=True`：PyTorch 2.5+ 支持，自动处理 Hq=8, Hkv=1
- 返回 `None` 作为 attn_weights：调用方 `_forward_transformer_layers` line 402 已用 `_, _` 丢弃

### 5.2 新增 `flash_attention_forward`

**位置**：同上

```python
def flash_attention_forward(
    module: nn.Module,
    query: torch.Tensor,       # (B, Hq, L, D)
    key: torch.Tensor,         # (B, Hkv, L, D)
    value: torch.Tensor,       # (B, Hkv, L, D)
    attention_mask: Optional[torch.Tensor],  # 接收但忽略
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    """使用 flash_attn_func 的注意力计算，causal=True 近似。"""
    from flash_attn.flash_attn_interface import flash_attn_func

    # 1. 转置: (B, H, L, D) → (B, L, H, D)（flash_attn 要求的格式）
    q = query.transpose(1, 2)   # (B, L, Hq, D)
    k = key.transpose(1, 2)     # (B, L, Hkv, D)
    v = value.transpose(1, 2)   # (B, L, Hkv, D)

    # 2. 调用 flash_attn（原生 GQA 支持：Hq != Hkv 时自动分组）
    attn_output = flash_attn_func(
        q, k, v,
        dropout_p=dropout if module.training else 0.0,
        softmax_scale=scaling,
        causal=True,  # 近似：prefix 变单向，actions 变因果
    )
    # 输出 shape: (B, L, Hq, D) — 已经是转置后的格式

    return attn_output, None
```

**注意**：
- `flash_attn_func` 原生接受不同的 Q/KV head 数，无需 `repeat_kv`
- `attention_mask` 被忽略（flash_attn 2.5.x 不支持自定义 mask）
- 输出已经是 `(B, L, H, D)` 格式，与 eager 的 `.transpose(1, 2).contiguous()` 后一致

### 5.3 更新 `get_attention_interface`

**位置**：`fluxvla/models/vlas/pi0_flowmatching.py` lines 192-209

```python
def get_attention_interface(self):
    if self.attention_implementation == 'fa2':
        attention_interface = flash_attention_forward   # 替换 NotImplementedError
    elif self.attention_implementation == 'sdpa':
        attention_interface = sdpa_attention_forward    # 新增
    elif self.attention_implementation == 'flex':
        raise NotImplementedError('Flex attention is not implemented (yet)')
    elif self.attention_implementation == 'eager':
        attention_interface = eager_attention_forward
    elif self.attention_implementation == 'xformer':
        raise NotImplementedError('Xformer attention is not implemented (yet)')
    else:
        raise ValueError(
            f'Invalid attention implementation: {self.attention_implementation}. '
            "Expected one of ['fa2', 'sdpa', 'flex', 'eager', 'xformer'].")
    return attention_interface
```

### 5.4 `att_output` reshape 兼容性

`_forward_transformer_layers` line 412：

```python
att_output = att_output.reshape(batch_size, -1, 1 * 8 * head_dim)
```

三种后端的输出都是 `(B, L, H, D)` 即 `(B, L, 8, 256)` 格式：
- **eager**：`.transpose(1, 2).contiguous()` → `(B, L, H, D)` ✓
- **sdpa**：`.transpose(1, 2).contiguous()` → `(B, L, H, D)` ✓  
- **fa2**：flash_attn 直接输出 `(B, L, H, D)` ✓

reshape 到 `(B, L, 8*256)` = `(B, L, 2048)` 后进入各自的 `o_proj`，**无需任何修改**。

---

## 6. 测试方案

### 6.1 数值正确性测试

```python
def test_sdpa_matches_eager():
    """验证 SDPA 与 eager 在 bf16 容差内数值一致"""
    # 构造 Pi0.5 配置: B=2, L=64, Hq=8, Hkv=1, D=256
    # 构造真实 mask pattern（使用 make_att_2d_masks）
    # 对比 eager_attention_forward vs sdpa_attention_forward
    # assert torch.allclose(out_sdpa, out_eager, atol=1e-2, rtol=1e-2)
```

### 6.2 FA2 输出合理性测试

```python
def test_fa2_output_sanity():
    """验证 FA2 输出形状正确、无 NaN/Inf"""
    # 使用 GQA 配置
    # 验证 shape, dtype, no NaN/Inf
```

### 6.3 Gradient Checkpointing 兼容性

```python
def test_sdpa_with_gradient_checkpointing():
    """验证 SDPA 在 gradient checkpointing 下可正常反向传播"""
    # torch.utils.checkpoint.checkpoint 包裹注意力计算
    # 验证 loss.backward() 不报错
```

---

## 7. 改动范围总结

| 文件 | 改动类型 | 说明 |
|------|---------|------|
| `fluxvla/engines/utils/model_utils.py` | **新增** 2 个函数 | `sdpa_attention_forward`, `flash_attention_forward` |
| `fluxvla/models/vlas/pi0_flowmatching.py` | **修改** 1 个函数 | `get_attention_interface` 添加 sdpa/fa2 分支 |
| `test/test_ops/test_flashattn.py` | **新增** 3 个测试 | 数值一致性、输出合理性、gradient checkpointing |

**不需要改动**：
- `_forward_transformer_layers` —— 通过函数指针切换，零结构变更
- `make_att_2d_masks` / `_prepare_attention_masks_4d` —— mask 构建保持不变
- `PI05FlowMatching` —— 继承父类变更
- `condition_gemma.py` —— 其 attention 路径与 joint attention 独立
- 训练配置 —— 只需将 `attention_implementation` 从 `'eager'` 改为 `'sdpa'`

---

## 8. 风险和注意事项

1. **PyTorch 版本要求**：`enable_gqa=True` 需要 PyTorch ≥ 2.5，需确认环境版本
2. **SDPA 后端选择**：传入 bool mask 时，SDPA 可能不选 FlashAttention kernel 而选 memory-efficient kernel，仍有加速但可能不是最优
3. **FA2 语义变更**：使用 `fa2` 后端时，prefix 双向注意力变为单向，可能影响 image-language 融合质量，建议训练时先用 `sdpa`
4. **head_dim=256**：FlashAttention-2 对 head_dim > 128 的支持因版本而异，需验证 flash_attn 2.5.x 是否支持 head_dim=256
5. **`attn_weights` 不可用**：SDPA 和 FA2 都不返回注意力权重，如有可视化需求需切回 eager
