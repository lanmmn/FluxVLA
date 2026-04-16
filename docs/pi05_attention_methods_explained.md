# Pi0.5 Attention 实现方式详解

> 环境参数：B=8, L=703, Hq=8, Hkv=1, head_dim=256, 18层, bf16, NVIDIA A100-SXM4-80GB

## 背景：Pi0.5 的注意力掩码模式

Pi0.5 的输入序列由 4 个段（segment）拼接而成：

```
[IMG(512) | LANG(180) | STATE(1) | ACTION(10)]  →  L=703
```

注意力规则是**后段可见前段，前段不可见后段**（因果式分段掩码），通过两个 1D 掩码实现：

- **`att_masks`** (B, L)：标记段边界位置（img[0]=True, lang[0]=True, state[0]=True, action[0]=True）
- **`pad_masks`** (B, L)：标记哪些位置是真实 token（非 padding）

核心逻辑（`make_att_2d_masks`）：

```python
cumsum = torch.cumsum(att_masks, dim=1)       # 每个位置获得段 ID
att_2d = cumsum[:, None, :] <= cumsum[:, :, None]  # q_seg >= kv_seg → 可见
pad_2d = pad_masks[:, None, :] & pad_masks[:, :, None]  # 双方都非 pad
att_2d = att_2d & pad_2d
```

这 4 种 attention 方法都是对**同一个掩码逻辑**的不同实现。

---

## 1. Eager Attention

**实现文件**：`fluxvla/engines/utils/model_utils.py:253-279`

### 核心流程

```python
def eager_attention_forward(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    key_states = repeat_kv(key, module.num_key_value_groups)      # (B, Hkv, L, D) → (B, Hq, L, D)
    value_states = repeat_kv(value, module.num_key_value_groups)  # 同上

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling  # (B, Hq, L, L)
    if attention_mask is not None:
        attn_weights = attn_weights + causal_mask          # 加掩码（0 或 -2.38e38）
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_output = torch.matmul(attn_weights, value_states)  # (B, Hq, L, D)
    attn_output = attn_output.transpose(1, 2).contiguous()  # (B, L, Hq*D)
    return attn_output, attn_weights
```

### 掩码格式

`attention_mask` 是 **4D float tensor**，shape `(B, 1, L, L)`，值为：

- `0.0` → 允许关注
- `-2.3819763e38`（接近 -inf）→ 屏蔽

通过 `_prepare_attention_masks_4d` 将 bool 掩码转换为 float：

```python
torch.where(att_4d_bool, 0.0, -2.3819763e38).to(DTYPE)
```

### GQA 处理

`repeat_kv` 将 KV head 从 `(B, Hkv=1, L, D=256)` 扩展为 `(B, Hq=8, L, D=256)`，即 1 个 KV head 重复 8 次。

### 显存分析

- **注意力权重矩阵**：`B × Hq × L × L × sizeof(bfloat16)` = 8 × 8 × 703 × 703 × 2 = **~50.5 MB/层**
- **4D float 掩码**：`B × 1 × L × L × sizeof(bfloat16)` = **~6.3 MB/层**
- softmax 在 float32 下计算，需要 `B × Hq × L × L × 4` = **~101 MB/层**的临时显存
- 18 层合计 **~4,894 MB**

### 速度分析

- seq_len=703 很短，matmul `(B×Hq, L, D) @ (B×Hq, D, L)` 计算量小
- 没有额外编译开销，直接 CUDA kernel 调用
- 每层 **~4.20 ms**，18 层 **~67.33 ms**

### 优缺点

| 优点 | 缺点 |
|------|------|
| 实现最简单，无编译 | 显存占用大（4D mask + 完整 softmax 矩阵） |
| 无环境依赖 | O(N²) 显存，长序列会爆 |
| debug 友好（可查看 attn_weights） | GQA 需要 repeat_kv，浪费计算 |

---

## 2. FlexAttention (非编译模式)

**实现文件**：`fluxvla/engines/utils/model_utils.py:215-250`

### 核心流程

```python
_flex_attention_compiled = torch.compile(flex_attention)  # 模块级全局编译一次

def flex_attention_forward(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    attn_output = _flex_attention_compiled(
        query, key, value,
        block_mask=attention_mask,
        scale=scaling,
        enable_gqa=True,     # 原生 GQA 支持，无需 repeat_kv
    )
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, None
```

### 掩码格式

使用 `BlockMask`（来自 `torch.nn.attention.flex_attention.create_block_mask`）：

```python
def create_pi05_block_mask(att_masks, pad_masks, device):
    segment_ids = torch.cumsum(att_masks.int(), dim=1)  # (B, L)

    def mask_mod(b, h, q_idx, kv_idx):
        q_seg = segment_ids[b, q_idx]
        kv_seg = segment_ids[b, kv_idx]
        segment_ok = q_seg >= kv_seg
        pad_ok = pad_masks[b, q_idx] & pad_masks[b, kv_idx]
        return segment_ok & pad_ok

    return create_block_mask(mask_mod, B=B, H=None, Q_LEN=L, KV_LEN=L, device=device)
```

`BlockMask` 是一种**块级稀疏掩码**的元数据表示：
- 不存储完整的 `(B, H, L, L)` 张量
- 仅记录哪些块（block）需要计算，哪些可以跳过
- 掩码函数 `mask_mod` 是纯 Python，由 `torch.compile` 编译成 Triton kernel

### GQA 处理

`enable_gqa=True` 让 FlexAttention 原生支持 GQA，**无需 repeat_kv**，直接传入 `(B, Hkv=1, L, D)` 的 K/V。

### 为什么在 A100 上退回非编译模式

`torch.compile(flex_attention)` 在编译时会生成 Triton kernel。对于 `head_dim=256`：

- Triton kernel 的 `tl.dot` 需要将 Q、K 块加载到 shared memory
- 每个 Q 块大小：`BLOCK_M × 256 × sizeof(bfloat16)`，K 块类似
- A100 的 shared memory 限制为 **166 KB/SM**
- 当 head_dim=256 时，仅 Q 块就需要 ~128KB，加上 K 块和累加器远超 166KB
- 编译失败，回退到 **非编译的 Python 解释器路径**，逐元素执行 mask_mod

### 显存分析

- BlockMask 只存储元数据（~KB 级别），不存储 4D float mask
- 但非编译模式下，实际计算路径退化为类似 Eager 的逐步操作
- 18 层 **~1,851 MB**（比 Eager 少，因为不分配 4D mask + 完整 softmax 缓存）

### 速度分析

- 非编译模式下每层 **~31.24 ms**，18 层 **~575.47 ms**
- 比 Eager 慢 **~7.4x**，因为：
  1. 没有 fused kernel，每个操作都是单独的 CUDA kernel
  2. `mask_mod` 在 Python 层逐元素求值
  3. 额外的 BlockMask 解析开销

### 优缺点

| 优点 | 缺点 |
|------|------|
| 原生 GQA（无需 repeat_kv） | head_dim=256 在 A100 上编译失败 |
| BlockMask 节省显存 | 非编译模式极慢（~8.5x slower） |
| 掩码逻辑用 Python 定义，灵活 | 依赖 torch.compile + Triton |
| 理论上编译后可超越 Eager | 编译环境要求苛刻 |

---

## 3. SDPA (带 bool mask)

**实现**：`torch.nn.functional.scaled_dot_product_attention`，项目中的 benchmark 函数：

```python
def sdpa_attn(q, k, v, bool_mask, scaling):
    k_exp = repeat_kv(k, NUM_Q_HEADS // NUM_KV_HEADS)
    v_exp = repeat_kv(v, NUM_Q_HEADS // NUM_KV_HEADS)
    out = F.scaled_dot_product_attention(
        q, k_exp, v_exp,
        attn_mask=bool_mask,   # (B, Hq, L, L) bool tensor
        scale=scaling,
    )
    return out.transpose(1, 2).contiguous()
```

### 掩码格式

SDPA 的 `attn_mask` 参数支持两种类型：

| 类型 | 行为 |
|------|------|
| `float` tensor | 直接加到注意力分数上 |
| `bool` tensor | `True` = 允许关注，`False` = 屏蔽（设为 -inf） |

本项目使用 **bool mask**，shape `(B, Hq, L, L)`：

```python
def make_sdpa_bool_mask(pad_masks, att_masks):
    cumsum = torch.cumsum(att_masks.int(), dim=1)
    att_2d = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d = pad_masks[:, None, :] & pad_masks[:, :, None]
    att_2d = att_2d & pad_2d
    return att_2d[:, None, :, :].expand(-1, NUM_Q_HEADS, -1, -1)
```

### 后端选择逻辑

PyTorch SDPA 有 3 个后端，按优先级自动选择：

1. **Flash Attention**（最快）：要求 `attn_mask` 为 `None` 或特定格式
2. **Memory-Efficient Attention**（xformers）：支持 float mask
3. **Math 后端**（fallback）：朴素实现，等同 Eager

**关键问题**：当传入 `bool` 类型的 `attn_mask` 时，SDPA **无法使用 Flash Attention 后端**，因为 Flash Attention API 不支持 bool mask（只支持 causal 标志或无 mask）。SDPA 退回到 **Memory-Efficient Attention** 或 **Math 后端**。

在 A100 上实测，bool mask 导致 SDPA 退回 Math 后端，每层 **~5.16 ms**，18 层 **~93.59 ms**。

### GQA 处理

SDPA 本身不原生支持 GQA，需要手动 `repeat_kv` 将 KV head 扩展到与 Q head 一致。

### 显存分析

- bool mask：(B × Hq × L × L × 1 byte) = 8 × 8 × 703 × 703 × 1 = **~25.3 MB/层**
- 比 float mask 节省一半，但仍然是 O(N²)
- 18 层 **~2,169 MB**

### 优缺点

| 优点 | 缺点 |
|------|------|
| PyTorch 内置，无需额外依赖 | bool mask 阻止 Flash Attention 后端 |
| API 简洁 | GQA 需要 repeat_kv |
| 无编译开销 | 退回 Math 后端时比 Eager 更慢 |

---

## 4. SDPA (无 mask)

**实现**：与 SDPA (带 mask) 相同，但不传入 `attn_mask`：

```python
def sdpa_attn_no_mask(q, k, v, _, scaling):
    k_exp = repeat_kv(k, NUM_Q_HEADS // NUM_KV_HEADS)
    v_exp = repeat_kv(v, NUM_Q_HEADS // NUM_KV_HEADS)
    out = F.scaled_dot_product_attention(
        q, k_exp, v_exp,
        scale=scaling,        # 无 attn_mask
    )
    return out.transpose(1, 2).contiguous()
```

### 为什么这么快

无 mask 时，SDPA 自动选择 **Flash Attention 后端**：

- Flash Attention 是一个 IO-aware 的 fused kernel
- 不在 GPU HBM 上 materialize 完整的 (B, H, L, L) 注意力矩阵
- 利用 SRAM 分块计算，将 Q、K、V 分块加载到片上 SRAM
- 在 SRAM 中完成 softmax + matmul，只写出最终输出
- 显存复杂度从 O(N²) 降到 O(N)

### 显存分析

- 无需存储任何 mask
- 注意力矩阵不写入 HBM
- 18 层 **~1,196 MB**（最低）

### 注意

这是**理论最优上界**。实际训练中 Pi0.5 必须使用分段掩码，不能完全无 mask。此 benchmark 仅用于衡量"如果掩码不是瓶颈，SDPA+Flash 能多快"。

### 优缺点

| 优点 | 缺点 |
|------|------|
| 最快（Flash Attention fused kernel） | 不支持自定义 mask |
| 显存最低（O(N) 而非 O(N²)） | 不适用于 Pi0.5 的分段掩码 |
| 无 repeat_kv 的显存开销 | 仅作为性能上界参考 |

---

## 汇总对比

```
                    Eager          Flex (非编译)    SDPA (bool mask)  SDPA (无 mask)
                 ─────────────  ───────────────  ────────────────  ────────────────
实现复杂度        ★☆☆☆☆         ★★★☆☆            ★★☆☆☆            ★☆☆☆☆
掩码格式          4D float       BlockMask        4D bool           无
GQA 处理          repeat_kv      原生支持          repeat_kv         repeat_kv
Flash Attention    ✗              ✗               ✗ (bool mask)    ✓
编译依赖          无             torch.compile    无                无
单层延迟          4.20 ms        31.24 ms          5.16 ms           1.32 ms
18 层延迟         67.33 ms      575.47 ms         93.59 ms          24.62 ms
18 层显存         4,894 MB      1,851 MB          2,169 MB          1,196 MB
```

### 关键瓶颈

1. **head_dim=256** 是当前所有问题的根源：导致 Triton shared memory 溢出，FlexAttention 无法编译
2. **bool mask 阻止 Flash Attention**：SDPA 收到 bool mask 后退回 Math 后端
3. **seq_len=703 较短**：Eager 的 O(N²) 开销不明显，fused kernel 的优势难以发挥

### 路径建议

| 时间线 | 方案 | 预期效果 |
|--------|------|----------|
| **短期** | 改回 `attention_implementation='eager'` | 单层 4.20ms，当前最快 |
| **中期** | 升级 PyTorch 2.7+ / Triton 3.2 | FlexAttention 编译可通过，预计 2-3ms/层 |
| **中期** | 降低 head_dim 到 128 | FlexAttention 编译通过 + 减少计算量 |
| **长期** | 实现 Flash Attention 2 后端 | 支持 GQA + 自定义 mask，预计 1-2ms/层 |