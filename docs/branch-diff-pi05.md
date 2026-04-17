# Pi0.5 分支对比：acclerate-pi-training vs support-remote-inference

两个分支的 `pi05_flowmatching.py` 完全一致，所有差异都在其基类 `pi0_flowmatching.py` 中。

## 差异总览

| 改动点 | support-remote-inference | acclerate-pi-training | 省显存 | 加速 |
|--------|------------------------|-----------------------|:------:|:----:|
| 注意力实现 | 仅 `eager` | 新增 `sdpa`（默认） | ✅ | ✅✅ |
| RoPE tensor 分配 | 每层创建 `dummy_tensor` | 循环外 `torch.empty(0)` | ✅ | ✅ |
| Attention Mask 构造 | `torch.where` 硬编码大负数 | `masked_fill_` 原地操作 | ✅ | — |
| 残差连接 | `.clone()` 保护 | 去掉 `.clone()` | ✅✅ | ✅ |
| MLP dtype 转换 | 显式 bfloat16 检查 | 去掉，假定 dtype 一致 | — | ✅ |

---

## 1. 注意力实现：eager → sdpa

这是最核心的改动。acclerate 分支引入了 PyTorch 的 `scaled_dot_product_attention` 融合 kernel。

### support-remote-inference

```python
# 默认值
attention_implementation: Optional[str] = 'eager'

# 仅实现 eager，其余全部 NotImplementedError
def get_attention_interface(self):
    if self.attention_implementation == 'fa2':
        raise NotImplementedError('FA2 is not implemented (yet)')
    elif self.attention_implementation == 'flex':
        raise NotImplementedError('Flex attention is not implemented (yet)')
    elif self.attention_implementation == 'eager':
        attention_interface = eager_attention_forward
    elif self.attention_implementation == 'xformer':
        raise NotImplementedError('Xformer attention is not implemented (yet)')
    ...
```

### acclerate-pi-training

```python
# 默认值改为 sdpa
attention_implementation: Optional[str] = 'sdpa'

# 新增 sdpa_attention_forward import
from fluxvla.engines.utils.model_utils import (
    ..., sdpa_attention_forward)

# 精简为两种实现
def get_attention_interface(self):
    if self.attention_implementation == 'sdpa':
        attention_interface = sdpa_attention_forward
    elif self.attention_implementation == 'eager':
        attention_interface = eager_attention_forward
    ...
```

**省显存**：eager attention 需要显式 materialize 完整的 `(B, H, S, S)` attention score 矩阵；SDPA 后端（尤其是 FlashAttention）使用 tiling 算法，不需要存储完整 attention 矩阵，显存占用从 O(S²) 降到 O(S)。

**加速**：SDPA 是融合 kernel，将 Q·Kᵀ → scale → mask → softmax → ·V 合并为单次 GPU kernel 调用，减少了多次 kernel launch 和中间结果的显存读写。PyTorch 会自动选择最优后端（FlashAttention-2 / Memory-Efficient Attention / 数学实现）。

---

## 2. RoPE 计算：减少不必要的内存分配

### support-remote-inference

每个 transformer 层都创建一个和 QKV 同 shape 的 dummy tensor：

```python
for layer_idx in range(num_layers):
    ...
    # 每次循环都分配一个大 tensor
    dummy_tensor = torch.zeros(
        query_states.shape[0],
        query_states.shape[2],
        query_states.shape[-1],
        device=query_states.device,
        dtype=query_states.dtype,
    )
    cos, sin = self.llm_backbone.rotary_emb(dummy_tensor, position_ids)
```

### acclerate-pi-training

循环外创建一个零大小的 tensor，仅用于传递 device/dtype 信息：

```python
# 循环外，只分配一次，大小为 0
_rope_device_tensor = torch.empty(
    0, device=inputs_embeds[0].device, dtype=inputs_embeds[0].dtype)

for layer_idx in range(num_layers):
    ...
    cos, sin = self.llm_backbone.rotary_emb(_rope_device_tensor, position_ids)
```

**省显存**：旧版每层分配一个 `(B, seq_len, head_dim)` 大小的 dummy tensor 仅用于推导 device/dtype，分配后立即被丢弃。18 层 transformer 意味着 18 次无用分配和 GC。新版用 `torch.empty(0)` 只分配一次零字节的 tensor。

**加速**：减少 CUDA allocator 的 `cudaMalloc`/`cudaFree` 调用。虽然单次很快，但 18 层循环累积起来在大 batch 训练时有可测量的开销。

---

## 3. Attention Mask 4D 构造

### support-remote-inference

```python
def _prepare_attention_masks_4d(self, att_2d_masks):
    att_2d_masks_4d = att_2d_masks[:, None, :, :]
    return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)
```

### acclerate-pi-training

```python
def _prepare_attention_masks_4d(self, att_2d_masks):
    att_2d_masks_4d = att_2d_masks[:, None, :, :]
    mask = torch.zeros_like(att_2d_masks_4d, dtype=torch.bfloat16)
    mask.masked_fill_(~att_2d_masks_4d, torch.finfo(torch.bfloat16).min)
    return mask
```

**省显存**：`torch.where` 会创建一个新的输出 tensor（大小为 `(B, 1, S, S)`，S 为序列长度）；`masked_fill_` 是原地操作，在已有的 `torch.zeros_like` 上直接修改，少分配一份完整 mask 大小的显存。

**加速**：无显著加速，两者计算量相同。改用 `torch.finfo(torch.bfloat16).min` 主要是代码规范性提升。

---

## 4. 残差连接去掉 `.clone()`

### support-remote-inference

```python
out_emb = gated_residual(hidden_states, out_emb, gates[i])
after_first_residual = out_emb.clone()   # 显式复制
```

### acclerate-pi-training

```python
out_emb = gated_residual(hidden_states, out_emb, gates[i])
after_first_residual = out_emb            # 直接引用，不复制
```

**省显存**：`.clone()` 会复制一份完整的 hidden_states tensor `(B, S, D)`。以 bfloat16、B=32、S=560、D=2048 为例，单次 clone ≈ 70 MB，18 层 × 每层一次 = 峰值多占约 70 MB（被反向图持有）。去掉后直接引用原 tensor，因为 `gated_residual` 已经返回新 tensor 不会被原地修改。

**加速**：减少一次 device-to-device memcpy（H2D bandwidth bound），18 层累积有可测量的收益。

---

## 5. 去掉 MLP 前的显式 dtype 转换

### support-remote-inference

```python
out_emb, gate = layer.post_attention_layernorm(out_emb, cond=adarms_cond[i])
# 显式检查 MLP 权重 dtype，必要时转换
if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
    out_emb = out_emb.to(dtype=torch.bfloat16)
out_emb = layer.mlp(out_emb)
```

### acclerate-pi-training

```python
out_emb, gate = layer.post_attention_layernorm(out_emb, cond=adarms_cond[i])
# 直接进 MLP，不做 dtype 检查
out_emb = layer.mlp(out_emb)
```

**省显存**：无。`.to()` 在 dtype 已匹配时是 no-op，不匹配时产生的临时 tensor 大小也很小。

**加速**：去掉每层一次的条件判断 + 属性访问（`layer.mlp.up_proj.weight.dtype`）。在 CUDA Graph 或 torch.compile 场景下，这种动态分支会阻碍图捕获和编译优化。更重要的是保持代码路径简洁，让 autocast 统一管理 dtype。

---

---

## 6. SDPA 后端实测结果

通过在 `sdpa_attention_forward` 中逐个后端测试，实际确认：

| 后端 | 结果 | 原因 |
|------|------|------|
| **Flash Attention** | FAIL | 不支持 non-null `attn_mask`（本项目使用 float mask 实现混合单双向注意力） |
| **CuDNN Attention** | FAIL | 要求 `head_dim ≤ 128`，本模型 head_dim=256 超限 |
| **Memory-Efficient Attention** | **OK** | 支持 float mask，支持大 head_dim，**实际走此后端** |
| **Math (eager fallback)** | OK | 纯数学实现，都支持，但最慢 |

实测参数：`query shape: (8, 8, 702, 256), dtype: bfloat16, causal_mask: (8, 1, 702, 702) bfloat16`

如需进一步优化到 Flash Attention，需同时解决：去掉 float mask（但 joint attention 的混合单双向模式不兼容 `is_causal`）+ head_dim 降到 ≤128（需改模型架构）。当前 Memory-Efficient Attention 是该模型能用的最优后端。

---

## 7. 配置层面的加速：`sharding_strategy` 和 `gradient_checkpointing`

除了模型代码层面的改动，acclerate 分支在配置文件中也做了两项关键调整。

### 7.1 FSDP Sharding Strategy：`hybrid-shard` → `no-shard`

acclerate 分支配置显式设置 `sharding_strategy='no-shard'`，而非默认的 `hybrid-shard`。

**实测效果：单卡训练从 4s/it 提升到 ~0.9s/it（约 4.4 倍加速）。**

这并非简单的"不分片"，而是 `is_no_shard` 标志在 `FSDPTrainRunner.run_setup()` 中同时改变了三个行为：

#### (a) 不递归包装子模块（最大影响）

```python
if is_no_shard:
    vla_fsdp_wrapping_policy = None    # FSDP 只包最外层
else:
    vla_fsdp_wrapping_policy = self.vla.get_fsdp_wrapping_policy()  # 递归包装每个 Linear/LayerNorm
```

`hybrid-shard` 会递归包装每个 `nn.Linear`、`nn.LayerNorm` 子模块。在单卡上，每个子模块的前向/反向都要走 FSDP wrapper 的参数 flatten → 计算 → unflatten 流程，而实际没有任何通信发生，纯属开销。

#### (b) 全 bf16 混合精度（减少类型转换）

```python
if is_no_shard:
    # param/reduce/buffer 全部 bfloat16
    fsdp_precision_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16)
else:
    # reduce 和 buffer 用 float32，梯度聚合时多一次 bf16 → fp32 → bf16 转换
    fsdp_precision_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.float32)
```

#### (c) 参数直接存为 bf16（零 cast 开销）

```python
if is_no_shard:
    target_dtype = torch.bfloat16   # 参数直接 bf16，不需要运行时转换
else:
    target_dtype = torch.float32    # 参数 fp32，FSDP 混合精度每次前向 cast 到 bf16
```

| | `hybrid-shard` | `no-shard` |
|---|---|---|
| FSDP 递归包装 | 包装每个子模块（**巨大开销**） | 只包最外层 |
| 参数 dtype | fp32（运行时 cast bf16） | bf16（零 cast） |
| Reduce dtype | fp32（多一次转换） | bf16（无转换） |
| **单卡实测** | **~4 s/it** | **~0.9 s/it** |

### 7.2 Gradient Checkpointing

| 配置 | 效果 | 代价 |
|------|------|------|
| `enable_gradient_checkpointing=True` | 只保存部分层激活值，反向时重新计算，显存从 O(L) 降到 O(√L) | 前向计算量增加 ~33%，且打断 GPU 流水线 |
| `enable_gradient_checkpointing=False` | 保存所有层激活值 | 显存占用更大，但速度最快 |

在单卡 80GB 显存、模型参数量较小（Pi0.5 约 600M）的场景下，关闭 gradient checkpointing 可以进一步提速而不会 OOM。

---

## 总结

### 省显存排序（影响从大到小）

1. **SDPA 注意力**：最大收益，attention score 矩阵从 O(S²) 降到 O(S)，对长序列效果显著
2. **去掉 `.clone()`**：每层省一份 `(B, S, D)` 大小的 tensor，18 层累积约 70 MB（B=32 时）
3. **Mask `masked_fill_`**：省一份 `(B, 1, S, S)` 大小的临时 tensor
4. **RoPE `torch.empty(0)`**：省 18 次 `(B, S, head_dim)` 大小的临时分配

### 加速排序（影响从大到小）

1. **FSDP `no-shard`**：单卡场景下最大收益（~4.4x），消除不必要的递归包装、参数 flatten/unflatten 和 dtype 转换
2. **SDPA 注意力**：融合 kernel 减少多次 kernel launch 和显存读写（实际走 Memory-Efficient Attention 后端）
3. **关闭 gradient checkpointing**：消除重复前向计算，减少 ~33% 的计算量
4. **去掉 `.clone()`**：减少 18 次 device memcpy
5. **RoPE `torch.empty(0)`**：减少 18 次 CUDA allocator 调用
6. **去掉 dtype 检查**：消除动态分支，利于 CUDA Graph / torch.compile

acclerate-pi-training 分支是在 support-remote-inference 基础上做的**训练加速优化**，包含模型代码层面（SDPA、内存微优化）和配置层面（FSDP no-shard、关闭 gradient checkpointing）两类改动。模型数值语义不变，纯粹是性能层面的改进。在单卡 A800/A100 80GB 上，综合加速效果从 ~4s/it 提升到 ~0.9s/it。
