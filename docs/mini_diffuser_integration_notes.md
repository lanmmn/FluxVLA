# Mini-Diffuser 集成 Pi0.5：方案演进记录

## 方案一：KV-Cache Split Path（失败）

### 思路

利用 Pi0.5 的 attention mask 单向性（prefix 不 attend to suffix），将训练的 forward 拆成两步：

```
Step 1: prefix-only pass (batch=B)
  embed_prefix(images) → prefix_embs
  forward_model([prefix_embs, None], use_cache=True) → KV cache

Step 2: suffix-only pass (batch=B×M)
  repeat KV cache M 次 → (B×M)
  embed_suffix(states, noisy_actions, time) → suffix_embs
  forward_model([None, suffix_embs], past_key_values=repeated_cache)
```

这和推理时的做法完全一致——推理时就是先跑 prefix 缓存 KV，再多次跑 suffix 做去噪。数学上严格等价，因为 prefix 的输出不受 suffix 影响。

**预期优势**：不仅 SigLIP 只跑 B 次，连 `llm_backbone` 的 18 层 transformer 也只跑 B 次。只有轻量的 `llm_expert`（hidden=1024，比 backbone 的 2048 小一半）跑 B×M 次。

### 失败原因：Gradient Checkpointing 与 KV Cache 不兼容

**问题根源**在 `ConditionGemmaModel`（基于 HuggingFace Gemma）的两个设计：

1. **模型级冲突**（第 907-911 行）：
   ```python
   if self.gradient_checkpointing and self.training and use_cache:
       use_cache = False  # 强制关闭！
   ```
   训练时开启了 gradient checkpointing，`use_cache=True` 会被强制改为 `False`，导致 prefix pass 根本不生成 KV cache。

2. **层级 checkpoint recompute 冲突**：`GemmaDecoderLayer` 继承自 `GradientCheckpointingLayer`，每个 decoder layer 的 forward 都会被 checkpoint。在 backward 时 recompute forward，`create_causal_mask` 会重新根据输入尺寸生成 mask，但此时 KV cache 的状态已经被修改过（各层的 cat 操作是有状态的），导致 mask 尺寸和实际 attention 的 K 维度不匹配：
   ```
   RuntimeError: The size of tensor a (1384) must match the size of tensor b (692)
   at non-singleton dimension 3
   ```
   其中 1384 = 692 × 2，正好是 M=2 的倍数关系。

**尝试过的修复**：临时关闭 `llm_backbone` 和 `llm_expert` 的 `gradient_checkpointing` 属性。但 `GradientCheckpointingLayer` 是在每个 layer 级别自动 checkpoint 的，不受模型级属性控制，无法简单绕过。

### 本质矛盾

推理时可以用 KV cache split 是因为：
- **推理时不需要 gradient checkpointing**（不做 backward）
- **推理时 `use_cache=True` 正常工作**

训练时 gradient checkpointing 是必须的（否则 80GB 显存装不下），而它和 KV cache 机制在 HuggingFace transformers 的实现中天然互斥。要修复需要改 `condition_gemma.py` 的底层实现，侵入性太大。

---

## 方案二：Joint Path + Prefix Embedding Repeat（成功）

### 思路

不拆分 prefix/suffix 的 forward 路径，而是：

```
Step 1: embed_prefix (batch=B)
  SigLIP(images) + LLM.embed_tokens(lang) → prefix_embs (B, L, D)

Step 2: repeat prefix embeddings (B → B×M)
  prefix_embs.repeat_interleave(M, dim=0) → (B×M, L, D)

Step 3: embed_suffix (batch=B×M)
  独立采样 B×M 个 noise/time
  embed_suffix(states, x_t, time) → suffix_embs (B×M, ...)

Step 4: joint _forward_transformer_layers (batch=B×M)
  和原版完全相同的路径，只是 batch 从 B 变成了 B×M
```

### 为什么能跑通

- 走的是 `_forward_transformer_layers` 的 **joint 路径**，不涉及 KV cache
- Gradient checkpointing 正常工作（每个 layer 的 checkpoint recompute 不依赖外部状态）
- FSDP 正常工作（标准的 module forward call）
- 不需要改任何底层代码

### 和方案一的对比

| | 方案一 (KV-Cache Split) | 方案二 (Embedding Repeat) |
|---|---|---|
| SigLIP | 只跑 B 次 | 只跑 B 次 |
| llm_backbone transformer 层 | 只跑 B 次 | 跑 B×M 次（但输入是重复的 embedding） |
| llm_expert transformer 层 | 跑 B×M 次 | 跑 B×M 次 |
| Gradient Checkpointing | 不兼容 | 完全兼容 |
| 代码改动 | 需要改底层 | 只改 Pi0.5 子类 |
| 计算节省 | 大（backbone 也省了） | 中（只省 SigLIP） |
| 显存节省 | 大 | 小（joint attention 序列更长） |

### 代价

方案二的 `_forward_transformer_layers` 中，`llm_backbone` 也要处理 B×M 份 prefix tokens。虽然这些 prefix tokens 是 embedding 的复制（不重跑 SigLIP），但 backbone 的 18 层 transformer attention + MLP 仍然要在 B×M 个 prefix 上计算。因此：

- **显存更高**：B=16, M=2 时 joint path 的有效 batch=32，OOM 了（80GB H100）
- **最终可用配置**：B=8, M=2（有效 denoising batch=16/卡）

---

## 后续优化方向

如果要实现方案一的完整效率提升，需要：

1. **修改 `ConditionGemmaModel`**：在训练时支持 KV cache + gradient checkpointing 共存。具体做法是让 `create_causal_mask` 正确处理 `past_key_values` 的 seq_length，以及确保 `GradientCheckpointingLayer` 的 recompute 不依赖 KV cache 的可变状态。

2. **或者自定义 suffix-only forward**：不走 `llm_expert.forward()`，而是手动实现一个类似 `_forward_transformer_layers` 但只处理 suffix 的函数，内部手动拼接 cached prefix KV。这样完全绕过 `create_causal_mask` 和 `GradientCheckpointingLayer`。
