# DreamZero FSDP 显存分析

这份文档说明：为什么 DreamZero 即使在 8 卡训练时也可能只能使用
`per_device_batch_size=1`，backward 阶段会出现哪些显存开销，以及
DreamZero 为什么比普通 VLA 动作预测训练更重。

## 为什么 8 卡仍然需要单卡 Batch Size 1

FSDP `full-shard` 会降低每张 GPU 上常驻的模型状态显存。它会对这些内容
做分片：

- 模型参数
- 梯度
- optimizer state

在 8 卡训练时，每个 rank 大致只保存每个 FSDP unit 的八分之一参数 shard
和 optimizer state shard。对于大模型来说，这能明显降低常驻显存。

但是，增加 GPU 数量并不会自动降低每个 rank 自己的 activation 显存。
activation 仍然由当前 rank 的本地 batch 产生。`per_device_batch_size`
从 1 增加到 4 时，每张卡都要处理 4 个样本，所以 activation 和很多临时
张量会近似线性增长。

也就是说：

```text
8 卡提升的是 global batch 能力。
它不会让每张卡上的 batch 4 变得和 batch 1 一样便宜。
```

对 DreamZero 来说，每个样本本身就很重，所以更稳的设置是：

```bash
--nproc_per_node 8
train_dataloader.per_device_batch_size=1
train_dataloader.batch_size=8
```

而下面这种设置风险很高：

```bash
--nproc_per_node 8
train_dataloader.per_device_batch_size=4
train_dataloader.batch_size=32
```

这会让每张 GPU 同时处理 4 个 DreamZero 样本。即使模型参数已经被 FSDP
分片，单卡 activation 峰值也很容易 OOM。

## Backward 阶段有哪些显存

backward 时，GPU 显存里不只有常驻参数 shard。下面这些内容可能同时存在：

- 当前 FSDP wrapped module all-gather 后的完整参数。
- 本地参数 shard 和 FSDP 元数据。
- forward 保存下来的 activation，或者 gradient checkpointing 重新计算出的
  activation。
- attention 中间张量，例如 Q/K/V、attention output、MLP 中间激活。
- prediction、target、mask、loss 相关中间张量。
- 当前 FSDP unit 的梯度。
- reduce-scatter 通信 buffer。
- optimizer state，例如 AdamW 的 `exp_avg` 和 `exp_avg_sq`。
- optimizer step 期间的临时张量。

之前观察到的 OOM 分别发生在两个阶段：

```text
optimizer.step()
```

这通常指向 AdamW optimizer 显存，尤其是 foreach 实现里的临时张量。

```text
loss.backward()
```

这通常指向 backward 峰值显存，包括 activation 重算、FSDP all-gather
完整参数、梯度和通信 buffer。

## 为什么 DreamZero 比普通 VLA 训练更重

DreamZero 训练不是普通的图像/语言到动作预测。它更接近 joint
video-action flow matching。

高层流程可以理解为：

```text
video window + language + state + action
-> WanBackbone 编码 text、image、video
-> 对 video latents 加噪
-> 对 actions 加噪
-> Wan DiT 同时预测 video noise 和 action noise
-> dynamics loss + action loss
```

相关代码路径大致是：

```python
vlm_outputs = self.vlm_backbone(
    video=video,
    input_ids=lang_tokens,
    attention_mask=lang_masks,
)

return self.vla_head(
    prompt_embs=vlm_outputs["prompt_embs"],
    latents=vlm_outputs["latents"],
    clip_feas=vlm_outputs["clip_feas"],
    ys=vlm_outputs["image_cond"],
    states=states,
    actions=actions,
    action_masks=action_masks,
    embodiment_ids=embodiment_ids,
)
```

在 `DreamZeroHead` 里，训练阶段会额外创建：

- video 随机噪声
- noisy video latents
- video training targets
- action 随机噪声
- noisy actions
- action training targets

然后 DiT 会同时接收 noisy latent 和 clean latent 信息：

```python
video_noise_pred, action_noise_pred = self.model(
    noisy_latents,
    timestep=timestep,
    clip_feature=clip_feas,
    y=image_cond,
    context=prompt_embs,
    seq_len=seq_len,
    state=states,
    embodiment_id=embodiment_ids,
    action=noisy_actions,
    timestep_action=timestep_action,
    clean_x=latents,
)
```

这比只从单帧 observation 和语言指令预测 action 的普通 VLA head 重很多。

## Token 和 Latent 开销

当前 DreamZero LIBERO 配置里：

```text
frame_window_size = 9
frame_seqlen = 128
```

latent video frame 数量是：

```text
1 + (9 - 1) // 4 = 3
```

所以 video latent token 数量大约是：

```text
3 * 128 = 384 tokens
```

训练时还会把 `clean_x=latents` 传入 DiT。DiT 内部使用类似 teacher-forcing
的结构，把 clean video tokens 和 noisy video tokens 结合起来，再叠加
action tokens、state tokens、CLIP conditioning 和 text cross-attention。

因此，每个样本在 forward 和 backward 中都很重。

## 实际建议

DreamZero full fine-tuning 使用 Wan 14B DiT 时，更合理的扩展方式是增加
GPU 数量来扩大 global batch，同时保持每张卡的 batch 很小：

```text
推荐：8 GPUs * per_device_batch_size 1 = global batch 8
高风险：8 GPUs * per_device_batch_size 4 = global batch 32
```

如果需要更大的 effective batch，正确方式应该是 gradient accumulation，
而不是直接提高 `per_device_batch_size`。但当前 runner 明确禁止了梯度累积：

```python
assert self.grad_accumulation_steps == 1
```

所以如果想在不增加单卡显存的情况下扩大 effective batch，需要先在训练
runner 里实现 gradient accumulation。
