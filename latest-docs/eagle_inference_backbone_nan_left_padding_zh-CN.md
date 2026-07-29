# EagleInferenceBackbone 输出 NaN 问题记录

> 日期：2026-06-23\
> 场景：Jetson Orin / FluxVLA Docker / GR00T Eagle UR3 真机推理\
> 现象：`actions` 全部为 `nan`

## 1. 现象

UR3 真机推理可以正常启动，ROS topic 也能读到，但模型输出动作全是 `nan`：

```text
predict time : 0.22988605499267578
actions : [[nan nan nan nan nan nan nan]
 [nan nan nan nan nan nan nan]
 ...]
```

这不是单纯的 ROS 通信问题，因为 `/joint_states`、`/gripper/position`、相机 topic 都能读到有效数据。

## 2. 已排除的问题

本次排查中确认以下内容都是 finite：

- 原始 ROS 观测：`qpos`、前相机图像、腕相机图像。
- `dataset_statistics.json` 中的 `private.proprio` 和 `private.action`。
- 当前 safetensors checkpoint 中的 action head 权重。
- 当前 safetensors checkpoint 中的 VLM / backbone 权重。
- dataset transform 后的模型输入：`images`、`lang_tokens`、`lang_masks`、`states`、`embodiment_ids`。

诊断输出示例：

```text
model_inputs: finite
images: finite=301056/301056
states: finite=64/64
embodiment_ids: finite=1/1
```

## 3. NaN 首次出现的位置

打开诊断开关：

```bash
export FLUXVLA_DEBUG_NAN=1
```

使用 `EagleInferenceBackbone` 时，NaN 首次出现在 VLM backbone 的 hidden state：

```text
model_inputs: finite
llava_vla.last_hidden_state: finite=0/1228800
flow_head.input_features: finite=0/1228800
raw_action: finite=0/224
postprocessed_actions: finite=0/224
```

这说明动作反归一化不是根因。`DenormalizePrivateAction` 只是把上游模型输出中的 NaN 继续传播出来。

## 4. 对比验证

将 UR3 推理配置中的 VLM backbone 从 `EagleInferenceBackbone` 切回普通 `EagleBackbone` 后，同一帧输入恢复正常：

```text
llava_vla.last_hidden_state: finite=1228800/1228800
flow_head.input_features: finite=1228800/1228800
flow_head.graph_actions: finite=1024/1024
raw_action: finite=224/224
postprocessed_actions: finite=224/224
```

因此根因锁定在 `EagleInferenceBackbone` 使用的优化版 Eagle VLM 前向，而不是 ROS、stats 或 checkpoint。

## 5. 问题本质

`EagleInferenceBackbone` 调用的是优化版：

```text
EagleInferenceBackbone
  -> Eagle2_5_VLInferenceForConditionalGeneration
  -> extract_language_feature()
  -> language CUDA Graph
  -> scaled_dot_product_attention()
```

问题本质是：优化版 Qwen language attention 没有正确处理左 padding 下的 fully-masked attention rows。

## 6. 为什么左 padding 会触发 fully-masked rows

推理输入会被 pad 到固定长度，例如 `max_len=600`。当前 prompt/tokenizer 默认是左 padding，所以序列长这样：

```text
位置:   0   1   2   3   4   5   6   7 ...
token: PAD PAD PAD PAD IMG IMG TEXT TEXT ...
mask:   0   0   0   0   1   1   1    1 ...
```

其中：

```text
attention_mask = 1  表示真实 token，可以被看见
attention_mask = 0  表示 padding token，应该被屏蔽
```

Qwen/Llama 这类 decoder-only language model 使用 causal attention。第 `i` 个 token 只能看自己和左边 token：

```text
第 i 个 query 只能 attend 到 key 0 ... i
```

现在看左 padding 区域里的 query：

```text
第 0 行只能看 [0]，但 0 是 PAD，被 padding mask 屏蔽。
第 1 行只能看 [0, 1]，但 0 和 1 都是 PAD。
第 2 行只能看 [0, 1, 2]，但它们也都是 PAD。
```

所以这些行没有任何合法 key 可以 attend，形成 fully-masked row。

叠加 causal mask 和 padding mask 后，某些 attention 行会变成：

```text
[-inf, -inf, -inf, -inf, ...]
```

## 7. 为什么全 `-inf` 会变成 NaN

attention softmax 的形式是：

```text
softmax(x_i) = exp(x_i) / sum(exp(x_j))
```

如果整行都是 `-inf`：

```text
exp(-inf) = 0
sum(exp(x_j)) = 0
```

于是得到：

```text
0 / 0 = nan
```

NaN 一旦进入 hidden state，后续矩阵乘、残差、norm 会继续传播，最终污染整块 VLM hidden state，并导致 `raw_action` 和后处理后的 `actions` 全部为 NaN。

## 8. 相关代码位置

优化版 attention mask 构造在：

```text
fluxvla/models/third_party_models/eagle2_hg_model/modeling_eagle2_5_vl_inference.py
```

关键逻辑：

```python
causal_mask = torch.tril(
    torch.ones(seq_len, seq_len, device=attention_mask.device, dtype=torch.bool))
padding_mask = attention_mask[:, None, None, :].bool()
combined = causal_mask[None, None, :, :] & padding_mask
self.buffers['attention_mask'].copy_(
    torch.where(combined, 0.0, float('-inf')).to(torch.bfloat16))
```

这里没有处理 fully-masked rows，所以会把整行 `-inf` 送入 `scaled_dot_product_attention()`。

## 9. 为什么 `ATTN_IMPLEMENTATION=eager` 没有解决

`ATTN_IMPLEMENTATION=eager` 能影响 HuggingFace 标准 attention 路径，但 `EagleInferenceBackbone` 调用的是自定义优化版 inference 文件。该文件内部已经绕开部分 HuggingFace attention 实现，走 Triton / CUDA Graph / `scaled_dot_product_attention()` 的固定路径。

因此即使设置：

```bash
ATTN_IMPLEMENTATION=eager
TRANSFORMERS_ATTN_IMPLEMENTATION=eager
```

也不一定能绕过这个优化版 language CUDA Graph 中的 mask 问题。

## 10. 已验证修复

已在优化版 `extract_language_feature()` 中对 fully-masked rows 做安全处理，避免整行 `-inf` 进入 attention softmax：

```python
fully_masked_rows = ~combined.any(dim=-1, keepdim=True)
combined = combined | fully_masked_rows
```

修复后可继续使用 `EagleInferenceBackbone`：

```python
inference_model = dict(
    type='LlavaVLA',
    pretrained_name_or_path='./checkpoints/GR00T-N1.5-3B',
    vlm_backbone=dict(
        type='EagleInferenceBackbone',
        vlm_path='fluxvla/models/third_party_models/eagle2_hg_model'),
    # ...
)
```

修复后验证结果：

```text
llava_vla.last_hidden_state: finite=1228800/1228800
raw_action: finite=224/224
postprocessed_actions: finite=224/224
```

如果运行的是未打补丁的旧代码，临时规避方案仍是回退到普通 `EagleBackbone`。

## 11. 速度影响

修复后优化版 `EagleInferenceBackbone` 首次预测仍可能包含 CUDA Graph / buffer 初始化成本，后续稳定预测约为：

```text
predict time : 0.195s
```

如果回退到普通 `EagleBackbone`，稳定预测约为 `0.27-0.37s`。这个差异主要来自是否使用 `EagleInferenceBackbone` 的端到端 Triton / CUDA Graph 优化前向，而不是单纯是否使用 `flash_attention_2`。

## 12. 长期修复方向

当前已采用方向 2 完成修复。后续仍可考虑以下增强：

1. 推理 tokenizer/prompt 改成右 padding。

   真实 token 在左边，padding 在右边。真实 query 前面总能看到真实 token，不容易出现左侧 padding query 的 fully-masked rows。

2. 保留左 padding，但在 `extract_language_feature()` 中处理 fully-masked rows。已完成。

   在把 mask 传给 attention 前，检测整行不可见的 query，对这些行做安全 unmask 或使用类似 Transformers `_unmask_unattended` 的逻辑，避免整行 `-inf` 进入 softmax。

3. 增加优化版 backbone 的 NaN guard。

   在 `EagleInferenceBackbone` 或 `Eagle2_5_VLInferenceForConditionalGeneration` 中输出 hidden state 后做 finite 检查，发现全 NaN 时明确报错或自动回退到普通 backbone，避免机械臂路径继续消费无效动作。

## 13. 推荐结论

当前真机测试可以使用已修复的优化版路径：

```text
UR3 真机推理：使用 EagleInferenceBackbone
旧代码临时规避：回退到 EagleBackbone
```

真机动作下发前仍建议保留一次 `FLUXVLA_DEBUG_NAN=1` 验证，确认 `last_hidden_state`、`raw_action` 和 `postprocessed_actions` 均为 finite。
