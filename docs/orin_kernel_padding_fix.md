# Orin Kernel Padding 修复说明

## 背景

Orin kernel 推理路径使用：

- `EagleInferenceBackbone`
- `FlowMatchingInferenceHead`
- CUDA Graph / fused kernels
- `bf16` buffers

对于 HUD04 WBT GR00T 模型，配置中的 action 维度为：

```python
action_dim = 64
ori_action_dim = 42
```

前 42 维是真实机器人动作，剩余 22 维是 padding。

## 问题

普通 eager 版本的 `FlowMatchingHead` 在推理时会清零 padding action 维度：

```python
actions[..., self.ori_action_dim:] = 0
```

它会在以下位置执行清零：

1. 初始随机 action noise 创建之后；
2. 每个 denoise step 之后；
3. RTC prefix inpainting 之后。

修复前，加速版 `FlowMatchingInferenceHead` 没有清零这些 padding 维度。这意味着 `42:64` 维包含随机噪声，并且这些噪声会在 denoising 过程中进入 action encoder。

即使最终返回的 action 会被切回 `ori_action_dim`，padding 噪声也已经可能影响了预测出来的真实 42 维 action 轨迹。

这不是普通的 `bf16` 数值误差，而是 eager 推理路径和 kernel 推理路径之间的语义不一致。

## 修改文件

```text
/home/limx/sober/FluxVLA/fluxvla/models/heads/flow_matching_inference_head.py
```

## 修复摘要

### 1. RTC prefix pinning 后清零 padding action 维度

在以下代码之后添加：

```python
actions = (
    actions * prefill_inv_mask + prefill_actions * prefill_mask)
```

新增逻辑：

```python
if (self.ori_action_dim is not None
        and self.ori_action_dim < self.action_dim):
    actions[..., self.ori_action_dim:] = 0
```

### 2. 每次 denoise 更新后清零 padding action 维度

在以下代码之后添加：

```python
actions = actions + dt * pred_velocity
```

新增逻辑：

```python
if (self.ori_action_dim is not None
        and self.ori_action_dim < self.action_dim):
    actions[..., self.ori_action_dim:] = 0
```

### 3. 最终 action 拷回 graph buffer 前清零 padding action 维度

在最终 RTC re-pin 之后添加：

```python
actions = (actions * prefill_inv_mask + prefill_actions * prefill_mask)
```

新增逻辑：

```python
if (self.ori_action_dim is not None
        and self.ori_action_dim < self.action_dim):
    actions[..., self.ori_action_dim:] = 0
```

### 4. 清零 RTC prefix 输入中的 padding 维度

在 `prev_actions` 被 pad/truncate 到 `action_dim` 之后添加：

```python
if (self.ori_action_dim is not None
        and self.ori_action_dim < self.action_dim):
    prev[..., self.ori_action_dim:] = 0
```

这样可以避免 padding prefix 维度进入 `prefill_actions`。

### 5. 清零初始随机 action noise 中的 padding 维度

在以下代码之后：

```python
init_actions = torch.randn(
    size=(1, self.num_steps, self.action_dim),
    dtype=input_features.dtype,
    device=input_features.device)
```

添加：

```python
if (self.ori_action_dim is not None
        and self.ori_action_dim < self.action_dim):
    init_actions[..., self.ori_action_dim:] = 0
```

## 预期效果

这会让加速版 Orin action head 在 padding action 维度上的语义更接近普通 eager `FlowMatchingHead` 的 RTC 推理行为。

它应该能减少 `42:64` 维随机 padding 噪声带来的污染。否则，这些噪声可能通过 action encoder 和 denoise loop 影响真实的 42 维 action 输出。

## 已执行验证

语法检查已通过：

```bash
cd /home/limx/sober/FluxVLA
python -m py_compile fluxvla/models/heads/flow_matching_inference_head.py
```

## 仍然存在的不等价点

这个修复不会让 Orin kernel 推理和普通 Eagle 推理达到 bit-exact。仍然存在的预期差异包括：

- `EagleInferenceBackbone` vs `EagleBackbone`；
- fused kernel / CUDA Graph 执行；
- `bf16` rounding 和 fused operation order；
- 固定 inference `prefix_len` vs 训练时 RTC delay 采样分布；
- async scheduling、`execute_horizon`、`publish_rate` 和 `target_hz` 的运行时行为。

