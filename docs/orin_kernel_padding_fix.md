# Orin Kernel Padding Fix

## Context

The Orin kernel inference path uses:

- `EagleInferenceBackbone`
- `FlowMatchingInferenceHead`
- CUDA Graph / fused kernels
- `bf16` buffers

For the HUD04 WBT GR00T model, the configured action dimensions are:

```python
action_dim = 64
ori_action_dim = 42
```

The first 42 dimensions are real robot actions. The remaining 22 dimensions are padding.

## Problem

The normal eager `FlowMatchingHead` clears padded action dimensions during inference:

```python
actions[..., self.ori_action_dim:] = 0
```

It does this:

1. after initial random action noise is created;
2. after every denoise step;
3. after RTC prefix inpainting.

Before the fix, the accelerated `FlowMatchingInferenceHead` did not clear these padded dimensions. This meant dimensions `42:64` contained random noise and were fed into the action encoder during denoising.

Even though the final returned action was sliced back to `ori_action_dim`, the padded noise could already have affected the predicted real 42-dimensional action trajectory.

This was a semantic mismatch between the eager and kernel inference paths, not just normal `bf16` numerical error.

## Modified file

```text
/home/limx/sober/FluxVLA/fluxvla/models/heads/flow_matching_inference_head.py
```

## Fix summary

### 1. Clear padded action dimensions after RTC prefix pinning

Added after:

```python
actions = (
    actions * prefill_inv_mask + prefill_actions * prefill_mask)
```

New logic:

```python
if (self.ori_action_dim is not None
        and self.ori_action_dim < self.action_dim):
    actions[..., self.ori_action_dim:] = 0
```

### 2. Clear padded action dimensions after every denoise update

Added after:

```python
actions = actions + dt * pred_velocity
```

New logic:

```python
if (self.ori_action_dim is not None
        and self.ori_action_dim < self.action_dim):
    actions[..., self.ori_action_dim:] = 0
```

### 3. Clear padded action dimensions before copying final actions back to graph buffer

Added after final RTC re-pin:

```python
actions = (actions * prefill_inv_mask + prefill_actions * prefill_mask)
```

New logic:

```python
if (self.ori_action_dim is not None
        and self.ori_action_dim < self.action_dim):
    actions[..., self.ori_action_dim:] = 0
```

### 4. Clear padded dimensions in RTC prefix input

After `prev_actions` is padded/truncated to `action_dim`, added:

```python
if (self.ori_action_dim is not None
        and self.ori_action_dim < self.action_dim):
    prev[..., self.ori_action_dim:] = 0
```

This prevents padded prefix dimensions from entering `prefill_actions`.

### 5. Clear padded dimensions in initial random action noise

After:

```python
init_actions = torch.randn(
    size=(1, self.num_steps, self.action_dim),
    dtype=input_features.dtype,
    device=input_features.device)
```

Added:

```python
if (self.ori_action_dim is not None
        and self.ori_action_dim < self.action_dim):
    init_actions[..., self.ori_action_dim:] = 0
```

## Expected effect

This makes the accelerated Orin action head closer to the normal eager `FlowMatchingHead` RTC inference semantics for padded action dimensions.

It should reduce contamination from random padding noise in dimensions `42:64`, which could otherwise affect the real 42-dimensional action output through the action encoder and denoise loop.

## Validation performed

Syntax check passed:

```bash
cd /home/limx/sober/FluxVLA
python -m py_compile fluxvla/models/heads/flow_matching_inference_head.py
```

## Remaining non-equivalences

This fix does not make Orin kernel inference bit-exact with normal Eagle inference. Remaining expected differences include:

- `EagleInferenceBackbone` vs `EagleBackbone`;
- fused kernel / CUDA Graph execution;
- `bf16` rounding and fused operation order;
- fixed inference `prefix_len` vs training-time RTC delay sampling distribution;
- async scheduling, `execute_horizon`, `publish_rate`, and `target_hz` runtime behavior.

