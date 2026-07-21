# VLA-JEPA LIBERO-10 Integration

This document is the implementation and validation record for the native
VLA-JEPA integration on branch `feat/vla-jepa-libero10`. It is updated with
the code so reviewers can trace the model, data, training, and evaluation
paths without reconstructing them from commits.

## Scope

- Train and evaluate VLA-JEPA on LIBERO-10 only.
- Initialize Qwen3-VL and V-JEPA2 from their upstream base checkpoints.
- Use FluxVLA's FSDP runner, checkpoint format, data pipeline, and
  `LiberoEvalRunner`.
- Defer human-video co-training and released VLA-JEPA checkpoint conversion to
  separate follow-up plans.

## Architecture Decisions

- `VLAJEPA` is a native FluxVLA registry model rather than a wrapper around
  `/root/code/VLA-JEPA`.
- The action policy uses a single-embodiment flow-matching head matching the
  VLA-JEPA training shape: seven action steps with seven action dimensions.
- Qwen3-VL consumes the current agent and wrist images. Its hidden states at
  dedicated world-action tokens condition the world predictor, while hidden
  states at embodied-action tokens condition the action head.
- The frozen `facebook/vjepa2-vitl-fpc64-256` encoder processes two views of
  eight consecutive frames. A trainable action-conditioned predictor forecasts
  the final latent tubelet from the preceding context.
- The optimization target is `action_loss + 0.1 * wm_loss`. Samples whose
  eight-frame window crosses an episode boundary keep their action loss but do
  not contribute world-model loss.
- Rollout inference executes only Qwen3-VL and the action head. World-model
  modules remain checkpoint-compatible but are not called by `predict_action`.

## Key Code Paths

| Responsibility                        | Path / registry interface                                                         |
| ------------------------------------- | --------------------------------------------------------------------------------- |
| Registered VLA and joint loss         | `fluxvla/models/vlas/vla_jepa.py` / `VLAJEPA`                                     |
| Single-embodiment action policy       | `fluxvla/models/heads/vla_jepa_flow_matching_head.py` / `VLAJEPAFlowMatchingHead` |
| Action-conditioned latent predictor   | `fluxvla/models/third_party_models/vjepa2/`                                       |
| Prompt and temporal video preparation | FluxVLA transforms `VLAJEPAPrompter` and `PrepareVLAJEPAVideo`                    |
| LIBERO-10 train/eval recipe           | `configs/vla_jepa/vla_jepa_qwen3vl_2b_libero_10_finetune.py`                      |

## Data and Tensor Flow

1. `ParquetDataset` loads one LIBERO-10 row, actions at offsets `0..6`, and
   video timestamps at offsets `0..7` for both camera views.
2. Temporal preparation emits current Qwen images plus
   `pixel_values_videos` with shape `[2, 8, 3, 256, 256]` per sample.
3. The collator produces batched Qwen inputs, state `[B, 8]`, action
   `[B, 7, 7]`, video `[B, 2, 8, 3, 256, 256]`, and frame-validity masks.
4. Qwen hidden states are gathered using tokenizer IDs, never by positional
   assumptions alone; token counts are validated before reshaping.
5. V-JEPA2 view features are concatenated on the feature axis. The predictor
   consumes context latents and world-action token features, then its L1 target
   is masked by complete-window validity.
6. `predict_action` returns normalized actions with shape `[B, 7, 7]`; the
   LIBERO evaluation transform denormalizes them with
   `libero_10_no_noops` statistics.

## Default Training Recipe

- Hardware target: 8 x 80 GB GPU.
- FSDP full-shard with BF16.
- Per-device batch 4, gradient accumulation 8, global batch 256.
- 30,000 optimizer steps and 5,000 warm-up steps.
- Qwen3-VL learning rate `1e-5`; predictor and action-head learning rate
  `1e-4`; V-JEPA2 encoder frozen.
- Dataset root: `datasets/libero_10_no_noops_lerobotv2.1`.

## Validation Record

### Core component milestone

Implemented the registered VLA, single-embodiment action head, tokenizer
extension, prompt/video transforms, and action-conditioned predictor.

Key technical details:

- The predictor implementation is carried under its own MIT license and keeps
  the source causal attention mask and RoPE behavior.
- The tokenizer adds three unique world-action tokens and one embodied-action
  token. The prompt contains 24 world-token occurrences (eight for each latent
  transition) and 32 embodied-token occurrences.
- `VLAJEPA` resizes Qwen's input embeddings before checkpoint loading and
  validates the token counts before gathering hidden states.
- View features are reshaped as `[B, V, N, D]` before concatenation, avoiding
  cross-sample mixing when `B > 1`.
- World loss compares three predicted latent transitions against latent frames
  1-3 and becomes a differentiable zero when no sample has a complete
  eight-frame window.

Executed:

```bash
pytest -q test/test_models/test_vla_jepa.py
```

Result: `6 passed`. The two warnings are upstream deprecations from timm's
legacy import path and PyTorch's legacy SDPA context manager; neither changes
the tested result.

### Remaining validation sequence

The required runtime sequence is:

1. Focused unit tests for tokens, prompt/video transforms, action head,
   predictor masking, and joint loss.
2. Batch-1 forward/backward validation.
3. Eight-GPU steps 0-10, checkpoint save, resume, and continuation to step 20.
4. One rollout for each of the ten LIBERO-10 tasks.

Environment constraints such as unavailable GPUs, datasets, simulator assets,
or pretrained checkpoints are recorded with the exact probe and command rather
than reported as successful validation.

## Review Boundary

The first review covers the native LIBERO-10 implementation and the validation
that the available environment can execute. It does not cover full 30k-step
convergence, 50-rollout benchmark statistics, human-video co-training, or
published VLA-JEPA checkpoint compatibility.
