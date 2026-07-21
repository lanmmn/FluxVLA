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

## Environment Preparation

The current Hugging Face client uses `hf`; the deprecated
`huggingface-cli download` command exits without downloading.

```bash
hf download Qwen/Qwen3-VL-2B-Instruct \
  --include '*.json' --include '*.txt' --include '*.safetensors'
hf download facebook/vjepa2-vitl-fpc64-256 \
  --include '*.json' --include '*.safetensors'
hf download limxdynamics/FluxVLAData \
  --repo-type dataset \
  --include 'libero_10_no_noops_lerobotv2.1/*' \
  --local-dir datasets
```

The V-JEPA2 include rule intentionally excludes the unused 4.8 GiB original
PyTorch checkpoint because Transformers loads `model.safetensors`.

LIBERO simulation uses the commits pinned by `requirements-sim.txt`. The
validated environment has robosuite 1.5.1 from `e293cc3` and an editable
LIBERO checkout at `058fda1` so scene XML and other assets remain available.
The preferred installation command is:

```bash
bash scripts/install_env.sh sim-only
```

On this host, an older editable LIBERO checkout and stale `~/.libero` paths
were already present. The isolated runtime used for validation is:

```text
LIBERO_CONFIG_PATH=/limx_embop/tos/limx_mani_data/fluxvla_work_dirs/libero_runtime_config
LIBERO source=/limx_embop/tos/limx_mani_data/fluxvla_work_dirs/libero_runtime_src/libero
robosuite commit=e293cc32ff3c48957a4ebcad09952432b0dc9049
LIBERO commit=058fda1ddebe92918af091cb6816759ca6d003f0
```

## Reproduction Commands

Formal 8-GPU training:

```bash
WANDB_MODE=disabled torchrun --standalone --nnodes 1 --nproc-per-node 8 \
  scripts/train.py \
  --config configs/vla_jepa/vla_jepa_qwen3vl_2b_libero_10_finetune.py \
  --work-dir work_dirs/vla_jepa_libero10
```

Ten-step smoke run and checkpoint creation:

```bash
WANDB_MODE=disabled torchrun --standalone --nnodes 1 --nproc-per-node 8 \
  scripts/train.py \
  --config configs/vla_jepa/vla_jepa_qwen3vl_2b_libero_10_finetune.py \
  --work-dir \
  /limx_embop/tos/limx_mani_data/fluxvla_work_dirs/vla_jepa_libero10_smoke \
  --cfg-options runner.max_steps=10 runner.save_iter_interval=10
```

Resume the same run through step 20:

```bash
WANDB_MODE=disabled torchrun --standalone --nnodes 1 --nproc-per-node 8 \
  scripts/train.py \
  --config configs/vla_jepa/vla_jepa_qwen3vl_2b_libero_10_finetune.py \
  --work-dir \
  /limx_embop/tos/limx_mani_data/fluxvla_work_dirs/vla_jepa_libero10_smoke \
  --resume-from \
  /limx_embop/tos/limx_mani_data/fluxvla_work_dirs/vla_jepa_libero10_smoke/checkpoints/latest-checkpoint.pt \
  --cfg-options runner.max_steps=20 runner.save_iter_interval=10
```

One rollout for every LIBERO-10 task:

```bash
LIBERO_CONFIG_PATH=/limx_embop/tos/limx_mani_data/fluxvla_work_dirs/libero_runtime_config \
  WANDB_MODE=disabled torchrun --standalone --nnodes 1 --nproc-per-node 8 \
  scripts/eval.py \
  --config configs/vla_jepa/vla_jepa_qwen3vl_2b_libero_10_finetune.py \
  --ckpt-path \
  /limx_embop/tos/limx_mani_data/fluxvla_work_dirs/vla_jepa_libero10_smoke/checkpoints/latest-checkpoint.safetensors \
  --cfg-options eval.num_trials_per_task=1
```

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

Result after adding config and incomplete-window coverage: `8 passed`. The two
warnings are upstream deprecations from timm's
legacy import path and PyTorch's legacy SDPA context manager; neither changes
the tested result.

Built `DistributedRepeatingDataset` from the real LIBERO-10 mount, fetched one
sample, and ran the configured `DictCollator`. The checked batch shapes were:

```text
states:                 [1, 8]
actions:                [1, 7, 7]
images:                 [1, 392, 1536]
img_masks:              [1, 2]
pixel_values_videos:    [1, 2, 8, 3, 256, 256]
frame_masks:            [1, 8]
lang_tokens:            [1, 128]
image_grid_thw:         [1, 2, 3]
```

This check exposed and fixed a mask-alignment bug in the first implementation:
after reducing 16 view-major temporal frames to two current Qwen images, the
transform must also reduce `img_masks` from 16 entries to two.

The evaluation-only transform chain was also executed against a synthetic
LIBERO observation using the tokenizer and statistics saved by the 10-step
checkpoint. It produced finite state `[8]`, Qwen pixels `[512, 1536]`, image
grid `[2, 3]`, and language arrays `[128]`. The restored token IDs occurred
`8/8/8/32` times for world-action 0/1/2 and embodied-action respectively, so
the rollout input contract matches training without invoking the world model.

The full inference model was then rebuilt on CPU and loaded from the actual
10-step safetensors checkpoint: all 1,617 tensors matched with zero missing and
zero unexpected keys. A real Qwen-plus-action-head `predict_action` call on the
synthetic observation returned finite FP32 actions with shape `[1, 7, 7]`.
Both V-JEPA modules were replaced with call-failing sentinels for this check;
neither sentinel fired, directly verifying that rollout inference skips the
world model. The CPU-only harness explicitly cast the action head and state to
BF16 to mirror `LiberoEvalRunner.run_setup()`, which casts the complete model
before normal GPU inference.

Environment snapshot during implementation:

- The mounted LIBERO-10 dataset is readable and contains 388 parquet files,
  776 videos, and the required v2.1 metadata.
- Eight A100 80 GB GPUs are present. They became free long enough to complete
  the batch-1 and 8-GPU smoke milestones; later an unrelated host-namespace
  job occupied the GPUs again. No unrelated process was stopped or modified.

### Batch-1 forward/backward milestone

Executed one complete forward, backward, optimizer, scheduler, and checkpoint
step on one GPU with per-device batch one. The finite joint loss was `3.0081`.

```bash
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled torchrun --standalone \
  --nnodes 1 --nproc-per-node 1 scripts/train.py \
  --config configs/vla_jepa/vla_jepa_qwen3vl_2b_libero_10_finetune.py \
  --work-dir work_dirs/vla_jepa_libero10_batch1 \
  --cfg-options train_dataloader.per_device_batch_size=1 \
  train_dataloader.per_device_num_workers=0 \
  runner.grad_accumulation_steps=1 runner.max_steps=1 \
  runner.save_iter_interval=1
```

The training step and `.pt` save completed, proving that the optimizer state is
serializable. Saving the second, model-only safetensors copy then exhausted the
root filesystem. The valid `.pt` was checked with `torch.load(..., mmap=True)`:
it contains `global_step=1`, 1,617 model tensors, optimizer state, and scheduler
state. The run was moved to:

```text
/limx_embop/tos/limx_mani_data/fluxvla_work_dirs/vla_jepa_libero10_batch1
```

All subsequent smoke checkpoints use the NAS path because a step checkpoint
requires approximately 30 GiB for `.pt` plus 12 GiB for `.safetensors`.

### Eight-GPU 10-step milestone

The default batch topology ran as configured: per-device batch 4, eight GPUs,
and gradient accumulation 8, for global batch 256. FSDP full-shard BF16 used a
maximum of approximately 61.6 GiB on the most-loaded GPU. Steps 1-10 completed
without OOM or non-finite losses:

```text
joint loss: 2.4006, 2.1272, 2.5460, 2.1451, 2.4354,
            2.2014, 2.3616, 2.0054, 2.0032, 2.1987
wm loss:    1.9770 -> 1.8245
```

The interval mean joint loss was `2.2425`. Both FluxVLA checkpoint forms were
written successfully:

```text
/limx_embop/tos/limx_mani_data/fluxvla_work_dirs/vla_jepa_libero10_smoke/checkpoints/step-000010-epoch-00-loss=2.2425.pt
/limx_embop/tos/limx_mani_data/fluxvla_work_dirs/vla_jepa_libero10_smoke/checkpoints/step-000010-epoch-00-loss=2.2425.safetensors
```

The `.pt` is 31,874,199,790 bytes and includes optimizer/scheduler state; the
model-only safetensors file is 12,321,890,692 bytes. The only process-exit
warning was PyTorch's existing `destroy_process_group()` warning.

The ten-step smoke override necessarily compresses the formal cosine schedule
to ten steps, so its learning rate reaches zero at step 10. Resuming this exact
checkpoint validates complete state restoration but is not a meaningful
learning-continuation experiment. Formal training retains the intended
30,000-step schedule and 5,000-step warm-up.

### Eight-GPU resume milestone

The complete 10-step `.pt` was resumed after the external GPU job ended. FSDP
reported successful restoration of model, optimizer, and scheduler state and
started at `global_step=10`, epoch 0. Steps 11-20 completed without OOM or
non-finite values:

```text
joint loss: 2.3147, 2.1565, 2.0635, 2.0970, 2.4507,
            2.0879, 2.1099, 1.9603, 1.9800, 2.1553
wm loss:    1.8300 -> 1.7980
```

The interval mean joint loss was `2.1376`. Memory-mapped inspection of the new
checkpoint confirms `global_step=20`, epoch 0, 1,617 model tensors, 1,029
optimizer states, and scheduler `last_epoch=20`:

```text
/limx_embop/tos/limx_mani_data/fluxvla_work_dirs/vla_jepa_libero10_smoke/checkpoints/step-000020-epoch-00-loss=2.1376.pt
/limx_embop/tos/limx_mani_data/fluxvla_work_dirs/vla_jepa_libero10_smoke/checkpoints/step-000020-epoch-00-loss=2.1376.safetensors
```

Their sizes are 31,874,199,790 and 12,321,890,692 bytes respectively, and both
`latest-checkpoint` symlinks resolve to step 20. As expected from the shortened
smoke schedule, step 10 restored at zero LR; the newly constructed 20-step
cosine schedule then continued from step 11. This does not affect the formal
30,000-step schedule.

### LIBERO-10 rollout milestone

The first simulator launch correctly rejected the pre-existing
`robosuite==1.4.1`. Installing the commits pinned by `requirements-sim.txt`
revealed two additional host-specific issues: the old editable LIBERO checkout
lacked robosuite 1.5 robot fields, and a non-editable wheel omitted scene XML
assets. The final fix uses robosuite 1.5.1 plus an isolated editable LIBERO
checkout and `LIBERO_CONFIG_PATH`. A task-0 environment probe successfully ran
reset, fixed initial-state loading, and one dummy step before the full launch.

The 8-rank evaluation completed exactly one rollout for each task: 10/10
episodes completed, 0/10 succeeded, per-task time was 25.91-29.72 seconds, and
the summed task time was 278.09 seconds. This run is a functional smoke test;
the zero success rate after only 20 optimizer steps is not an acceptance
failure. Summary JSON, per-task JSON, eight rank logs, and ten rollout videos
are under:

```text
/limx_embop/tos/limx_mani_data/fluxvla_work_dirs/vla_jepa_libero10_smoke/eval_runs/step-000020-epoch-00-loss=2.1376/EVAL-libero_10-vla_jepa-2026_07_21-13_11_55
```

All first-round runtime acceptance items are now complete. Remaining warnings
are upstream deprecations, optional robosuite robot packages not used by Panda
OSC control, missing optional OpenGL acceleration, and the existing process
group shutdown warning.

## Review Boundary

The first review covers the native LIBERO-10 implementation and all four smoke
validation milestones. It does not cover full 30k-step convergence, 50-rollout
benchmark statistics, human-video co-training, or published VLA-JEPA
checkpoint compatibility.
