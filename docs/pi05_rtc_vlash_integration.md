# PI0.5 RTC VLASH Integration

## Goal

Apply VLASH-style shared-observation multi-branch RTC training to FluxVLA PI0.5 training.

Target training entry:

`configs/pi05/pi05_paligemma_ur3_rtc_vlash_full_finetune.py`

## What Changed

### 1. RTC utilities

File: `fluxvla/engines/utils/rtc_training.py`

- Added `get_training_delay_weights(...)`
- Purpose: when all RTC delay branches are trained together, aggregate branch losses with either:
  - the original sampled RTC delay distribution
  - uniform weighting

### 2. Shared-observation attention helper

File: `fluxvla/engines/utils/model_utils.py`

- Added `build_shared_obs_att_2d_masks_and_position_ids(...)`
- Purpose:
  - keep prefix tokens shared
  - tile suffix tokens across delay branches
  - block cross-offset suffix-to-suffix attention
  - generate consistent position ids

### 3. PI0 / PI0.5 training forward path

File: `fluxvla/models/vlas/pi0_flowmatching.py`

Because `PI05FlowMatching` inherits from `PI0FlowMatching`, the PI0.5 training path is updated through the shared base class.

Added:

- `_rtc_shared_observation_enabled()`
- `_expand_shared_observation_branches(...)`
- `_concat_shared_suffix_branches(...)`
- `_prepare_shared_rtc_training_inputs(...)`
- `_prepare_shared_rtc_attention_inputs(...)`

Updated training `forward(...)`:

- Old RTC path:
  - sample one delay per sample
  - compute one suffix branch per sample

- New shared-observation RTC path:
  - expand one observation into all delays `[0, max_delay)`
  - compute prefix once
  - concatenate all suffix branches into one shared transformer pass
  - block cross-offset suffix attention
  - aggregate losses across branches

### 4. Config split

Files:

- `configs/pi05/pi05_paligemma_ur3_rtc_full_finetune.py`
- `configs/pi05/pi05_paligemma_ur3_rtc_vlash_full_finetune.py`

Decision:

- Keep the original RTC config path available
- Add a separate VLASH-style config for training

### 5. Standalone validation scripts

Files:

- `scripts/test_shared_obs_attention_mask.py`
- `scripts/test_pi05_rtc_shared_obs_smoke.py`

What they verify:

- `test_shared_obs_attention_mask.py`
  - shared-observation attention mask blocks cross-offset suffix attention
  - prefix visibility is preserved

- `test_pi05_rtc_shared_obs_smoke.py`
  - PI0.5 shared-observation RTC forward path executes
  - `shared_predictions` is returned
  - branch count matches `max_delay`
  - loss is finite

Run commands:

```bash
python scripts/test_shared_obs_attention_mask.py
python scripts/test_pi05_rtc_shared_obs_smoke.py
```

## Training Launch

For the VLASH-style PI0.5 RTC training variant, use:

```json
{
  "name": "Train pi0.5 rtc ur3 vlash",
  "type": "debugpy",
  "request": "launch",
  "python": "/root/miniconda3/bin/python",
  "program": "/root/miniconda3/bin/torchrun",
  "args": [
    "--nnodes", "1",
    "--nproc_per_node", "2",
    "--master_port", "29501",
    "scripts/train.py",
    "--config", "configs/pi05/pi05_paligemma_ur3_rtc_vlash_full_finetune.py",
    "--work-dir", "/limx_embop/tos/users/sober/checkpoint/ur3/pi05_paligemma_ur3_rtc_vlash_full_finetune_bs4",
    "--eval-after-train",
    "--cfg-options",
    "train_dataloader.per_device_batch_size=2",
    "train_dataloader.batch_size=4",
    "runner.max_keep_ckpts=5",
    "runner.max_epochs=10",
    "runner.save_epoch_interval=1"
  ],
  "console": "integratedTerminal",
  "justMyCode": false,
  "env": {
    "CUDA_VISIBLE_DEVICES": "0, 1",
    "HF_ENDPOINT": "https://hf-mirror.com",
    "WANDB_MODE": "disabled"
  }
}
```

## Notes

- This version is training-path only.
- Inference RTC logic is unchanged.
- `shared_observation_loss_weighting='distribution'` is the closest match to the original sampled RTC objective.
- `uniform` is available if you want strict VLASH-style equal weighting across delay branches.
