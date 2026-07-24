# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Two-camera done-dim WBT inference for the original four-step teacher."""

from copy import deepcopy as _deepcopy
from pathlib import Path as _Path

_base_globals = {}
_base_candidates = [
    _Path('configs/gr00t/gr00t_hud04_rtc_done_full_finetune.py'),
    _Path('/workspace/FluxVLA/configs/gr00t/'
          'gr00t_hud04_rtc_done_full_finetune.py'),
    _Path('/data/FluxVLA/configs/gr00t/'
          'gr00t_hud04_rtc_done_full_finetune.py'),
]
_base_path = next(path for path in _base_candidates if path.exists())
exec(compile(_base_path.read_text(), str(_base_path), 'exec'), _base_globals)

WBT_DISCRETE_ACTION_DIMS = _base_globals['WBT_DISCRETE_ACTION_DIMS']
WBT_DISCRETE_STATE_DIMS = _base_globals['WBT_DISCRETE_STATE_DIMS']
WBT_DISCRETE_NORM_TYPE = _base_globals['WBT_DISCRETE_NORM_TYPE']
WBT_CONTINUOUS_NORM_TYPE = _base_globals['WBT_CONTINUOUS_NORM_TYPE']
WBT_DONE_DIM_INDEX = _base_globals['WBT_DONE_DIM_INDEX']
WBT_NORM_KW = _deepcopy(_base_globals['WBT_NORM_KW'])
WBT_DENORM_KW = _deepcopy(_base_globals['WBT_DENORM_KW'])

model = _deepcopy(_base_globals['model'])
train_dataloader = _deepcopy(_base_globals['train_dataloader'])
runner = _deepcopy(_base_globals['runner'])
inference = _deepcopy(_base_globals['inference'])
inference_model = _deepcopy(_base_globals['inference_model'])

inference['publish_rate'] = 30
inference['async_remaining_actions_threshold'] = 8
inference['execute_horizon'] = 16
inference['target_hz'] = 50
inference['rtc_config']['prefix_len'] = 4
inference['camera_names'] = ['head', 'left_wrist']
inference['stop_on_final_done'] = True
# The June checkpoint uses a 2-dim open/closed hand state. When physical hand
# feedback is unavailable, feed back the most recently executed model hand
# action instead of waiting on /brainco1/hand/state.
inference['operator']['use_finger_state'] = False

inference_model['vlm_backbone']['type'] = 'EagleInferenceBackbone'
inference_model['vla_head'].pop('rtc_training_config', None)
inference_model['vla_head'].update(
    type='FlowMatchingInferenceHead',
    num_inference_timesteps=4,
    max_input_seq_len=580,
    diffusion_model_cfg=dict(
        attention_head_dim=48,
        cross_attention_dim=2048,
        dropout=0.2,
        final_dropout=True,
        interleave_self_attention=True,
        norm_type='ada_norm',
        num_attention_heads=32,
        num_layers=16,
        output_dim=1024,
        positional_embeddings=None,
    ),
)

del _base_candidates, _base_globals, _base_path, _deepcopy, _Path
