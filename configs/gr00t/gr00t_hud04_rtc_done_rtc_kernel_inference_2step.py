# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Two-camera done-dim WBT inference for the teacher run directly at two steps."""

from copy import deepcopy as _deepcopy
from pathlib import Path as _Path

_base_globals = {}
_base_candidates = [
    _Path('configs/gr00t/gr00t_hud04_rtc_done_rtc_kernel_inference.py'),
    _Path('/workspace/FluxVLA/configs/gr00t/'
          'gr00t_hud04_rtc_done_rtc_kernel_inference.py'),
    _Path('/data/FluxVLA/configs/gr00t/'
          'gr00t_hud04_rtc_done_rtc_kernel_inference.py'),
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

# Keep the exact teacher model, cameras, RTC prefix, done state machine, and
# prompt order from the four-step config; change only the denoising step count.
inference_model['vla_head']['num_inference_timesteps'] = 2

del _base_candidates, _base_globals, _base_path, _deepcopy, _Path
