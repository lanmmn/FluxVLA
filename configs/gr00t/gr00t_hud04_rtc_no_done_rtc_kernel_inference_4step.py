# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Native four-step Basket teacher using the same RTC kernel inference path."""

from copy import deepcopy as _deepcopy
from pathlib import Path as _Path

_base_globals = {}
_base_candidates = [
    _Path('configs/gr00t/gr00t_hud04_rtc_no_done_rtc_kernel_inference.py'),
    _Path('/workspace/FluxVLA/configs/gr00t/'
          'gr00t_hud04_rtc_no_done_rtc_kernel_inference.py'),
    _Path('/home/limx/sober/FluxVLA/configs/gr00t/'
          'gr00t_hud04_rtc_no_done_rtc_kernel_inference.py'),
]
_base_path = next(path for path in _base_candidates if path.exists())
exec(compile(_base_path.read_text(), str(_base_path), 'exec'), _base_globals)

WBT_DISCRETE_ACTION_DIMS = _base_globals['WBT_DISCRETE_ACTION_DIMS']
WBT_DISCRETE_STATE_DIMS = _base_globals['WBT_DISCRETE_STATE_DIMS']
WBT_DISCRETE_NORM_TYPE = _base_globals['WBT_DISCRETE_NORM_TYPE']
WBT_CONTINUOUS_NORM_TYPE = _base_globals['WBT_CONTINUOUS_NORM_TYPE']
WBT_NORM_KW = _deepcopy(_base_globals['WBT_NORM_KW'])
WBT_DENORM_KW = _deepcopy(_base_globals['WBT_DENORM_KW'])

model = _deepcopy(_base_globals['model'])
train_dataloader = _deepcopy(_base_globals['train_dataloader'])
runner = _deepcopy(_base_globals['runner'])
inference = _deepcopy(_base_globals['inference'])
inference_model = _deepcopy(_base_globals['inference_model'])

# Keep the teacher's native denoising step count while retaining the exact
# accelerated backbone/head, RTC settings, cameras, and operator of the
# existing direct-two-step baseline.
inference_model['vla_head']['num_inference_timesteps'] = 4

del _base_candidates, _base_globals, _base_path, _deepcopy, _Path
