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
"""HUD04 kernel inference through OliRTCInferenceRunner and OliOperator.

This keeps Oli's prompt-ID/execution-count interaction and adds asynchronous
RTC prefix conditioning. One requested execution means one generated action
chunk; overlapping RTC prefix steps are not executed twice.
"""

from copy import deepcopy as _deepcopy
from pathlib import Path as _Path

_base_globals = {}
_base_candidates = [
    _Path('configs/gr00t/gr00t_hud04_rtc_no_done_oli_kernel_inference.py'),
    _Path('/workspace/FluxVLA/configs/gr00t/'
          'gr00t_hud04_rtc_no_done_oli_kernel_inference.py'),
]
_base_path = next(path for path in _base_candidates if path.exists())
exec(compile(_base_path.read_text(), str(_base_path), 'exec'), _base_globals)

model = _deepcopy(_base_globals['model'])
train_dataloader = _deepcopy(_base_globals['train_dataloader'])
runner = _deepcopy(_base_globals['runner'])
inference_model = _deepcopy(_base_globals['inference_model'])
inference = _deepcopy(_base_globals['inference'])

inference.update(
    type='OliRTCInferenceRunner',
    async_remaining_actions_threshold=8,
    rtc_config=dict(
        enabled=True,
        method='prefix',
        prefix_len=7,
    ),
)

del _base_candidates, _base_globals, _base_path, _deepcopy, _Path
