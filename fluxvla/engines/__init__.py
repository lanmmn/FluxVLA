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

from .utils import *  # noqa: F401, F403


def _is_optional_torch_distributed_error(exc: ModuleNotFoundError) -> bool:
    name = getattr(exc, 'name', '') or ''
    return name.startswith('torch.distributed') or name.startswith(
        'torch._C._distributed_c10d')


for _module in ('metrics', 'operators', 'processors', 'runners'):
    try:
        exec(f'from .{_module} import *')
    except ModuleNotFoundError as exc:
        if not _is_optional_torch_distributed_error(exc):
            raise
