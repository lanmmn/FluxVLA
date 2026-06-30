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

import argparse
import time

from mmengine import Config

import fluxvla.collators  # noqa: F401
import fluxvla.datasets  # noqa: F401
import fluxvla.engines.operators  # noqa: F401
import fluxvla.tokenizers  # noqa: F401
import fluxvla.transforms  # noqa: F401
from fluxvla.engines import build_runner_from_cfg
from fluxvla.engines.utils.torch_utils import \
    configure_inference_attention_defaults
from fluxvla.models.backbones.vlms.eagle import EagleBackbone  # noqa: F401
from fluxvla.models.backbones.vlms.eagle import \
    EagleInferenceBackbone  # noqa: F401
from fluxvla.models.heads.flow_matching_head import \
    FlowMatchingHead  # noqa: F401
from fluxvla.models.heads.flow_matching_inference_head import \
    FlowMatchingInferenceHead  # noqa: F401
from fluxvla.models.vlas.llava_vla import LlavaVLA  # noqa: F401


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to the configuration file.',
    )
    parser.add_argument(
        '--ckpt-path',
        type=str,
        default=None,
        help='Path to the checkpoint file.')
    args = parser.parse_args()
    return args


def inference(args, cfg):
    runner = build_runner_from_cfg(cfg.inference_runner)
    print(runner)


if __name__ == '__main__':
    startup_t0 = time.perf_counter()
    configure_inference_attention_defaults()
    args = parse_args()
    stage_t0 = time.perf_counter()
    cfg = Config.fromfile(args.config)
    if args.ckpt_path is not None:
        cfg.inference.ckpt_path = args.ckpt_path
    cfg.inference.cfg = cfg
    stage_t0 = time.perf_counter()
    inference_runner = build_runner_from_cfg(cfg.inference)
    build_elapsed = time.perf_counter() - stage_t0
    print(f'[Startup] build_runner_from_cfg: {build_elapsed:.1f}s', flush=True)
    stage_t0 = time.perf_counter()
    inference_runner.run_setup()
    print(
        f'[Startup] run_setup_ros: {time.perf_counter() - stage_t0:.1f}s',
        flush=True)
    startup_elapsed = time.perf_counter() - startup_t0
    print(
        f'[Startup] total_before_interactive_loop: {startup_elapsed:.1f}s',
        flush=True)

    inference_runner.run()
