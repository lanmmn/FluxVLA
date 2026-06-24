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
import copy

from mmengine import Config, DictAction

from fluxvla.engines import build_runner_from_cfg, initialize_overwatch

overwatch = initialize_overwatch(__name__)


def _as_list(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _get_eval_runner_cfg(cfg):
    """Return runner config from old or namespaced eval config."""
    if hasattr(cfg.eval, 'runner'):
        return copy.deepcopy(cfg.eval.runner)
    return copy.deepcopy(cfg.eval)


def _get_eval_value(cfg, key, default=None):
    """Read eval runner value from old or namespaced eval config."""
    if hasattr(cfg.eval, 'runner') and hasattr(cfg.eval.runner, key):
        return getattr(cfg.eval.runner, key)
    return cfg.eval.get(key, default)


def _resolve_suite_max_steps(max_steps, suite):
    if isinstance(max_steps, dict):
        return max_steps.get(suite)
    return max_steps


def _cleanup_eval_runner(eval_runner):
    if eval_runner is None:
        return
    cleanup = getattr(eval_runner, 'cleanup', None)
    if callable(cleanup):
        cleanup()


def _run_eval(cfg, args, suite_name=None):
    eval_cfg = _get_eval_runner_cfg(cfg)
    if suite_name is not None:
        eval_cfg.task_suite_name = suite_name
        eval_cfg.max_steps = _resolve_suite_max_steps(
            _get_eval_value(cfg, 'max_steps'), suite_name)
    eval_cfg.cfg = cfg
    eval_cfg.ckpt_path = args.ckpt_path
    if hasattr(eval_cfg,
               'processor') and not hasattr(eval_cfg.processor, 'model_path'):
        eval_cfg.processor.model_path = args.ckpt_path
    eval_runner = None
    try:
        eval_runner = build_runner_from_cfg(eval_cfg)
        eval_runner.run_setup()
        eval_runner.run()
    finally:
        _cleanup_eval_runner(eval_runner)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a model with the given configuration.')
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
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help=  # noqa: E251
        'override some settings in the used config, the key-value pair in xxx=yyy format'  # noqa: E501
    )
    args, unknown = parser.parse_known_args()
    return args, unknown


if __name__ == '__main__':
    args, _ = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    suite_names = _as_list(_get_eval_value(cfg, 'task_suite_name'))
    for suite_name in suite_names:
        _run_eval(cfg, args, suite_name=suite_name)
