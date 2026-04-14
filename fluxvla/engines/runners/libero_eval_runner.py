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

import json
import math
import os
import time
from pathlib import Path
from typing import Dict

import torch
import torch.distributed as dist
import tqdm
from safetensors.torch import load_file

from fluxvla.engines.utils import initialize_overwatch
from fluxvla.engines.utils.eval_utils import (get_libero_dummy_action,
                                              get_libero_env,
                                              save_rollout_video)
from fluxvla.engines.utils.name_map import str_to_dtype
from fluxvla.engines.utils.torch_utils import set_seed_everywhere
from libero.libero import benchmark
from ..utils.root import RUNNERS

overwatch = initialize_overwatch(__name__)


@RUNNERS.register_module()
class LiberoEvalRunner:
    """Runner for evaluating models using Hugging Face Transformers.
    This class sets up the evaluation environment, loads the model,
    and runs the evaluation process.
    Args:
        cfg (Dict): Configuration dictionary containing model and
            evaluation settings.
        seed (int): Random seed for reproducibility.
        ckpt_path (str): Path to the model checkpoint.
        model_family (str): Model family for evaluation.
        task_suite_name (str): Name of the task suite for evaluation.
        dataset (Dict): Configuration for the dataset to be used in evaluation.
        denormalize_action (Dict): Configuration for denormalizing actions.
        eval_chunk_size (int): Size of the chunks for evaluation.
            Default is 1.
        resize_size (int): Size to which images will be resized.
            Default is 224.
        num_trials_per_task (int): Number of trials per task in the evaluation.
            Default is 50.
        num_steps_wait (int): Number of steps to wait before
            starting evaluation.
            Default is 10.
        mixed_precision_dtype (str): Data type for mixed precision training.
            Default is 'bf16'.
        enable_mixed_precision_training (bool): Whether to enable mixed
            precision training.
            Default is True.
    """

    def __init__(self,
                 cfg: Dict,
                 seed: int,
                 ckpt_path: str,
                 model_family: str,
                 task_suite_name: str,
                 dataset: Dict,
                 denormalize_action: Dict,
                 eval_chunk_size: int = 1,
                 resize_size: int = 224,
                 num_trials_per_task: int = 50,
                 num_steps_wait: int = 10,
                 mixed_precision_dtype: str = 'bf16',
                 enable_mixed_precision_training: bool = True,
                 offload_inference: Dict = None):
        from fluxvla.engines import (build_dataset_from_cfg,
                                     build_transform_from_cfg,
                                     build_vla_from_cfg)
        self.device_id = overwatch.local_rank()

        self.vla = build_vla_from_cfg(cfg.model).eval()
        # Load checkpoint weights if ckpt_path is provided
        if ckpt_path is not None:
            assert Path.exists(Path(ckpt_path)), \
                f'Checkpoint path {ckpt_path} does not exist!'

            if ckpt_path.endswith('.safetensors'):
                state_dict = load_file(ckpt_path, device='cpu')
            else:
                checkpoint = torch.load(ckpt_path, map_location='cpu')
                if isinstance(checkpoint, dict) and 'model' in checkpoint:
                    state_dict = checkpoint['model']
                else:
                    state_dict = checkpoint
            self.vla.load_state_dict(state_dict, strict=True)
        self.cfg = cfg
        self.seed = seed
        self.ckpt_path = ckpt_path

        if self._use_offload:
            # Remote mode: server handles denormalization and preprocessing
            self.dataset = None
            self.denormalize_action = None
        else:
            data_stat_path = os.path.join(
                Path(self.ckpt_path).resolve().parent.parent,
                'dataset_statistics.json')  # noqa: E501
            assert os.path.exists(data_stat_path), \
                f'Dataset statistics file not found at {data_stat_path}!'
            denormalize_action['norm_stats'] = data_stat_path
            self.denormalize_action = build_transform_from_cfg(
                denormalize_action)
            dataset['task_suite_name'] = task_suite_name
            dataset['norm_stats'] = data_stat_path
            self.dataset = build_dataset_from_cfg(dataset)

            if os.path.isfile(data_stat_path):
                with open(data_stat_path, 'r') as f:
                    norm_stats = json.load(f)
                self.vla.norm_stats = norm_stats
            else:
                overwatch.warning(
                    'WARNING: No local dataset_statistics.json file found for current checkpoint.\n'  # noqa: E501
                    'You can ignore this if you are loading the base VLA (i.e. not fine-tuned) checkpoint.'  # noqa: E501
                    'Otherwise, you may run into errors when trying to call `predict_action()` due to an absent `unnorm_key`.'  # noqa: E501
                )

        self.eval_chunk_size = eval_chunk_size
        self.model_family = model_family
        self.task_suite_name = task_suite_name
        self.resize_size = resize_size
        self.num_trials_per_task = num_trials_per_task
        self.num_steps_wait = num_steps_wait
        self.mixed_precision_dtype = str_to_dtype(mixed_precision_dtype)
        self.enable_mixed_precision_training = enable_mixed_precision_training
        self.distributed_state = overwatch.distributed_state

    def run_setup(self):
        """Set up the evaluation environment and model."""
        set_seed_everywhere(self.seed)
        torch.cuda.set_device(device_id := self.device_id)  # noqa: F841
        self.vla.eval()
        self.vla.freeze_vision_backbone = True
        self.vla.freeze_llm_backbone = True
        self.vla.freeze_projector = True
        self.vla.freeze_vlm_backbone = True
        if not self._use_offload:
            self.vla.cuda(self.device_id)

    def run(self):
        """Run the evaluation process."""
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[self.task_suite_name]()
        num_tasks_in_suite = task_suite.n_tasks
        global_episodes = list(
            range(num_tasks_in_suite * self.num_trials_per_task))
        overwatch.info(f'Task suite: {self.task_suite_name}')
        overwatch.info(f'Running evaluation on {num_tasks_in_suite} tasks '
                       f'with {self.num_trials_per_task} trials each.')
        overwatch.info(f'Using model family: {self.model_family}')
        overwatch.info(f'Using resize size: {self.resize_size}')
        overwatch.info(f'Using evaluation chunk size: {self.eval_chunk_size}')
        overwatch.info(
            f'Using mixed precision dtype: {self.mixed_precision_dtype}')
        rank = overwatch.rank()
        world_size = overwatch.world_size()
        local_episodes = global_episodes[rank::world_size]
        num_local_episodes = math.ceil(len(global_episodes) / world_size)
        data_time = time.strftime('%Y_%m_%d-%H_%M_%S')
        run_id = f'EVAL-{self.task_suite_name}-{self.model_family}-{data_time}'  # noqa: E501
        local_log_filepath = os.path.join(
            Path(self.ckpt_path).resolve().parent.parent, run_id + '.txt')
        log_file = open(local_log_filepath, 'w')
        total_episodes, total_successes = torch.zeros(
            1, device=torch.cuda.current_device()), torch.zeros(
                1, device=torch.cuda.current_device())
        unnorm_key = self.task_suite_name
        if rank == 0:
            pbar = tqdm.tqdm(
                total=len(global_episodes),
                desc='Evaluation',
                dynamic_ncols=True)
        else:
            pbar = None
        if self.model_family == 'openvla':
            # In some cases, the key must be manually modified (e.g. after
            # training on a modified version of the dataset
            # with the suffix "_no_noops" in the dataset name)
            if unnorm_key not in self.vla.norm_stats and f'{unnorm_key}_no_noops' in self.vla.norm_stats:  # noqa: E501
                unnorm_key = f'{unnorm_key}_no_noops'
            assert unnorm_key in self.vla.norm_stats, f'Action un-norm key {unnorm_key} not found in VLA `norm_stats`!'  # noqa: E501

        # ---- Profiling: global accumulators ----
        _prof_keys = ['data_preprocess', 'model_inference',
                      'action_postprocess', 'denormalize', 'env_step']
        # Remote inference sub-phase keys
        _remote_keys = ['serialize', 'network', 'server_infer', 'deserialize']
        _prof_global = {k: 0.0 for k in _prof_keys}
        _prof_global['total'] = 0.0
        for rk in _remote_keys:
            _prof_global[rk] = 0.0
        _prof_global_steps = 0

        for id in range(num_local_episodes):
            if id >= len(local_episodes):
                step_tensor = torch.zeros(
                    1, device=torch.cuda.current_device())
            else:
                local_id = local_episodes[id]
                # Get task ID from local episode index
                task_id = local_id // self.num_trials_per_task
                # Get trial ID within the task
                trial_id = local_id % self.num_trials_per_task

                # Log the current task and trial
                overwatch.info(f'Evaluating Task {task_id}, Trial {trial_id}')
                log_file.write(
                    f'Evaluating Task {task_id}, Trial {trial_id}\n')

                # Initialize the task suite and environment
                # Get task
                task = task_suite.get_task(task_id)

                # Get default LIBERO initial states
                initial_states = task_suite.get_task_init_states(task_id)

                # Initialize LIBERO environment and task description
                env, task_description = get_libero_env(task, resolution=256)
                overwatch.info(f'\nTask: {task_description}')
                log_file.write(f'\nTask: {task_description}\n')

                # Reset environment
                env.reset()

                # Set initial states
                obs = env.set_init_state(initial_states[trial_id])

                # Setup
                t = 0
                replay_images = []
                if self.task_suite_name == 'libero_spatial':
                    max_steps = 220  # longest training demo has 193 steps
                elif self.task_suite_name == 'libero_object':
                    max_steps = 280  # longest training demo has 254 steps
                elif self.task_suite_name == 'libero_goal':
                    max_steps = 300  # longest training demo has 270 steps
                elif self.task_suite_name == 'libero_10':
                    max_steps = 520  # longest training demo has 505 steps
                elif self.task_suite_name == 'libero_90':
                    max_steps = 400  # longest training demo has 373 steps

                overwatch.info(f'Starting episode {trial_id+1}...')
                log_file.write(f'Starting episode {trial_id+1}...\n')

                # ---- Profiling: per-episode accumulators ----
                _prof_ep = {k: 0.0 for k in _prof_keys}
                _prof_ep['total'] = 0.0
                for rk in _remote_keys:
                    _prof_ep[rk] = 0.0
                _prof_ep_steps = 0

                while t < max_steps + self.num_steps_wait:
                    # IMPORTANT: Do nothing for the first
                    # few timesteps
                    # because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < self.num_steps_wait:
                        obs, reward, done, info = env.step(
                            get_libero_dummy_action())
                        t += 1
                        continue
                    _t_step_start = time.perf_counter()

                    # Phase 1: data preprocessing
                    _t0 = time.perf_counter()
                    obs['task_description'] = task_description
                    if self._use_offload:
                        # Remote: send raw obs, server does preprocessing
                        obs['unnorm_key'] = unnorm_key
                        batch = obs
                        if len(replay_images) == 0:
                            _first_img = next(
                                (v for v in obs.values()
                                 if hasattr(v, 'shape') and len(
                                     getattr(v, 'shape', ())) == 3),
                                None)
                            if _first_img is not None:
                                replay_images.append(_first_img)
                    else:
                        batch, replay_img = self.dataset(obs)
                        batch['unnorm_key'] = unnorm_key
                        if len(replay_images) == 0:
                            replay_images.append(replay_img)
                    _prof_ep['data_preprocess'] += time.perf_counter() - _t0

                    # Phase 2: model inference
                    _t0 = time.perf_counter()
                    with torch.autocast(
                            'cuda',
                            dtype=self.mixed_precision_dtype,
                            enabled=self.enable_mixed_precision_training):
                        with torch.no_grad():
                            actions = self.vla.predict_action(**batch)
                    _prof_ep['model_inference'] += time.perf_counter() - _t0

                    # Collect remote inference sub-phases
                    if self._use_offload and hasattr(self.vla, '_last_profile'):
                        _lp = self.vla._last_profile
                        _prof_ep['serialize'] += _lp.get('serialize_ms', 0.0)
                        _prof_ep['network'] += _lp.get('network_ms', 0.0)
                        _prof_ep['server_infer'] += _lp.get(
                            'server_infer_ms', 0.0)
                        _prof_ep['deserialize'] += _lp.get(
                            'deserialize_ms', 0.0)

                    # Phase 3: action postprocess
                    _t0 = time.perf_counter()
                    if len(actions.shape) == 3:
                        actions = actions[
                            0, :self.eval_chunk_size, :].cpu().numpy()
                    else:
                        assert len(actions.shape) == 2, \
                            f'Unexpected action shape: {actions.shape}'
                        actions = actions[0, None, :].cpu().numpy()
                    _prof_ep['action_postprocess'] += time.perf_counter() - _t0

                    for action in actions:
                        # Phase 4: denormalize
                        _t0 = time.perf_counter()
                        if self._use_offload:
                            # Server already denormalized
                            action_denormed = action
                        else:
                            inputs = dict(
                                action=action,
                                task_suite_name=self.task_suite_name,
                            )
                            action_denormed = self.denormalize_action(inputs)
                        _prof_ep['denormalize'] += time.perf_counter() - _t0

                        # Phase 5: env step
                        _t0 = time.perf_counter()
                        obs, reward, done, info = env.step(
                            action_denormed.tolist())
                        if self._use_offload:
                            _first_img = next(
                                (v for v in obs.values()
                                 if hasattr(v, 'shape') and len(
                                     getattr(v, 'shape', ())) == 3),
                                None)
                            if _first_img is not None:
                                replay_images.append(_first_img)
                        else:
                            obs['task_description'] = task_description
                            batch, replay_img = self.dataset(obs)
                            replay_images.append(replay_img)
                        _prof_ep['env_step'] += time.perf_counter() - _t0

                        if done:
                            total_successes += 1
                            break
                        t += 1

                    _prof_ep['total'] += time.perf_counter() - _t_step_start
                    _prof_ep_steps += 1

                    if done:
                        break
                # ---- Profiling: episode summary ----
                if _prof_ep_steps > 0:
                    n = _prof_ep_steps
                    _msg = (f'[Profiling] Task {task_id} Trial {trial_id} '
                            f'({n} steps):')
                    for k in _prof_keys:
                        _msg += f'  {k}={_prof_ep[k]/n*1000:.1f}ms'
                    _msg += f'  total={_prof_ep["total"]/n*1000:.1f}ms'
                    if self._use_offload:
                        _msg += (
                            f'\n  [Remote detail] '
                            f'serialize={_prof_ep["serialize"]/n:.1f}ms  '
                            f'network={_prof_ep["network"]/n:.1f}ms  '
                            f'server_infer='
                            f'{_prof_ep["server_infer"]/n:.1f}ms  '
                            f'deserialize='
                            f'{_prof_ep["deserialize"]/n:.1f}ms')
                    overwatch.info(_msg)
                    log_file.write(_msg + '\n')
                    # Accumulate to global
                    for k in _prof_keys:
                        _prof_global[k] += _prof_ep[k]
                    _prof_global['total'] += _prof_ep['total']
                    if self._use_offload:
                        for rk in _remote_keys:
                            _prof_global[rk] += _prof_ep[rk]
                    _prof_global_steps += _prof_ep_steps

                total_episodes += 1
                step_tensor = torch.ones(1, device=torch.cuda.current_device())
                # Save a replay video of the episode
                save_rollout_video(
                    replay_images,
                    local_id,
                    success=done,
                    task_description=task_description,
                    work_dir=Path(self.ckpt_path).resolve().parent.parent,
                    log_file=log_file)
                env.close()

                # except Exception as e:
                #     print(f'Error during action prediction: {e}')
                #     log_file.write(f'Caught exception: {e}\n')
                #     action = get_libero_dummy_action()
            dist.barrier()
            dist.all_reduce(step_tensor, op=dist.ReduceOp.SUM)
            if rank == 0 and pbar is not None:
                pbar.update(int(step_tensor.item()))

            global_episodes = total_episodes.clone()
            global_successes = total_successes.clone()
            dist.all_reduce(global_episodes, op=dist.ReduceOp.SUM)
            dist.all_reduce(global_successes, op=dist.ReduceOp.SUM)
            done = done.item() if isinstance(done, torch.Tensor) else done
            if rank == 0:
                # Log current results
                overwatch.info(
                    f'# episodes completed so far: {int(global_episodes[0])}')
                overwatch.info(
                    f'# successes: {int(global_successes[0])} ({global_successes[0] / global_episodes[0] * 100:.1f}%)'  # noqa: E501
                )
                log_file.write(f'Success: {done}\n')
                log_file.write(
                    f'# episodes completed so far: {global_episodes[0]}\n')
                log_file.write(
                    f'# successes: {global_successes[0]} ({global_successes[0] / global_episodes[0] * 100:.1f}%)\n'  # noqa: E501
                )
                log_file.flush()

        # ---- Profiling: global summary ----
        if _prof_global_steps > 0 and rank == 0:
            n = _prof_global_steps
            _msg = f'[Profiling Global] {n} total steps avg per step:'
            for k in _prof_keys:
                _msg += f'  {k}={_prof_global[k]/n*1000:.1f}ms'
            _msg += f'  total={_prof_global["total"]/n*1000:.1f}ms'
            if self._use_offload:
                _msg += (
                    f'\n  [Remote detail] '
                    f'serialize={_prof_global["serialize"]/n:.1f}ms  '
                    f'network={_prof_global["network"]/n:.1f}ms  '
                    f'server_infer='
                    f'{_prof_global["server_infer"]/n:.1f}ms  '
                    f'deserialize='
                    f'{_prof_global["deserialize"]/n:.1f}ms')
            overwatch.info(_msg)
            log_file.write(_msg + '\n')
            log_file.flush()

        dist.barrier()
        exit(0)
