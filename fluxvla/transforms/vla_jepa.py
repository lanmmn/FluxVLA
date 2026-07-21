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
"""Prompt and temporal-video transforms for VLA-JEPA."""

from typing import List

import numpy as np
import torch

from fluxvla.engines import TRANSFORMS


def build_vla_jepa_special_tokens(
        num_world_steps: int = 3,
        action_token_format: str = '<|action_{}|>',
        embodied_action_token: str = '<|embodied_action|>') -> List[str]:
    """Return the ordered added-token vocabulary used by VLA-JEPA."""
    if num_world_steps <= 0:
        raise ValueError('num_world_steps must be positive')
    tokens = [action_token_format.format(i) for i in range(num_world_steps)]
    tokens.append(embodied_action_token)
    return tokens


@TRANSFORMS.register_module()
class VLAJEPAPrompter:
    """Build the latent-dynamics prompt used for training and rollout."""

    def __init__(
        self,
        num_frames: int = 8,
        tubelet_size: int = 2,
        num_action_tokens_per_timestep: int = 8,
        num_embodied_action_tokens: int = 32,
        action_token_format: str = '<|action_{}|>',
        embodied_action_token: str = '<|embodied_action|>',
        prompt_template: str = (
            'Your task is {instruction}. Infer the temporal dynamics from '
            'frames {actions} and produce the corresponding policy actions '
            '{embodied_actions}.'),
        **kwargs,
    ) -> None:
        if num_frames <= tubelet_size or num_frames % tubelet_size != 0:
            raise ValueError(
                'num_frames must be divisible by tubelet_size and include '
                'at least two tubelets')
        self.num_world_steps = num_frames // tubelet_size - 1
        self.num_action_tokens_per_timestep = (num_action_tokens_per_timestep)
        self.num_embodied_action_tokens = num_embodied_action_tokens
        self.action_token_format = action_token_format
        self.embodied_action_token = embodied_action_token
        self.prompt_template = prompt_template

    def __call__(self, inputs: dict) -> dict:
        if 'task_description' not in inputs:
            raise KeyError("VLAJEPAPrompter requires 'task_description'")
        action_prompt = ''.join(
            self.action_token_format.format(step) *
            self.num_action_tokens_per_timestep
            for step in range(self.num_world_steps))
        embodied_prompt = (
            self.embodied_action_token * self.num_embodied_action_tokens)
        inputs['prompt'] = self.prompt_template.format(
            instruction=str(inputs['task_description']),
            actions=action_prompt,
            embodied_actions=embodied_prompt,
        )
        return inputs


@TRANSFORMS.register_module()
class PrepareVLAJEPAVideo:
    """Split current Qwen images from a two-view temporal video window.

    ``ProcessParquetInputs`` emits images in view-major order: all timestamps
    for view 0 followed by all timestamps for view 1. This transform keeps the
    first frame of each view under ``images`` and preprocesses every frame for
    V-JEPA2 under ``pixel_values_videos``.
    """

    def __init__(self,
                 video_processor_path: str,
                 num_views: int = 2,
                 num_frames: int = 8,
                 processor=None,
                 **kwargs) -> None:
        if num_views <= 0 or num_frames <= 0:
            raise ValueError('num_views and num_frames must be positive')
        if processor is None:
            from transformers import AutoVideoProcessor
            processor = AutoVideoProcessor.from_pretrained(
                video_processor_path)
        self.processor = processor
        self.num_views = num_views
        self.num_frames = num_frames

    @staticmethod
    def _to_chw_tensor(image) -> torch.Tensor:
        tensor = torch.as_tensor(np.asarray(image))
        if tensor.ndim != 3:
            raise ValueError(
                f'Each video frame must be rank 3, got {tuple(tensor.shape)}')
        if tensor.shape[0] == 3:
            return tensor.contiguous()
        if tensor.shape[-1] == 3:
            return tensor.permute(2, 0, 1).contiguous()
        raise ValueError('Each video frame must be CHW or HWC RGB, got '
                         f'{tuple(tensor.shape)}')

    def _process_view(self, images: List) -> torch.Tensor:
        video = torch.stack([self._to_chw_tensor(image) for image in images])
        output = self.processor(videos=video, return_tensors='pt')
        if 'pixel_values_videos' not in output:
            raise KeyError(
                "V-JEPA2 processor did not return 'pixel_values_videos'")
        pixels = output['pixel_values_videos']
        if pixels.ndim == 5 and pixels.shape[0] == 1:
            pixels = pixels[0]
        expected_prefix = (self.num_frames, 3)
        if pixels.ndim != 4 or tuple(pixels.shape[:2]) != expected_prefix:
            raise ValueError(
                'V-JEPA2 processor must return [T, 3, H, W] per view, got '
                f'{tuple(pixels.shape)}')
        return pixels

    def __call__(self, inputs: dict) -> dict:
        if 'images' not in inputs:
            raise KeyError("PrepareVLAJEPAVideo requires 'images'")
        images = list(inputs['images'])
        expected = self.num_views * self.num_frames
        if len(images) != expected:
            raise ValueError(
                f'Expected {expected} view-major frames, got {len(images)}')

        videos = []
        current_images = []
        for view in range(self.num_views):
            start = view * self.num_frames
            view_images = images[start:start + self.num_frames]
            current_images.append(view_images[0])
            videos.append(self._process_view(view_images))

        inputs['images'] = current_images
        inputs['pixel_values_videos'] = torch.stack(videos, dim=0)
        return inputs
