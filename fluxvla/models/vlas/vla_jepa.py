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
"""Native FluxVLA implementation of VLA-JEPA."""

from functools import partial
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy

from fluxvla.engines import VLAS, build_tokenizer_from_cfg
from fluxvla.models.third_party_models.vjepa2 import (
    ACBlock, VisionTransformerPredictorAC)
from .llava_vla import LlavaVLA


@VLAS.register_module()
class VLAJEPA(LlavaVLA):
    """Qwen3-VL action policy with an action-conditioned latent predictor."""

    def __init__(
        self,
        vla_head: Dict,
        vlm_backbone: Dict,
        tokenizer: Dict,
        vj_encoder_path: str = 'facebook/vjepa2-vitl-fpc64-256',
        vj_predictor_cfg: Optional[Dict] = None,
        num_views: int = 2,
        num_frames: int = 8,
        num_action_tokens_per_timestep: int = 8,
        num_embodied_action_tokens: int = 32,
        action_token_format: str = '<|action_{}|>',
        embodied_action_token: str = '<|embodied_action|>',
        world_model_loss_weight: float = 0.1,
        vj_encoder=None,
        world_predictor=None,
        **kwargs,
    ) -> None:
        super().__init__(
            vla_head=vla_head, vlm_backbone=vlm_backbone, **kwargs)
        if num_frames <= 0 or num_views <= 0:
            raise ValueError('num_frames and num_views must be positive')
        if world_model_loss_weight < 0:
            raise ValueError('world_model_loss_weight must be non-negative')

        self.policy_tokenizer = build_tokenizer_from_cfg(tokenizer)
        self.num_views = num_views
        self.num_frames = num_frames
        self.num_action_tokens_per_timestep = (num_action_tokens_per_timestep)
        self.num_embodied_action_tokens = num_embodied_action_tokens
        self.world_model_loss_weight = world_model_loss_weight

        if vj_encoder is None:
            from transformers import AutoModel
            vj_encoder = AutoModel.from_pretrained(vj_encoder_path)
        self.vj_encoder = vj_encoder
        self.vj_encoder.requires_grad_(False)
        self.vj_encoder.eval()

        tubelet_size = int(self.vj_encoder.config.tubelet_size)
        if num_frames % tubelet_size != 0:
            raise ValueError(
                f'num_frames={num_frames} must be divisible by V-JEPA2 '
                f'tubelet_size={tubelet_size}')
        self.latent_frames = num_frames // tubelet_size
        self.num_world_steps = self.latent_frames - 1
        if self.num_world_steps <= 0:
            raise ValueError('VLA-JEPA requires at least two latent frames')

        self.world_action_tokens = [
            action_token_format.format(i) for i in range(self.num_world_steps)
        ]
        self.embodied_action_token = embodied_action_token
        self.world_action_token_ids = self._token_ids(self.world_action_tokens)
        self.embodied_action_token_id = self._token_ids(
            [embodied_action_token])[0]
        self._resize_vlm_embeddings(len(self.policy_tokenizer))

        if world_predictor is None:
            predictor_cfg = dict(vj_predictor_cfg or {})
            predictor_cfg.setdefault('num_frames', self.latent_frames)
            predictor_cfg.setdefault('img_size',
                                     self.vj_encoder.config.image_size)
            predictor_cfg.setdefault('tubelet_size', 1)
            predictor_cfg.setdefault(
                'embed_dim', self.vj_encoder.config.hidden_size * num_views)
            predictor_cfg.setdefault('action_embed_dim',
                                     self.vlm_backbone.embed_dim)
            predictor_cfg.setdefault('num_add_tokens',
                                     num_action_tokens_per_timestep)
            world_predictor = VisionTransformerPredictorAC(**predictor_cfg)
        self.world_predictor = world_predictor
        self.all_module_keys = [
            'vlm_backbone', 'vla_head', 'vj_encoder', 'world_predictor'
        ]

    def _token_ids(self, tokens: List[str]) -> List[int]:
        ids = self.policy_tokenizer.convert_tokens_to_ids(tokens)
        if isinstance(ids, int):
            ids = [ids]
        if len(set(ids)) != len(tokens):
            raise ValueError(
                f'VLA-JEPA special tokens do not have unique IDs: {tokens}')
        unknown_id = getattr(self.policy_tokenizer, 'unk_token_id', None)
        if unknown_id is not None and any(token_id == unknown_id
                                          for token_id in ids):
            raise ValueError(
                'VLA-JEPA special tokens are missing from tokenizer; add '
                'them through additional_special_tokens')
        return list(ids)

    def _resize_vlm_embeddings(self, vocabulary_size: int) -> None:
        vlm = getattr(self.vlm_backbone, 'vlm', None)
        if vlm is None or not hasattr(vlm, 'resize_token_embeddings'):
            raise TypeError(
                'VLAJEPA requires a VLM backbone with resize_token_embeddings')
        embedding = vlm.get_input_embeddings()
        if embedding.num_embeddings != vocabulary_size:
            vlm.resize_token_embeddings(vocabulary_size)

    def train(self, mode: bool = True):
        super().train(mode)
        # The source encoder is a fixed target representation and must stay
        # deterministic even while the policy and predictor train.
        self.vj_encoder.eval()
        return self

    def _extract_condition_tokens(
            self, last_hidden_state: torch.Tensor,
            lang_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        text_length = lang_tokens.shape[1]
        if last_hidden_state.shape[1] < text_length:
            raise ValueError(
                'VLM hidden sequence is shorter than the token sequence: '
                f'{last_hidden_state.shape[1]} < {text_length}')
        text_hidden = last_hidden_state[:, -text_length:, :]
        world_ids = torch.as_tensor(
            self.world_action_token_ids,
            device=lang_tokens.device,
            dtype=lang_tokens.dtype,
        )

        world_features = []
        embodied_features = []
        expected_world_tokens = (
            self.num_world_steps * self.num_action_tokens_per_timestep)
        for sample_ids, sample_hidden in zip(lang_tokens, text_hidden):
            world_mask = torch.isin(sample_ids, world_ids)
            embodied_mask = sample_ids.eq(self.embodied_action_token_id)
            world_count = int(world_mask.sum().item())
            embodied_count = int(embodied_mask.sum().item())
            if world_count != expected_world_tokens:
                raise ValueError(
                    'VLA-JEPA prompt has an invalid world-action token count: '
                    f'expected {expected_world_tokens}, got {world_count}. '
                    'Check prompt truncation and tokenizer metadata.')
            if embodied_count != self.num_embodied_action_tokens:
                raise ValueError(
                    'VLA-JEPA prompt has an invalid embodied-action token '
                    f'count: expected {self.num_embodied_action_tokens}, '
                    f'got {embodied_count}.')
            world_features.append(sample_hidden[world_mask])
            embodied_features.append(sample_hidden[embodied_mask])
        return torch.stack(world_features), torch.stack(embodied_features)

    def _encode_video(self, pixel_values_videos: torch.Tensor) -> torch.Tensor:
        if pixel_values_videos.ndim != 6:
            raise ValueError(
                'pixel_values_videos must have shape [B, V, T, C, H, W], '
                f'got {tuple(pixel_values_videos.shape)}')
        batch_size, views, frames, channels, height, width = (
            pixel_values_videos.shape)
        if (views != self.num_views or frames != self.num_frames
                or channels != 3):
            raise ValueError(
                'Unexpected V-JEPA2 video shape: expected '
                f'[B, {self.num_views}, {self.num_frames}, 3, H, W], got '
                f'{tuple(pixel_values_videos.shape)}')
        videos = pixel_values_videos.reshape(batch_size * views, frames,
                                             channels, height, width)
        self.vj_encoder.eval()
        with torch.no_grad():
            features = self.vj_encoder.get_vision_features(
                pixel_values_videos=videos)
        if features.ndim != 3:
            raise ValueError(
                'V-JEPA2 features must have shape [B*V, N, D], got '
                f'{tuple(features.shape)}')
        features = features.reshape(batch_size, views, features.shape[1],
                                    features.shape[2])
        return torch.cat([features[:, view] for view in range(views)], dim=-1)

    def _world_model_loss(self, world_features: torch.Tensor,
                          pixel_values_videos: torch.Tensor,
                          frame_masks: torch.Tensor) -> torch.Tensor:
        latents = self._encode_video(pixel_values_videos)
        if latents.shape[1] % self.latent_frames != 0:
            raise ValueError(
                f'V-JEPA2 token count {latents.shape[1]} is not divisible by '
                f'latent frame count {self.latent_frames}')
        tokens_per_frame = latents.shape[1] // self.latent_frames
        context = latents[:, :tokens_per_frame * self.num_world_steps]
        target = latents[:, tokens_per_frame:]
        prediction = self.world_predictor(context, world_features)
        if prediction.shape != target.shape:
            raise ValueError(
                'World predictor/target shape mismatch: '
                f'{tuple(prediction.shape)} != {tuple(target.shape)}')

        if frame_masks.shape != (latents.shape[0], self.num_frames):
            raise ValueError('frame_masks must have shape '
                             f'[{latents.shape[0]}, {self.num_frames}], got '
                             f'{tuple(frame_masks.shape)}')
        valid_samples = frame_masks.to(torch.bool).all(dim=1)
        sample_losses = F.l1_loss(
            prediction, target, reduction='none').mean(dim=(1, 2))
        if valid_samples.any():
            return sample_losses[valid_samples].mean()
        return prediction.sum() * 0.0

    def _forward_vlm(self, images: torch.Tensor, lang_tokens: torch.Tensor,
                     img_masks: torch.Tensor, lang_masks: torch.Tensor,
                     image_grid_thw: Optional[torch.Tensor]) -> torch.Tensor:
        last_hidden_state, _, _ = self.vlm_backbone(
            images=images,
            lang_tokens=lang_tokens,
            img_masks=img_masks,
            lang_masks=lang_masks,
            image_grid_thw=image_grid_thw,
        )
        return last_hidden_state

    def forward(
        self,
        lang_tokens: torch.LongTensor,
        lang_masks: torch.Tensor,
        images: torch.Tensor,
        img_masks: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        action_masks: torch.Tensor,
        pixel_values_videos: torch.Tensor,
        frame_masks: torch.Tensor,
        image_grid_thw: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        hidden = self._forward_vlm(images, lang_tokens, img_masks, lang_masks,
                                   image_grid_thw)
        world_features, embodied_features = self._extract_condition_tokens(
            hidden, lang_tokens)
        action_output = self.vla_head(
            input_features=embodied_features,
            states=states,
            actions=actions,
            action_masks=action_masks,
            sample_weight=kwargs.get('sample_weight'),
        )
        world_loss = self._world_model_loss(world_features,
                                            pixel_values_videos, frame_masks)
        action_loss = action_output['loss']
        total_loss = action_loss + self.world_model_loss_weight * world_loss
        return {
            'loss': total_loss,
            'action_loss': action_loss,
            'wm_loss': world_loss,
            'pred_actions': action_output['pred_actions'],
        }

    @torch.no_grad()
    def predict_action(
        self,
        images: torch.Tensor,
        lang_tokens: torch.LongTensor,
        states: torch.Tensor,
        img_masks: torch.Tensor,
        lang_masks: torch.Tensor,
        image_grid_thw: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        hidden = self._forward_vlm(images, lang_tokens, img_masks, lang_masks,
                                   image_grid_thw)
        _, embodied_features = self._extract_condition_tokens(
            hidden, lang_tokens)
        actions = self.vla_head.predict_action(
            input_features=embodied_features, states=states)
        return actions.float()

    def get_fsdp_wrapping_policy(self):
        base_policy = super().get_fsdp_wrapping_policy()
        module_classes = {ACBlock}
        try:
            from transformers.models.vjepa2.modeling_vjepa2 import VJEPA2Layer
            module_classes.add(VJEPA2Layer)
        except ImportError:
            pass
        world_policy = partial(
            _module_wrap_policy, module_classes=module_classes)
        return partial(_or_policy, policies=[base_policy, world_policy])
