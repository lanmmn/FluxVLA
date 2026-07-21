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
"""Single-embodiment flow-matching policy head used by VLA-JEPA."""

from functools import partial
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.fsdp.wrap import _module_wrap_policy
from torch.distributions import Beta

from fluxvla.engines import HEADS
from fluxvla.engines.losses import reduce_action_bc_loss
from fluxvla.models.blocks.cross_attention_dit import DiT
from .flow_matching_head import SinusoidalPositionalEncoding


class MLP(nn.Module):
    """Two-layer projection used for state and action decoding."""

    def __init__(self, input_dim: int, hidden_dim: int,
                 output_dim: int) -> None:
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer2(F.relu(self.layer1(x)))


class ActionEncoder(nn.Module):
    """Encode a noisy action trajectory together with diffusion time."""

    def __init__(self, action_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.layer1 = nn.Linear(action_dim, hidden_size)
        self.layer2 = nn.Linear(2 * hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.position = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions: torch.Tensor,
                timesteps: torch.Tensor) -> torch.Tensor:
        batch_size, horizon, _ = actions.shape
        if timesteps.ndim == 1:
            timesteps = timesteps.unsqueeze(1).expand(-1, horizon)
        if timesteps.shape != (batch_size, horizon):
            raise ValueError(
                'timesteps must have shape [B] or [B, T], got '
                f'{tuple(timesteps.shape)} for actions {tuple(actions.shape)}')

        action_features = self.layer1(actions)
        time_features = self.position(timesteps).to(action_features.dtype)
        features = torch.cat([action_features, time_features], dim=-1)
        features = F.silu(self.layer2(features))
        return self.layer3(features)


@HEADS.register_module()
class VLAJEPAFlowMatchingHead(nn.Module):
    """VLA-JEPA's single-embodiment flow-matching action policy.

    Unlike :class:`FlowMatchingHead`, this head intentionally uses ordinary
    linear layers rather than category-specific parameters. LIBERO-10 has one
    embodiment and the layout matches the released VLA-JEPA robot fine-tuning
    recipe.
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        state_dim: int = 8,
        input_embedding_dim: int = 768,
        action_dim: int = 7,
        action_horizon: int = 7,
        backbone_embedding_dim: int = 2048,
        num_inference_timesteps: int = 4,
        num_target_vision_tokens: int = 32,
        add_positional_embeddings: bool = True,
        max_seq_len: int = 1024,
        num_timestep_buckets: int = 1000,
        noise_s: float = 0.999,
        noise_beta_alpha: float = 1.5,
        noise_beta_beta: float = 1.0,
        diffusion_model_cfg: Optional[Dict] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        if action_horizon <= 0 or action_dim <= 0:
            raise ValueError('action_horizon and action_dim must be positive')

        if diffusion_model_cfg is None:
            diffusion_model_cfg = dict(
                attention_head_dim=64,
                num_attention_heads=12,
                cross_attention_dim=backbone_embedding_dim,
                num_layers=16,
                output_dim=hidden_size,
                dropout=0.2,
                final_dropout=True,
                interleave_self_attention=True,
                norm_type='ada_norm',
                positional_embeddings=None,
            )

        self.model = DiT(**diffusion_model_cfg)
        self.state_encoder = (
            MLP(state_dim, hidden_size, input_embedding_dim)
            if state_dim > 0 else None)
        self.action_encoder = ActionEncoder(action_dim, input_embedding_dim)
        self.action_decoder = MLP(hidden_size, hidden_size, action_dim)
        self.future_tokens = nn.Embedding(num_target_vision_tokens,
                                          input_embedding_dim)
        nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)

        self.position_embedding = None
        if add_positional_embeddings:
            self.position_embedding = nn.Embedding(max_seq_len,
                                                   input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.backbone_embedding_dim = backbone_embedding_dim
        self.num_inference_timesteps = num_inference_timesteps
        self.num_timestep_buckets = num_timestep_buckets
        self.noise_s = noise_s
        self.beta_dist = Beta(noise_beta_alpha, noise_beta_beta)

    def _validate_inputs(self, input_features: torch.Tensor,
                         states: Optional[torch.Tensor]) -> None:
        if input_features.ndim != 3:
            raise ValueError('input_features must have shape [B, N, D], got '
                             f'{tuple(input_features.shape)}')
        if input_features.shape[-1] != self.backbone_embedding_dim:
            raise ValueError('Unexpected VLM feature dimension: expected '
                             f'{self.backbone_embedding_dim}, got '
                             f'{input_features.shape[-1]}')
        if self.state_encoder is not None and states is None:
            raise ValueError('states are required when state_dim is non-zero')

    def _state_features(
            self, states: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if self.state_encoder is None:
            return None
        if states.ndim == 3 and states.shape[1] == 1:
            states = states[:, 0]
        if states.ndim != 2:
            raise ValueError('states must have shape [B, D] or [B, 1, D], '
                             f'got {tuple(states.shape)}')
        return self.state_encoder(states).unsqueeze(1)

    def _action_features(self, actions: torch.Tensor,
                         timesteps: torch.Tensor) -> torch.Tensor:
        features = self.action_encoder(actions, timesteps)
        if self.position_embedding is not None:
            positions = torch.arange(
                features.shape[1], device=features.device, dtype=torch.long)
            features = features + self.position_embedding(positions).unsqueeze(
                0)
        return features

    def _denoise(self, actions: torch.Tensor, input_features: torch.Tensor,
                 state_features: Optional[torch.Tensor],
                 timesteps: torch.Tensor) -> torch.Tensor:
        action_features = self._action_features(actions, timesteps)
        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(
            actions.shape[0], -1, -1)
        suffix = [future_tokens, action_features]
        if state_features is not None:
            suffix.insert(0, state_features)
        hidden_states = torch.cat(suffix, dim=1)
        output = self.model(
            hidden_states=hidden_states,
            encoder_hidden_states=input_features,
            timestep=timesteps,
            return_all_hidden_states=False,
        )
        return self.action_decoder(output[:, -actions.shape[1]:])

    def forward(self,
                input_features: torch.Tensor,
                states: Optional[torch.Tensor],
                actions: torch.Tensor,
                action_masks: Optional[torch.Tensor] = None,
                sample_weight: Optional[torch.Tensor] = None,
                **kwargs) -> Dict[str, torch.Tensor]:
        self._validate_inputs(input_features, states)
        expected = (input_features.shape[0], self.action_horizon,
                    self.action_dim)
        if tuple(actions.shape) != expected:
            raise ValueError(f'actions must have shape {expected}, got '
                             f'{tuple(actions.shape)}')

        noise = torch.randn_like(actions)
        time = self.sample_time(actions.shape[0], actions.device,
                                actions.dtype)
        noisy_actions = ((1 - time[:, None, None]) * noise +
                         time[:, None, None] * actions)
        velocity = actions - noise
        discrete_time = (time * self.num_timestep_buckets).long()
        prediction = self._denoise(noisy_actions, input_features,
                                   self._state_features(states), discrete_time)
        losses = F.mse_loss(prediction, velocity, reduction='none')
        loss = reduce_action_bc_loss(
            losses, action_mask=action_masks, sample_weight=sample_weight)
        return {'pred_actions': prediction, 'loss': loss}

    @torch.no_grad()
    def predict_action(self, input_features: torch.Tensor,
                       states: Optional[torch.Tensor],
                       **kwargs) -> torch.Tensor:
        self._validate_inputs(input_features, states)
        batch_size = input_features.shape[0]
        actions = torch.randn(
            batch_size,
            self.action_horizon,
            self.action_dim,
            device=input_features.device,
            dtype=input_features.dtype,
        )
        state_features = self._state_features(states)
        step_size = 1.0 / self.num_inference_timesteps
        for step in range(self.num_inference_timesteps):
            discrete_time = int(step * self.num_timestep_buckets /
                                self.num_inference_timesteps)
            timesteps = torch.full((batch_size, ),
                                   discrete_time,
                                   device=actions.device)
            actions = actions + step_size * self._denoise(
                actions, input_features, state_features, timesteps)
        return actions

    def sample_time(self, batch_size: int, device: torch.device,
                    dtype: torch.dtype) -> torch.Tensor:
        sample = self.beta_dist.sample([batch_size]).to(
            device=device, dtype=dtype)
        return (self.noise_s - sample) / self.noise_s

    def get_fsdp_wrapping_policy(self) -> Callable:
        return partial(_module_wrap_policy, module_classes={DiT})
