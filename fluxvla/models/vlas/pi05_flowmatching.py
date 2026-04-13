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

from typing import List, Optional

import torch
import torch.nn.functional as F
from torch.profiler import record_function

from fluxvla.engines import VLAS
from fluxvla.engines.utils.model_utils import (create_sinusoidal_pos_embedding,
                                               make_att_2d_masks)
from fluxvla.engines.utils.overwatch import initialize_overwatch
from .pi0_flowmatching import PI0FlowMatching

overwatch = initialize_overwatch(__name__)


@VLAS.register_module()
class PI05FlowMatching(PI0FlowMatching):
    """PI0 Flow Matching Model for Vision-Language Alignment.
    Implemented based on https://arxiv.org/abs/2504.16054

    This model is designed to handle vision-language alignment tasks
    using flow matching techniques, leveraging a vision backbone,
    language model backbone, projector, and a VLA head.

    Args:
        state_proj (Dict): Configuration dictionary for the state
            projector.
        action_in_proj (Dict): Configuration dictionary for the action
            input projector.
        action_out_proj (Dict): Configuration dictionary for the action
            output projector.
        action_time_mlp_in (Dict): Configuration dictionary for the action
            time MLP input.
        action_time_mlp_out (Dict): Configuration dictionary for the action
            time MLP output.
        vlm_backbone (str): Identifier for the vision-language model backbone.
        vla_head (str): Identifier for the vision-language alignment head.
        enable_mixed_precision_training (bool): Whether to enable mixed
            precision training.
        freeze_vision_backbone (bool): Whether to freeze the vision backbone.
        freeze_llm_backbone (bool): Whether to freeze the language model
            backbone.
        freeze_projector (bool): Whether to freeze the projector.
        vision_backbone_fp32 (bool): Whether to use FP32 for the vision
            backbone.
        unfreeze_last_layer (bool): Whether to unfreeze the last layer
            of the model.
        ignore_index (int): Index to ignore in loss calculations.
        norm_stats (Dict, optional): Normalization statistics for the model.
        **kwargs: Additional keyword arguments for model configuration.
    """

    def __init__(self, **kwargs):
        self.mini_batches = kwargs.pop('mini_batches', 1)
        rtc_cfg = kwargs.get('rtc_training_config')
        if rtc_cfg and rtc_cfg.get('enabled', False):
            raise ValueError(
                'PI05FlowMatching does not support training-time RTC. '
                'Its architecture cannot inject per-position timesteps '
                'without model modifications. Please disable '
                'rtc_training_config or use test-time RTC (guidance) '
                'instead.')
        super().__init__(**kwargs)

    def predict_action(self,
                       *args,
                       rtc_config=None,
                       prev_actions=None,
                       prefix_len=0,
                       **kwargs):
        if (prev_actions is not None and prefix_len > 0 and rtc_config
                and rtc_config.get('method', 'prefix') == 'prefix'):
            raise ValueError(
                'PI05FlowMatching does not support RTC prefix mode at '
                'inference. Its embed_suffix only accepts a scalar timestep '
                'and cannot handle per-position time injection. '
                "Use method='guidance' for test-time RTC instead.")
        return super().predict_action(
            *args,
            rtc_config=rtc_config,
            prev_actions=prev_actions,
            prefix_len=prefix_len,
            **kwargs)

    def embed_suffix(self, states, noisy_actions, timestep):
        """Embed the suffix tokens for the Pi0 head.

        Args:
            state (torch.Tensor): The state tensor of shape (bsize, state_dim).
            noisy_actions (torch.Tensor): The noisy actions tensor of shape
                (bsize, n_action_steps, action_dim).
            timestep (torch.Tensor): The timestep tensor of shape (bsize,).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple
                containing the embedded suffix tokens, padding masks,
                and attention masks.
        """
        embs = []
        pad_masks = []
        att_masks = []

        # Embed state
        bsize = states.shape[0]
        dtype = states.dtype
        device = states.device
        if self.state_proj is not None:
            state_emb = self.state_proj(states)
            embs.append(state_emb[:, None, :])
            pad_masks.append(
                torch.ones(bsize, 1, dtype=torch.bool, device=device))
            att_masks += [1]

        # Set attention masks so that image and language
        # inputs do not attend to state or actions

        # Embed timestep using sine-cosine positional
        # encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.proj_width,
            min_period=4e-3,
            max_period=4.0,
            device=device)
        time_emb = time_emb.type(dtype=dtype)

        # Fuse timestep + action information using an MLP
        action_emb = self.action_in_proj(noisy_actions)
        time_emb = F.silu(self.time_mlp_in(time_emb))
        time_emb = F.silu(self.time_mlp_out(time_emb))
        # Add to input tokens
        embs.append(action_emb)

        bsize, action_time_dim = action_emb.shape[:2]
        action_time_mask = torch.ones(
            bsize, action_time_dim, dtype=torch.bool, device=device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state
        # inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.n_action_steps - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(
            att_masks, dtype=torch.bool, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, time_emb

    def forward(self,
                images: List[torch.Tensor],
                lang_tokens: torch.Tensor,
                states: torch.Tensor,
                actions: torch.Tensor,
                action_masks: Optional[torch.Tensor] = None,
                img_masks: Optional[torch.Tensor] = None,
                lang_masks: Optional[torch.Tensor] = None,
                past_key_values=None,
                use_cache: Optional[bool] = None,
                fill_kv_cache: Optional[bool] = None,
                noise=None,
                time=None,
                *args,
                **kwarg):
        M = self.mini_batches
        if M <= 1:
            return super().forward(
                images=images,
                lang_tokens=lang_tokens,
                states=states,
                actions=actions,
                action_masks=action_masks,
                img_masks=img_masks,
                lang_masks=lang_masks,
                past_key_values=past_key_values,
                use_cache=use_cache,
                fill_kv_cache=fill_kv_cache,
                noise=noise,
                time=time,
                *args,
                **kwarg)

        B = actions.shape[0]
        B_M = B * M
        device = actions.device

        # === Phase A: Prefix computation (batch=B only) ===
        # embed_prefix runs SigLIP + LLM embed on B images (expensive).
        with record_function("embed_prefix"):
            prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
                images=images,
                lang_tokens=lang_tokens,
                img_masks=img_masks,
                lang_masks=lang_masks)

        # === Phase B: Replicate prefix embeddings (B -> B*M) ===
        # Cheap memory copy — avoids re-running SigLIP M times.
        with record_function("repeat_prefix"):
            prefix_embs = prefix_embs.repeat_interleave(M, dim=0)
            prefix_pad_masks = prefix_pad_masks.repeat_interleave(M, dim=0)
            prefix_att_masks = prefix_att_masks.repeat_interleave(M, dim=0)

        # === Phase C: Build suffix inputs (batch=B*M) ===
        with record_function("embed_suffix"):
            actions_expanded = actions.repeat_interleave(M, dim=0)
            states_expanded = states.repeat_interleave(M, dim=0)

            noise = self.sample_noise(actions_expanded.shape, device)
            time = self.sample_time(B_M, device)

            x_t = (
                time[:, None, None] * noise +
                (1 - time[:, None, None]) * actions_expanded)
            u_t = noise - actions_expanded

            suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = (
                self.embed_suffix(states_expanded, x_t, time))

        # === Phase D: Joint forward pass (batch=B*M) ===
        # Uses the standard _forward_transformer_layers joint path,
        # fully compatible with gradient checkpointing and FSDP.
        with record_function("joint_forward"):
            inputs_embeds = [prefix_embs, suffix_embs]
            pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
            att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

            attention_masks = make_att_2d_masks(pad_masks, att_masks)
            position_ids = torch.cumsum(pad_masks, dim=1) - 1
            att_2d_masks_4d = self._prepare_attention_masks_4d(attention_masks)

            suffix_out, _ = self.forward_model(
                inputs_embeds=inputs_embeds,
                attention_masks=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                fill_kv_cache=fill_kv_cache,
                adarms_cond=[None, adarms_cond],
                time=time)

        # === Phase E: Loss (batch=B*M) ===
        with record_function("compute_loss"):
            suffix_out = suffix_out[:, -self.n_action_steps:]
            suffix_out = suffix_out.to(dtype=torch.float32)
            v_t = self.action_out_proj(suffix_out)

            if self.ori_action_dim is not None:
                v_t = v_t[:, :, :self.ori_action_dim]
                u_t = u_t[:, :, :self.ori_action_dim]

            if action_masks is not None:
                action_masks_expanded = action_masks.repeat_interleave(
                    M, dim=0)
                losses = F.mse_loss(u_t, v_t, reduction='none')
                losses = losses * action_masks_expanded.unsqueeze(-1)
                loss = losses.sum() / (
                    action_masks_expanded.sum() * u_t.shape[-1] + 1e-8)
            else:
                loss = F.mse_loss(u_t, v_t)

        return dict(predictions=v_t, loss=loss)
