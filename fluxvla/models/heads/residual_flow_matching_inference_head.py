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
"""Kernel-fused two-step inference head with a trained velocity residual.

This is the deployment counterpart of the training-time
``ResidualProgressiveDistillationFlowMatchingHead``.  The base
``FlowMatchingInferenceHead`` reimplements the denoising loop with fused
Triton kernels and CUDA-graph replay, therefore it never calls the eager
``denoise_step`` where the training residual is applied.  To keep deployment
numerically identical to the distilled student, the bounded continuous
velocity correction is inlined into the fused loop right after the base
velocity is produced and before the Euler update.

The residual MLP is a plain ``nn.Module`` (small, ~3.2e4 params), so it is
loaded by the standard ``load_state_dict`` path together with the base head
weights (checkpoint keys ``velocity_residual.0/2.{weight,bias}``) and simply
called inside the graph-captured region -- no manual weight stacking needed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from fluxvla.engines import HEADS
from fluxvla.ops.atomic_ops import dit_block_cross, dit_block_self, vl_sa_block
from fluxvla.ops.triton.position_embedding import \
    fused_position_embedding_add_inplace
from .flow_matching_inference_head import (FlowMatchingInferenceHead,
                                           _action_encode, _cat_mlp,
                                           _timestep_embedding)


@HEADS.register_module()
class ResidualFlowMatchingInferenceHead(FlowMatchingInferenceHead):
    """FlowMatchingInferenceHead plus a zero-initialized velocity residual.

    Args:
        residual_hidden_dim: Hidden width of the per-frame correction MLP.
            Must match the trained checkpoint (default 256).
        residual_max_abs: Maximum absolute velocity correction in normalized
            action space, bounded with ``tanh`` (default 0.1). Must match the
            trained checkpoint.
        continuous_action_dim: Number of leading continuous action channels the
            residual corrects (default 40). Discrete/padding channels are left
            untouched, exactly as in training.
    """

    def __init__(self,
                 *args,
                 residual_hidden_dim: int = 256,
                 residual_max_abs: float = 0.1,
                 continuous_action_dim: int = 40,
                 **kwargs):
        super().__init__(*args, **kwargs)
        if residual_hidden_dim <= 0:
            raise ValueError("residual_hidden_dim must be positive.")
        if residual_max_abs <= 0:
            raise ValueError("residual_max_abs must be positive.")
        if continuous_action_dim <= 0:
            raise ValueError("continuous_action_dim must be positive.")

        # Supervised (valid) action dimension used to build the residual
        # feature vector, matching the training head. Falls back to action_dim
        # when ori_action_dim is unset.
        self.residual_valid_dim = (self.ori_action_dim
                                   if self.ori_action_dim is not None
                                   else self.action_dim)
        if continuous_action_dim > self.residual_valid_dim:
            raise ValueError(
                "continuous_action_dim cannot exceed the supervised action "
                "dimension.")
        self.residual_continuous_dim = continuous_action_dim
        self.residual_max_abs = float(residual_max_abs)

        residual_input_dim = 2 * self.residual_valid_dim + 1
        # Identical structure/ordering to the training head so the checkpoint
        # keys ``velocity_residual.0`` and ``velocity_residual.2`` line up.
        self.velocity_residual = nn.Sequential(
            nn.Linear(residual_input_dim, residual_hidden_dim),
            nn.SiLU(),
            nn.Linear(residual_hidden_dim, continuous_action_dim),
        )
        # Match the base fused weights (bf16/cuda) so the graph-captured
        # matmuls stay in a single dtype. load_state_dict copies values in
        # place afterwards without changing dtype/device.
        self.velocity_residual.to(dtype=torch.bfloat16, device="cuda")

    def record_run(self):
        """Core denoising computation for CUDA Graph capture.

        Verbatim copy of ``FlowMatchingInferenceHead.record_run`` with the
        trained velocity residual applied to the leading continuous channels
        of ``pred_velocity`` before every Euler update.
        """
        input_features = self.buffers["input_features"]
        embodiment_ids = self.buffers["embodiment_ids"]

        # VLLN (LayerNorm)
        input_features = F.layer_norm(input_features,
                                      [input_features.shape[-1]],
                                      self.weights["vlln_w"],
                                      self.weights["vlln_b"])

        # VL self-attention blocks
        for i in range(self.vl_nl):
            input_features = vl_sa_block(
                input_features,
                self.weights["vl_sa_norm1_w"][i],
                self.weights["vl_sa_norm1_b"][i],
                self.weights["vl_sa_qkv_w"][i],
                self.weights["vl_sa_qkv_b"][i],
                self.weights["vl_sa_attn_out_w"][i],
                self.weights["vl_sa_attn_out_b"][i],
                self.weights["vl_sa_norm3_w"][i],
                self.weights["vl_sa_norm3_b"][i],
                self.weights["vl_sa_ff_up_w_T"][i],
                self.weights["vl_sa_ff_up_b"][i],
                self.weights["vl_sa_ff_down_w"][i],
                self.weights["vl_sa_ff_down_b"][i],
                self.vl_nh,
                self.vl_hd,
                ff_features=self.vl_dim,
                ff_hidden=self.vl_ff)
        self.buffers["input_features"].copy_(input_features)

        # # State encoder
        self.buffers["state_features"].copy_(
            _cat_mlp(self.buffers["states"].unsqueeze(1),
                     self.weights["state_encoder_layer1_W"],
                     self.weights["state_encoder_layer1_b"],
                     self.weights["state_encoder_layer2_W"],
                     self.weights["state_encoder_layer2_b"], embodiment_ids))

        dt = 1.0 / self.num_inference_timesteps
        inner_dim = self.dit_dim
        actions = self.buffers["actions"]
        device = actions.device

        # RTC prefix-conditioning masks (all-zero -> no-op for inference).
        prefill_actions = self.buffers["prefill_actions"]
        prefill_mask = self.buffers["prefill_mask"]
        prefill_inv_mask = 1 - prefill_mask
        prefix_ts_mask = self.buffers["prefix_ts_mask"]
        prefix_ts_inv_mask = 1 - prefix_ts_mask

        vdim = self.residual_valid_dim
        cdim = self.residual_continuous_dim

        for t in range(self.num_inference_timesteps):
            t_cont = t / float(self.num_inference_timesteps)
            t_discretized = int(t_cont * self.num_timestep_buckets)

            timesteps_tensor = torch.full(
                size=(1, ), fill_value=t_discretized, device=device)

            actions = (
                actions * prefill_inv_mask + prefill_actions * prefill_mask)
            if (self.ori_action_dim is not None
                    and self.ori_action_dim < self.action_dim):
                actions[..., self.ori_action_dim:] = 0

            ts_action = (
                t_discretized * prefix_ts_inv_mask +
                self.num_timestep_buckets * prefix_ts_mask)

            # Action encoder
            action_features = _action_encode(
                actions, ts_action, self.weights["action_encoder_W1_W"],
                self.weights["action_encoder_W1_b"],
                self.weights["action_encoder_W2_W"],
                self.weights["action_encoder_W2_b"],
                self.weights["action_encoder_W3_W"],
                self.weights["action_encoder_W3_b"], embodiment_ids,
                self.input_embedding_dim)

            if self.add_positional_embeddings:
                action_features = fused_position_embedding_add_inplace(
                    action_features, self.weights["position_embedding_w"])

            future_tok = self.weights["future_tokens_w"].unsqueeze(0)
            sa_embs = torch.cat(
                (self.buffers["state_features"], future_tok, action_features),
                dim=1)

            temb = _timestep_embedding(timesteps_tensor).to(
                dtype=actions.dtype)
            temb = F.silu(
                F.linear(temb, self.weights["dit_timestep_linear1_w"],
                         self.weights["dit_timestep_linear1_b"]))
            temb = F.linear(temb, self.weights["dit_timestep_linear2_w"],
                            self.weights["dit_timestep_linear2_b"])

            hidden_states = sa_embs
            cross_idx, self_idx = 0, 0
            for i in range(self.dit_nl):
                if i % 2 == 1:
                    hidden_states = dit_block_self(
                        hidden_states,
                        temb,
                        self.weights["dit_norm1_linear_w"][i],
                        self.weights["dit_norm1_linear_b"][i],
                        self.weights["dit_self_qkv_w"][self_idx],
                        self.weights["dit_self_qkv_b"][self_idx],
                        self.weights["dit_attn_out_w"][i],
                        self.weights["dit_attn_out_b"][i],
                        self.weights["dit_ff_up_w_T"][i],
                        self.weights["dit_ff_up_b"][i],
                        self.weights["dit_ff_down_w"][i],
                        self.weights["dit_ff_down_b"][i],
                        self.dit_nh,
                        self.dit_hd,
                        inner_dim,
                        ff_features=inner_dim,
                        ff_hidden=self.dit_ff)
                    self_idx += 1
                else:
                    hidden_states = dit_block_cross(
                        hidden_states,
                        input_features,
                        temb,
                        self.weights["dit_norm1_linear_w"][i],
                        self.weights["dit_norm1_linear_b"][i],
                        self.weights["dit_cross_q_w"][cross_idx],
                        self.weights["dit_cross_q_b"][cross_idx],
                        self.weights["dit_cross_kv_w"][cross_idx],
                        self.weights["dit_cross_kv_b"][cross_idx],
                        self.weights["dit_attn_out_w"][i],
                        self.weights["dit_attn_out_b"][i],
                        self.weights["dit_ff_up_w_T"][i],
                        self.weights["dit_ff_up_b"][i],
                        self.weights["dit_ff_down_w"][i],
                        self.weights["dit_ff_down_b"][i],
                        self.dit_nh,
                        self.dit_hd,
                        inner_dim,
                        ff_features=inner_dim,
                        ff_hidden=self.dit_ff)
                    cross_idx += 1

            shift, scale = F.linear(
                F.silu(temb), self.weights["dit_proj_out_1_w"],
                self.weights["dit_proj_out_1_b"]).chunk(
                    2, dim=1)
            hidden_states = (
                F.layer_norm(hidden_states, [inner_dim]) *
                (1 + scale[:, None]) + shift[:, None])
            model_output = F.linear(hidden_states,
                                    self.weights["dit_proj_out_2_w"],
                                    self.weights["dit_proj_out_2_b"])

            pred = _cat_mlp(model_output,
                            self.weights["action_decoder_layer1_W"],
                            self.weights["action_decoder_layer1_b"],
                            self.weights["action_decoder_layer2_W"],
                            self.weights["action_decoder_layer2_b"],
                            embodiment_ids)

            pred_velocity = pred[:, -self.num_steps:]

            # --- Trained velocity residual (continuous channels only) ---
            # Mirrors ResidualProgressiveDistillationFlowMatchingHead: the
            # per-frame feature is [actions[:vdim], base_velocity[:vdim],
            # t_encoder/num_timestep_buckets]. The tanh-bounded delta is added
            # only to the leading ``cdim`` continuous velocity channels.
            time_feature = (ts_action.to(dtype=pred_velocity.dtype).unsqueeze(-1)
                            / float(self.num_timestep_buckets))
            residual_features = torch.cat(
                (actions[..., :vdim], pred_velocity[..., :vdim], time_feature),
                dim=-1)
            continuous_delta = self.residual_max_abs * torch.tanh(
                self.velocity_residual(residual_features))
            pred_velocity = pred_velocity.clone()
            pred_velocity[..., :cdim] = pred_velocity[..., :cdim] + continuous_delta

            actions = actions + dt * pred_velocity
            if (self.ori_action_dim is not None
                    and self.ori_action_dim < self.action_dim):
                actions[..., self.ori_action_dim:] = 0

        actions = (actions * prefill_inv_mask + prefill_actions * prefill_mask)
        if (self.ori_action_dim is not None
                and self.ori_action_dim < self.action_dim):
            actions[..., self.ori_action_dim:] = 0
        self.buffers["actions"].copy_(actions)
