#!/usr/bin/env python

from __future__ import annotations

import gc
import os

import numpy as np
import torch

from fluxvla.engines import build_vla_from_cfg, set_seed_everywhere

PI05_CKPT_PATH = './checkpoints/pi05_base/model.safetensors'
PI05_DATA_DIR = 'test/data/models/vlas/pi05'


def build_cfg():
    return dict(
        type='PI05FlowMatching',
        llm_backbone=dict(
            type='ConditionGemmaModel',
            adarms_cond_dim=None,
            attention_bias=False,
            attention_dropout=0.0,
            bos_token_id=2,
            eos_token_id=1,
            head_dim=256,
            hidden_act='gelu_pytorch_tanh',
            hidden_activation='gelu_pytorch_tanh',
            hidden_size=2048,
            initializer_range=0.02,
            intermediate_size=16384,
            max_position_embeddings=8192,
            model_type='gemma',
            num_attention_heads=8,
            num_hidden_layers=18,
            num_key_value_heads=1,
            rms_norm_eps=1e-06,
            rope_theta=10000.0,
            torch_dtype='float32',
            use_cache=True,
            vocab_size=257152,
        ),
        vision_backbone=dict(
            type='SigLIPViTBackbone',
            vision_backbone_id='siglip_224',
            vision_config=dict(
                attention_dropout=0.0,
                hidden_act='gelu_pytorch_tanh',
                hidden_size=1152,
                image_size=224,
                intermediate_size=4304,
                layer_norm_eps=1e-06,
                model_type='siglip_vision_model',
                num_attention_heads=16,
                num_channels=3,
                num_hidden_layers=27,
                patch_size=14,
                projection_dim=2048,
                projector_hidden_act='gelu_fast',
                torch_dtype='float32',
                vision_use_head=False,
            ),
        ),
        projector=dict(
            type='LinearProjector',
            in_dim=1152,
            out_dim=2048,
        ),
        proj_width=1024,
        n_action_steps=10,
        action_in_proj=dict(type='LinearProjector', in_dim=32, out_dim=1024),
        action_out_proj=dict(type='LinearProjector', in_dim=1024, out_dim=32),
        time_mlp_in=dict(type='LinearProjector', in_dim=1024, out_dim=1024),
        time_mlp_out=dict(type='LinearProjector', in_dim=1024, out_dim=1024),
        max_action_dim=32,
        llm_expert=dict(
            type='ConditionGemmaModel',
            attention_bias=False,
            adarms_cond_dim=1024,
            attention_dropout=0.0,
            bos_token_id=2,
            eos_token_id=1,
            head_dim=256,
            hidden_act='gelu_pytorch_tanh',
            hidden_activation='gelu_pytorch_tanh',
            hidden_size=1024,
            initializer_range=0.02,
            intermediate_size=4096,
            max_position_embeddings=8192,
            model_type='gemma',
            num_attention_heads=8,
            num_hidden_layers=18,
            num_key_value_heads=1,
            pad_token_id=0,
            rms_norm_eps=1e-06,
            rope_theta=10000.0,
            torch_dtype='float32',
            transformers_version='4.48.1',
            use_adarms=True,
            use_cache=True,
            vocab_size=257152),
        freeze_llm_backbone=False,
        freeze_vision_backbone=False,
        pretrained_name_or_path=PI05_CKPT_PATH,
        name_mapping={
            'llm_backbone':
            'paligemma_with_expert.paligemma.model.language_model',
            'llm_backbone.embed_tokens':
            'paligemma_with_expert.paligemma.lm_head',
            'vision_backbone.vision':
            'paligemma_with_expert.paligemma.model.vision_tower',
            'projector.projector':
            'paligemma_with_expert.paligemma.model.multi_modal_projector.linear',
            'llm_expert': 'paligemma_with_expert.gemma_expert.model',
            'time_mlp_in.projector': 'time_mlp_in',
            'time_mlp_out.projector': 'time_mlp_out',
            'action_in_proj.projector': 'action_in_proj',
            'action_out_proj.projector': 'action_out_proj',
        },
        params_to_change_dtype=[
            'llm_expert.llm.model.layers',
            'vlm_backbone.vlm.model.language_model.layers',
            'vlm_backbone.vlm.model.vision_tower',
            'vlm_backbone.vlm.model.multi_modal_projector',
        ])


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this smoke test.')
    if not os.path.exists(PI05_CKPT_PATH):
        raise FileNotFoundError(f'Checkpoint not found: {PI05_CKPT_PATH}')

    gc.collect()
    torch.cuda.empty_cache()
    set_seed_everywhere(0)
    vla = build_vla_from_cfg(build_cfg()).cuda()
    vla.from_pretrained()
    vla.eval()
    vla.rtc_training_config = dict(
        enabled=True,
        max_delay=3,
        shared_observation=True,
        distribution='uniform',
        shared_observation_loss_weighting='uniform',
        temperature=1.0,
    )

    images = torch.from_numpy(
        np.load(os.path.join(PI05_DATA_DIR, 'images.npy'),
                allow_pickle=True)).cuda()
    img_masks = torch.from_numpy(
        np.load(os.path.join(PI05_DATA_DIR, 'img_masks.npy'),
                allow_pickle=True)).cuda()
    lang_tokens = torch.from_numpy(
        np.load(os.path.join(PI05_DATA_DIR, 'lang_tokens.npy'),
                allow_pickle=True)).cuda()
    lang_masks = torch.from_numpy(
        np.load(os.path.join(PI05_DATA_DIR, 'lang_masks.npy'),
                allow_pickle=True)).cuda()
    states = torch.from_numpy(
        np.load(os.path.join(PI05_DATA_DIR, 'suffix_state.npy'),
                allow_pickle=True)).cuda()
    x_t = torch.from_numpy(
        np.load(os.path.join(PI05_DATA_DIR, 'suffix_x_t.npy'),
                allow_pickle=True)).cuda()

    actions = x_t.clone()
    action_masks = torch.ones(actions.shape[0], actions.shape[1],
                              device=actions.device)

    with torch.no_grad():
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=True):
            output = vla.forward(
                images=images,
                img_masks=img_masks,
                lang_tokens=lang_tokens,
                lang_masks=lang_masks,
                states=states,
                actions=actions,
                action_masks=action_masks,
            )

    assert 'shared_predictions' in output
    assert 'shared_rtc_branch_count' in output
    assert output['shared_rtc_branch_count'] == 3
    assert output['shared_predictions'].shape[0] == actions.shape[0]
    assert output['shared_predictions'].shape[1] == 3
    assert output['shared_predictions'].shape[2] == actions.shape[1]
    assert torch.isfinite(output['loss']).item()

    print('pi05_rtc_shared_obs_smoke: PASS')


if __name__ == '__main__':
    main()
