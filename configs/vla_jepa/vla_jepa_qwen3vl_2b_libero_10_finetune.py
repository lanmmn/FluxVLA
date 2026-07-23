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

import copy

_qwen3vl_base = 'Qwen/Qwen3-VL-2B-Instruct'
_vjepa2_base = 'facebook/vjepa2-vitl-fpc64-256'
_libero_root = 'datasets/libero_10_no_noops_lerobotv2.1'
_statistic_name = 'libero_10_no_noops'
_special_tokens = [
    '<|action_0|>',
    '<|action_1|>',
    '<|action_2|>',
    '<|embodied_action|>',
]

_tokenizer = dict(
    type='PretrainedTokenizer',
    model_path=_qwen3vl_base,
    padding_side='right',
    additional_special_tokens=_special_tokens,
)

model = dict(
    type='VLAJEPA',
    pretrained_name_or_path=None,
    strict_mapping=False,
    tokenizer=copy.deepcopy(_tokenizer),
    num_views=2,
    num_frames=8,
    num_action_tokens_per_timestep=8,
    num_embodied_action_tokens=32,
    world_model_loss_weight=0.1,
    vj_encoder_path=_vjepa2_base,
    vj_predictor_cfg=dict(
        patch_size=16,
        predictor_embed_dim=1024,
        depth=12,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        use_rope=True,
        use_activation_checkpointing=True,
    ),
    vlm_backbone=dict(
        type='Qwen3VL',
        vlm_backbone_id='qwen3_2b_vl_pt',
        vlm_path=_qwen3vl_base,
        vlm_config=None,
        use_projection=False,
        attn_implementation='sdpa',
    ),
    vla_head=dict(
        type='VLAJEPAFlowMatchingHead',
        state_dim=8,
        hidden_size=1024,
        input_embedding_dim=768,
        backbone_embedding_dim=2048,
        action_dim=7,
        action_horizon=7,
        num_inference_timesteps=4,
        num_target_vision_tokens=32,
        add_positional_embeddings=True,
        max_seq_len=1024,
        num_timestep_buckets=1000,
        noise_s=0.999,
        noise_beta_alpha=1.5,
        noise_beta_beta=1.0,
        diffusion_model_cfg=dict(
            attention_head_dim=64,
            num_attention_heads=12,
            cross_attention_dim=2048,
            num_layers=16,
            output_dim=1024,
            dropout=0.2,
            final_dropout=True,
            interleave_self_attention=True,
            norm_type='ada_norm',
            positional_embeddings=None,
        ),
    ),
    freeze_vlm_backbone=False,
    freeze_projector=False,
)

# Keep the complete module graph so training checkpoints load without ignored
# world-model keys. VLAJEPA.predict_action does not execute the world model.
inference_model = copy.deepcopy(model)

train_dataloader = dict(
    per_device_batch_size=4,
    per_device_num_workers=4,
    dataset=dict(
        type='DistributedRepeatingDataset',
        name_mappings={
            'observation.state': ['proprio'],
            'action': ['action'],
        },
        statistic_keys=['observation.state', 'timestamp', 'action'],
        statistic_name=_statistic_name,
        datasets=dict(
            type='ParquetDataset',
            data_root_path=_libero_root,
            transforms=[
                dict(
                    type='ProcessParquetInputs',
                    parquet_keys=[
                        'observation.state',
                        'timestamp',
                        'actions',
                        'info',
                        'stats',
                        'action_masks',
                    ],
                    video_keys=[
                        'observation.images.image',
                        'observation.images.wrist_image',
                    ],
                    name_mappings={
                        'observation.state': ['states'],
                        'actions': ['actions'],
                    },
                ),
                dict(
                    type='PrepareVLAJEPAVideo',
                    video_processor_path=_vjepa2_base,
                    num_views=2,
                    num_frames=8,
                ),
                dict(
                    type='VLAJEPAPrompter',
                    num_frames=8,
                    tubelet_size=2,
                    num_action_tokens_per_timestep=8,
                    num_embodied_action_tokens=32,
                ),
                dict(
                    type='ProcessPrompts',
                    tokenizer=copy.deepcopy(_tokenizer),
                    max_len=128,
                ),
                dict(type='ResizeImages', height=224, width=224),
                dict(
                    type='QWen2VLImageTransform',
                    min_pixels=56 * 56,
                    max_pixels=28 * 28 * 1280,
                    patch_size=16,
                    temporal_patch_size=2,
                    merge_size=2,
                    image_mean=[0.48145466, 0.4578275, 0.40821073],
                    image_std=[0.26862954, 0.26130258, 0.27577711],
                ),
                dict(
                    type='NormalizeStatesAndActions',
                    action_dim=7,
                    state_dim=8,
                    state_key='proprio',
                    action_key='action',
                    norm_type='mean_std',
                ),
            ],
            action_window_size=7,
            action_key='action',
            use_delta=False,
            statistic_name=_statistic_name,
            window_start_idx=0,
            frame_window_size=8,
            frame_sample_stride=1,
        ),
    ),
)

runner = dict(
    type='FSDPTrainRunner',
    # max_steps=30000,
    optimizer=dict(
        type='AdamW',
        lr=1e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=1e-8,
        paramwise_learning_rate={
            'vlm_backbone': 1e-5,
            'vla_head': 1e-4,
            'world_predictor': 1e-4,
        },
    ),
    max_grad_norm=1.0,
    sampler=None,
    tokenizer=copy.deepcopy(_tokenizer),
    collator=dict(
        type='DictCollator',
        keys=[
            'states',
            'timestamp',
            'images',
            'img_masks',
            'lang_tokens',
            'lang_masks',
            'actions',
            'action_masks',
            'image_grid_thw',
            'pixel_values_videos',
            'frame_masks',
        ],
        meta_keys=['task_description', 'prompt', 'info', 'stats'],
    ),
    metric=dict(
        type='VLAMetric',
        active_trackers=('jsonl', 'wandb'),
        run_dir='work_dirs',
        window_size=1,
    ),
    lr_scheduler=dict(
        type='linear-warmup+cosine-decay',
        warmup_ratio=1 / 6,
    ),
    grad_accumulation_steps=1,
    save_iter_interval=10000,
    max_keep_ckpts=3,
    enable_gradient_checkpointing=True,
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16',
    sharding_strategy='full-shard',
    change_key_name=False,
)

eval = dict(
    type='LiberoEvalRunner',
    task_suite_name='libero_10',
    model_family='vla_jepa',
    norm_stats_key=_statistic_name,
    eval_chunk_size=7,
    resize_size=224,
    num_trials_per_task=50,
    num_steps_wait=10,
    seed=7,
    dataset=dict(
        type='LiberoParquetEvalDataset',
        transforms=[
            dict(
                type='ProcessLiberoEvalInputs',
                img_keys=[
                    'agentview_image',
                    'robot0_eye_in_hand_image',
                ],
            ),
            dict(type='ConvertPILImageToNumpyArray'),
            dict(
                type='QWen2VLImageTransform',
                min_pixels=56 * 56,
                max_pixels=28 * 28 * 1280,
                patch_size=16,
                temporal_patch_size=2,
                merge_size=2,
                image_mean=[0.48145466, 0.4578275, 0.40821073],
                image_std=[0.26862954, 0.26130258, 0.27577711],
                img_key='pixel_values',
                to_tensor=True,
            ),
            dict(
                type='VLAJEPAPrompter',
                num_frames=8,
                tubelet_size=2,
                num_action_tokens_per_timestep=8,
                num_embodied_action_tokens=32,
            ),
            dict(
                type='ProcessPrompts',
                tokenizer=copy.deepcopy(_tokenizer),
                max_len=128,
            ),
            dict(
                type='LiberoProprioFromInputs',
                state_dim=8,
                norm_type='mean_std',
                pos_key='robot0_eef_pos',
                quat_key='robot0_eef_quat',
                gripper_key='robot0_gripper_qpos',
                out_key='states',
            ),
        ],
    ),
    denormalize_action=dict(
        type='DenormalizeLiberoAction',
        norm_type='mean_std',
        action_dim=7,
    ),
)
