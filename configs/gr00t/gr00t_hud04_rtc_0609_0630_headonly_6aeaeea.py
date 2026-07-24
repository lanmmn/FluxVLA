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

# Derived from commit 6aeaeea739d256e49bff48de9f6062160423c0fe.
# Head-camera-only variant: the wrist camera is not loaded or tokenized.

# WBT action layout (43 dims, padded to 64 for the model):
#   [0:31]   joint_cmd (31)            — continuous
#   [31:34]  base_pos x/y/z (3)        — continuous
#   [34:40]  base_rot6d (6)            — continuous
#   [40:42]  hand_closed left/right    — discrete {0, 1}
#   [42]     done                      — discrete {0, 1}
# Single source of truth for everything that depends on this layout —
# norm transforms (train + inference), denorm, done detection.
WBT_DISCRETE_ACTION_DIMS = [40, 41, 42]
WBT_DISCRETE_STATE_DIMS = [31, 32]       # state has only hand_closed (no done)
WBT_DISCRETE_NORM_TYPE = 'min_max'       # binary {0,1} -> {-1,+1}
WBT_CONTINUOUS_NORM_TYPE = 'quantile'    # q01/q99 -> resists spike inflation
WBT_DONE_DIM_INDEX = 42
WBT_NORM_KW = dict(
    norm_type=WBT_CONTINUOUS_NORM_TYPE,
    discrete_action_dims=WBT_DISCRETE_ACTION_DIMS,
    discrete_state_dims=WBT_DISCRETE_STATE_DIMS,
    discrete_norm_type=WBT_DISCRETE_NORM_TYPE,
)
WBT_DENORM_KW = dict(
    norm_type=WBT_CONTINUOUS_NORM_TYPE,
    discrete_action_dims=WBT_DISCRETE_ACTION_DIMS,
    discrete_norm_type=WBT_DISCRETE_NORM_TYPE,
)

model = dict(
    type='LlavaVLA',
    pretrained_name_or_path=  # noqa: E251
    './checkpoints/GR00T-N1.5-3B',
    vlm_backbone=dict(
        type='EagleBackbone',
        vlm_path=  # noqa: E251
        'fluxvla/models/third_party_models/eagle2_hg_model',
        vlm_config=dict(max_input_seq_len=900)),
    vla_head=dict(
        type='FlowMatchingHead',
        state_dim=64,
        hidden_size=1024,
        input_embedding_dim=1536,
        num_layers=1,
        num_heads=4,
        num_inference_timesteps=4,
        num_steps=32,
        traj_length=10,  # no use param
        action_dim=64,  # from 32 expand to 64
        ori_action_dim=43,
        done_loss_weight=1.0,
        rtc_training_config=dict(
            enabled=True,
            max_delay=7,
            distribution='exponential',  # 'exponential'（推荐）或 'uniform'
        )),
    freeze_vlm_backbone=False,
    name_mapping={
        'vlm_backbone.vlm': 'backbone.eagle_model',
        'vla_head': 'action_head'
    },
    freeze_projector=False)

inference_model = dict(
    type='LlavaVLA',
    pretrained_name_or_path=  # noqa: E251
    './checkpoints/GR00T-N1.5-3B',
    vlm_backbone=dict(
        type='EagleBackbone',
        vlm_path=  # noqa: E251
        'fluxvla/models/third_party_models/eagle2_hg_model',
        vlm_config=dict(max_input_seq_len=900)),
    vla_head=dict(
        type='FlowMatchingHead',
        state_dim=64,
        hidden_size=1024,
        input_embedding_dim=1536,
        num_layers=1,
        num_heads=4,
        num_inference_timesteps=4,
        num_steps=32,
        traj_length=10,  # no use param
        action_dim=64,  # from 32 expand to 64
        ori_action_dim=43,
        done_loss_weight=1.0,
        rtc_training_config=dict(
            enabled=True,
            max_delay=7,
            distribution='exponential',  # 'exponential'（推荐）或 'uniform'
        )),
)

train_dataloader = dict(
    per_device_batch_size=8,
    per_device_num_workers=8,
    dataset=dict(
        type='DistributedRepeatingDataset',
        name_mappings={
            'observation.state': ['proprio'],
            'action': ['action']
        },
        statistic_keys=['observation.state', 'timestamp', 'action'],
        # statistic_name='hud04_water',
        datasets=[
            dict(
                type='ParquetDataset',
                data_root_path=  # noqa: E251
                '/mnt/workspace/boris/processed/wbt_done_dim_0609_0630',
                transforms=[
                    dict(
                        type='ProcessParquetInputs',
                        embodiment_id=0,
                        parquet_keys=[
                            'observation.state', 'timestamp', 'actions',
                            'info', 'stats', 'action_masks'
                        ],
                        video_keys=[
                            'observation.images.head',
                        ],
                        name_mappings={'observation.state': ['states']}),
                    dict(type='ParquetPrompter'),
                    dict(
                        type='ProcessPromptsWithImage',
                        max_len=900,
                        num_images=1,
                        tokenizer=dict(
                            type='PretrainedTokenizer',
                            model_path=  # noqa: E251
                            'fluxvla/models/third_party_models/eagle2_hg_model',  # noqa: E501
                            # special_tokens={'pad_token': '<PAD>'}
                        )),
                    dict(type='ResizeImages', height=224, width=224),
                    dict(
                        type='NormalizeImages',
                        means=[[123.515625, 116.04492188, 103.59375],
                               [123.515625, 116.04492188, 103.59375],
                               [123.515625, 116.04492188, 103.59375]],
                        stds=[[58.27148438, 57.02636719, 57.27539062],
                              [58.27148438, 57.02636719, 57.27539062],
                              [58.27148438, 57.02636719, 57.27539062]],
                    ),
                    dict(
                        type='NormalizeStatesAndActions',
                        state_dim=64,
                        action_dim=64,
                        state_key='proprio',
                        action_key='action',
                        **WBT_NORM_KW)
                ],
                action_window_size=32,
                action_key='action'),
        ]))

runner = dict(
    type='FSDPTrainRunner',
    max_epochs=30,
    save_epoch_interval=2,
    max_keep_ckpts=15,
    learning_rate=2e-5,
    weight_decay=0.0,
    max_grad_norm=1.0,
    sampler=None,
    tokenizer=dict(
        type='PretrainedTokenizer',
        model_path=  # noqa: E251
        'fluxvla/models/third_party_models/eagle2_hg_model',
        # special_tokens={'pad_token': '<PAD>'}
    ),
    collator=dict(
        type='DictCollator',
        keys=[
            'states', 'timestamp', 'images', 'img_masks', 'lang_tokens',
            'lang_masks', 'actions', 'action_masks', 'embodiment_ids'
        ],
        meta_keys=['task_description', 'prompt', 'info', 'stats']),
    metric=dict(
        type='VLAMetric',
        active_trackers=('jsonl', 'wandb'),
        run_dir='work_dirs',
        grad_accumulation_steps=1,
        window_size=1),
    lr_scheduler_type='constant',
    warmup_ratio=0.0,
    enable_gradient_checkpointing=False,
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16',
    change_key_name=False)

inference = dict(
    type='Teleop02WbtRTCInferenceRunner',
    seed=7,
    async_execution=True,
    async_remaining_actions_threshold=6,
    execute_horizon=16,
    target_hz=50,
    interpolation_method='cubic',
    rtc_config=dict(
        enabled=True,
        method='prefix',
        prefix_len=4,
    ),
    task_descriptions={
        '0':
        'Navigate forward to the first box. Grasp the plush toy from the first box. Turn around and move to the chair. Release the plush toy onto the chair.',  # noqa: E501
        '1':
        'Turn around and move back to the first box. Bend down, grasp the first box with both hands, and lift it. Carry the first box to the second box located in front of you. Place the first box on top of the second box.',  # noqa: E501
        '2':
        'Turn right and walk to the table. Pick up the basket from the floor with the right hand. Pick up the plush toys on the table with the left hand, one by one, and place them into the basket. After all plush toys are in the basket, place the basket on the floor.',  # noqa: E501
        '3':
        'Walk behind the sofa. Grasp the clothes with the left hand and drape them over the right forearm. Walk to the clothes rack and grasp the clothes with the left hand. Walk to the laundry basket and put the clothes into it one by one.',  # noqa: E501
        '4':
        'Turn right and walk to the low table. Bend down and pick up the plastic cup on the table with the right hand, and pick up the paper ball with the left hand. Turn left and walk to the trash can. Drop the trash into the trash can one by one.',  # noqa: E501
        '5':
        'Walk to the chair on the left. Rotate the chair with the left hand. Push the chair under the table with both hands.',  # noqa: E501
        '6':
        'Turn left and walk to the bookshelf. Grasp a wallet from the bookshelf. Turn around and walk to the person behind you. Hand the wallet to the person.',  # noqa: E501
        '7':
        'Turn right and walk back to the starting position.',  # noqa: E501
    },
    done_dim_index=WBT_DONE_DIM_INDEX,
    done_threshold=0.7,
    done_window=8,
    done_advance_cooldown=25,
    done_subtask_order=['3', '0', '1', '2', '5', '4', '6', '7'],
    stop_on_final_done=False,
    interactive=False,
    camera_names=['head'],
    mixed_precision_dtype='bf16',
    dataset=dict(
        type='PrivateInferenceDataset',
        embodiment_id=0,
        img_keys=['head'],
        transforms=[
            dict(
                type='ProcessPromptsWithImage',
                max_len=900,
                num_images=1,
                tokenizer=dict(type='PretrainedTokenizer'
                               # special_tokens={'pad_token': '<PAD>'}
                               )),
            dict(type='ResizeImages', height=224, width=224),
            dict(
                type='NormalizeImages',
                means=[[123.515625, 116.04492188, 103.59375]],
                stds=[[58.27148438, 57.02636719, 57.27539062]],
            ),
            dict(
                type='NormalizeStatesAndActions',
                state_dim=64,
                state_key='proprio',
                action_key='action',
                **WBT_NORM_KW)
        ]),
    denormalize_action=dict(
        type='DenormalizePrivateAction',
        action_dim=43,
        **WBT_DENORM_KW),
    operator=dict(
        type='Teleop02WbtOperator',
        head_rgb_topic='/head/color/image_raw/compressed',
        use_left_wrist_camera=False,
        joint_state_topic='/joint/state',
        finger_state_topic='/brainco1/hand/state',
        finger_cmd_topic='/brainco1/hand/cmd',
        teleop_wbt_topic='/teleop_cmd_WBT',
        cmd_vel_topic='/sdk_cmd_vel_vla',
    ))
