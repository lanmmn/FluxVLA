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

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from fluxvla.models.heads import VLAJEPAFlowMatchingHead
from fluxvla.models.third_party_models.vjepa2 import \
    VisionTransformerPredictorAC
from fluxvla.models.vlas.vla_jepa import VLAJEPA
from fluxvla.tokenizers import PretrainedTokenizer
from fluxvla.transforms.vla_jepa import PrepareVLAJEPAVideo, VLAJEPAPrompter


class _FakeVideoProcessor:

    def __call__(self, videos, return_tensors):
        assert return_tensors == 'pt'
        return {'pixel_values_videos': videos.float().unsqueeze(0)}


class _FakeActionHead(nn.Module):

    def forward(self,
                input_features,
                states,
                actions,
                action_masks,
                sample_weight=None):
        return {
            'loss': actions.sum() * 0 + 2.0,
            'pred_actions': torch.zeros_like(actions),
        }


class _JointLossHarness(VLAJEPA):

    def __init__(self):
        nn.Module.__init__(self)
        self.vla_head = _FakeActionHead()
        self.world_model_loss_weight = 0.1

    def _forward_vlm(self, images, lang_tokens, img_masks, lang_masks,
                     image_grid_thw):
        return torch.zeros(lang_tokens.shape[0], lang_tokens.shape[1], 4)

    def _extract_condition_tokens(self, last_hidden_state, lang_tokens):
        batch_size = lang_tokens.shape[0]
        return (torch.zeros(batch_size, 2, 4), torch.zeros(batch_size, 3, 4))

    def _world_model_loss(self, world_features, pixel_values_videos,
                          frame_masks):
        return pixel_values_videos.sum() * 0 + 3.0


class TestVLAJEPATransforms(unittest.TestCase):

    def test_prompt_token_counts(self):
        transform = VLAJEPAPrompter(
            num_frames=8,
            tubelet_size=2,
            num_action_tokens_per_timestep=8,
            num_embodied_action_tokens=32,
        )
        result = transform({'task_description': 'open the drawer'})
        prompt = result['prompt']
        self.assertEqual(prompt.count('<|action_0|>'), 8)
        self.assertEqual(prompt.count('<|action_1|>'), 8)
        self.assertEqual(prompt.count('<|action_2|>'), 8)
        self.assertEqual(prompt.count('<|embodied_action|>'), 32)

    def test_video_split_preserves_view_major_order(self):
        transform = PrepareVLAJEPAVideo(
            video_processor_path='unused',
            num_views=2,
            num_frames=8,
            processor=_FakeVideoProcessor(),
        )
        images = []
        for view in range(2):
            for frame in range(8):
                images.append(
                    np.full((3, 4, 4), view * 100 + frame, dtype=np.uint8))
        result = transform({'images': images})
        self.assertEqual(len(result['images']), 2)
        self.assertEqual(int(result['images'][0][0, 0, 0]), 0)
        self.assertEqual(int(result['images'][1][0, 0, 0]), 100)
        self.assertEqual(result['pixel_values_videos'].shape, (2, 8, 3, 4, 4))
        self.assertEqual(
            int(result['pixel_values_videos'][1, 7, 0, 0, 0]), 107)


class TestVLAJEPATokenizer(unittest.TestCase):

    def test_added_tokens_survive_save_and_reload(self):
        source = Path('checkpoints/clip-vit-base-patch32')
        if not source.exists():
            self.skipTest(f'local tokenizer fixture missing: {source}')
        tokens = [
            '<|action_0|>', '<|action_1|>', '<|action_2|>',
            '<|embodied_action|>'
        ]
        tokenizer = PretrainedTokenizer(
            source.as_posix(), additional_special_tokens=tokens)
        original_ids = tokenizer.convert_tokens_to_ids(tokens)
        with tempfile.TemporaryDirectory() as directory:
            tokenizer.save_pretrained(directory)
            reloaded = PretrainedTokenizer(directory)
            self.assertEqual(
                reloaded.convert_tokens_to_ids(tokens), original_ids)


class TestVLAJEPAFlowMatchingHead(unittest.TestCase):

    def setUp(self):
        self.head = VLAJEPAFlowMatchingHead(
            hidden_size=12,
            state_dim=3,
            input_embedding_dim=16,
            action_dim=2,
            action_horizon=3,
            backbone_embedding_dim=16,
            num_inference_timesteps=2,
            num_target_vision_tokens=2,
            diffusion_model_cfg=dict(
                attention_head_dim=8,
                num_attention_heads=2,
                cross_attention_dim=16,
                num_layers=1,
                output_dim=12,
                dropout=0.0,
                final_dropout=False,
                interleave_self_attention=True,
                norm_type='ada_norm',
                positional_embeddings=None,
            ),
        )

    def test_forward_and_predict_shapes(self):
        features = torch.randn(2, 4, 16)
        states = torch.randn(2, 3)
        actions = torch.randn(2, 3, 2)
        output = self.head(
            input_features=features,
            states=states,
            actions=actions,
            action_masks=torch.ones(2, 3),
        )
        self.assertEqual(output['pred_actions'].shape, (2, 3, 2))
        self.assertTrue(torch.isfinite(output['loss']))
        output['loss'].backward()
        self.assertIsNotNone(self.head.action_encoder.layer1.weight.grad)

        prediction = self.head.predict_action(features, states)
        self.assertEqual(prediction.shape, (2, 3, 2))
        self.assertTrue(torch.isfinite(prediction).all())


class TestVLAJEPAPredictor(unittest.TestCase):

    def test_predictor_shape_and_causal_mask(self):
        predictor = VisionTransformerPredictorAC(
            num_frames=4,
            img_size=(8, 8),
            patch_size=4,
            tubelet_size=1,
            embed_dim=6,
            predictor_embed_dim=96,
            depth=1,
            num_heads=3,
            action_embed_dim=10,
            num_add_tokens=2,
            use_rope=True,
        )
        context = torch.randn(2, 12, 6)
        action_tokens = torch.randn(2, 6, 10)
        output = predictor(context, action_tokens)
        self.assertEqual(output.shape, context.shape)
        self.assertEqual(predictor.attn_mask.shape, (24, 24))
        block_size = 6
        self.assertTrue(predictor.attn_mask[:block_size, :block_size].all())
        self.assertFalse(predictor.attn_mask[:block_size, block_size:].any())


class TestVLAJEPAComposition(unittest.TestCase):

    def test_joint_loss_weight(self):
        model = _JointLossHarness()
        output = model(
            lang_tokens=torch.zeros(1, 4, dtype=torch.long),
            lang_masks=torch.ones(1, 4, dtype=torch.bool),
            images=torch.zeros(1, 3, 2, 2),
            img_masks=torch.ones(1, 1, dtype=torch.bool),
            states=torch.zeros(1, 3),
            actions=torch.zeros(1, 3, 2),
            action_masks=torch.ones(1, 3),
            pixel_values_videos=torch.zeros(1, 2, 8, 3, 2, 2),
            frame_masks=torch.ones(1, 8),
        )
        self.assertAlmostEqual(output['loss'].item(), 2.3, places=6)
        self.assertAlmostEqual(output['action_loss'].item(), 2.0, places=6)
        self.assertAlmostEqual(output['wm_loss'].item(), 3.0, places=6)


if __name__ == '__main__':
    unittest.main()
