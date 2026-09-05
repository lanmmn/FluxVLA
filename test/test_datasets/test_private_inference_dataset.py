# Copyright 2026 Limx Dynamics

from unittest.mock import patch

from fluxvla.datasets.parquet_dataset import PrivateInferenceDataset
from fluxvla.transforms.transform_actions import JointSignTransform


def test_private_inference_dataset_model_path_is_selective():
    captured_tokenizer_cfg = {}

    class FakeTokenizer:

        def __call__(self, *args, **kwargs):
            return {'input_ids': [1]}

    def fake_build_tokenizer(cfg):
        captured_tokenizer_cfg.update(cfg)
        return FakeTokenizer()

    with patch(
            'fluxvla.engines.build_tokenizer_from_cfg',
            side_effect=fake_build_tokenizer):
        dataset = PrivateInferenceDataset(
            norm_stats={'private': {}},
            model_path='/tmp/model-root',
            transforms=[
                dict(type='JointSignTransform', signs=[1, -1]),
                dict(
                    type='ProcessPrompts',
                    tokenizer=dict(type='PretrainedTokenizer'),
                    max_len=2),
            ],
        )

    assert isinstance(dataset.transforms[0], JointSignTransform)
    assert captured_tokenizer_cfg['model_path'] == '/tmp/model-root/tokenizer'
