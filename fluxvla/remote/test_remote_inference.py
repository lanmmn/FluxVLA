"""End-to-end integration test for remote inference with server-side preprocessing.

Usage::

    python -m fluxvla.remote.test_remote_inference
"""
import sys
import threading
import time

import numpy as np
import torch

sys.path.insert(0, '.')

from fluxvla.remote.remote_vla import RemoteVLAZmq
from fluxvla.remote.server_client import ObsSerializer
from fluxvla.remote.vla_server import VLAInferPipeline, create_vla_server

# ========================= Mocks =========================


class MockVLA(torch.nn.Module):

    def __init__(self, action_dim=14, action_chunk=50):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
        self._action_dim = action_dim
        self._action_chunk = action_chunk

    def predict_action(self, **batch):
        bs = batch['images'].shape[0]
        return torch.randn(bs, self._action_chunk, self._action_dim).cuda()


class MockDataset:
    """Simulates PrivateInferenceDataset(obs) -> batch dict."""

    def __call__(self, obs):
        imgs = []
        for v in obs.values():
            if isinstance(v,
                          np.ndarray) and v.ndim == 3 and v.dtype == np.uint8:
                img = v[:224, :224, :].astype(np.float32) / 255.0
                imgs.append(torch.from_numpy(img).permute(2, 0, 1))
        if not imgs:
            imgs = [torch.zeros(3, 224, 224)]
        images = torch.stack(imgs).unsqueeze(0).cuda()
        qpos = obs.get('qpos', np.zeros(14, dtype=np.float32))
        return {
            'images': images,
            'img_masks': torch.ones(1, len(imgs), dtype=torch.bool).cuda(),
            'lang_tokens': torch.randint(0, 1000, (1, 20)).cuda(),
            'lang_masks': torch.ones(1, 20, dtype=torch.bool).cuda(),
            'states': torch.from_numpy(qpos).unsqueeze(0).float().cuda(),
        }


class MockLiberoDataset:
    """Simulates LiberoParquetEvalDataset(obs) -> (batch, replay_img)."""

    def __call__(self, obs):
        batch = MockDataset()(obs)
        replay_img = next(
            (v for v in obs.values()
             if isinstance(v, np.ndarray) and v.ndim == 3),
            None,
        )
        return batch, replay_img


# ========================= Tests =========================


def test_obs_serializer():
    obs = {
        'cam_high': np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
        'cam_left_wrist':
        np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
        'qpos': np.random.randn(14).astype(np.float32),
        'task_description': 'pick up the box',
    }
    payload = ObsSerializer.to_bytes(obs)
    obs2 = ObsSerializer.from_bytes(payload)

    assert set(obs2.keys()) == set(obs.keys())
    for k in ['cam_high', 'cam_left_wrist']:
        assert obs2[k].shape == obs[k].shape
        assert obs2[k].dtype == np.uint8
    np.testing.assert_array_almost_equal(obs2['qpos'], obs['qpos'])
    assert obs2['task_description'] == obs['task_description']

    raw_size = sum(v.nbytes for v in obs.values() if isinstance(v, np.ndarray))
    print(f'  Payload: {len(payload)/1024:.0f} KB '
          f'(raw numpy: {raw_size/1024:.0f} KB, '
          f'ratio: {len(payload)/raw_size:.2f}x)')
    print('  PASSED')


def test_end_to_end(port, dataset_cls, obs, action_shape, label,
                    serializer='msgpack'):
    model = MockVLA(
        action_dim=action_shape[-1],
        action_chunk=action_shape[-2],
    ).cuda().eval()
    pipeline = VLAInferPipeline(
        model, device='cuda:0', mixed_precision_dtype=torch.float32)
    server = create_vla_server(
        pipeline, host='127.0.0.1', port=port, dataset=dataset_cls())
    t = threading.Thread(target=server.run, daemon=True)
    t.start()

    client = RemoteVLAZmq(
        host='127.0.0.1', port=port, timeout_s=10.0, device='cuda:0',
        serializer=serializer)
    for _ in range(50):
        if client.ping():
            break
        time.sleep(0.1)
    else:
        raise RuntimeError('Server failed to start within 5s')
    try:
        actions = client.predict_action(**obs)
        assert actions.shape == torch.Size(action_shape), (
            f'Shape mismatch: {actions.shape} != {action_shape}')
        assert actions.device.type == 'cuda'

        stats = client._last_profile
        print(f'  {label} [{serializer}]: shape={actions.shape}, '
              f"payload={stats['payload_kb']:.0f}KB, "
              f"total={stats['total_ms']:.1f}ms")

        for _ in range(5):
            actions = client.predict_action(**obs)
        print('  5 additional calls: OK')
        print('  PASSED')
    finally:
        client.close()
        server.close()
        time.sleep(0.6)


def main():
    print('=== Test 1: ObsSerializer roundtrip ===')
    test_obs_serializer()

    print('\n=== Test 2: Aloha-style raw obs ===')
    test_end_to_end(
        port=25551,
        dataset_cls=MockDataset,
        obs={
            'cam_high':
            np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
            'cam_left_wrist':
            np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
            'cam_right_wrist':
            np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
            'qpos':
            np.random.randn(14).astype(np.float32),
            'task_description':
            'pick up the brown bird toy',
            'unnorm_key':
            'private',
        },
        action_shape=(1, 50, 14),
        label='Aloha',
    )

    print('\n=== Test 3: UR3-style raw obs ===')
    test_end_to_end(
        port=25552,
        dataset_cls=MockDataset,
        obs={
            'cam_high':
            np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
            'cam_left_wrist':
            np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
            'qpos':
            np.random.randn(7).astype(np.float32),
            'task_description':
            'grasp the bottle',
            'unnorm_key':
            'private',
        },
        action_shape=(1, 50, 14),
        label='UR3',
    )

    print('\n=== Test 4: Libero-style raw obs (tuple dataset) ===')
    test_end_to_end(
        port=25553,
        dataset_cls=MockLiberoDataset,
        obs={
            'agentview_image':
            np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
            'robot0_eye_in_hand_image':
            np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
            'robot0_eef_pos':
            np.array([0.1, 0.2, 0.3], dtype=np.float32),
            'robot0_eef_quat':
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            'robot0_gripper_qpos':
            np.array([0.04], dtype=np.float32),
            'task_description':
            'pick up the red cup',
            'unnorm_key':
            'libero_10',
        },
        action_shape=(1, 50, 14),
        label='Libero',
    )

    print('\n=== Test 5: Aloha-style raw obs (protobuf) ===')
    test_end_to_end(
        port=25554,
        dataset_cls=MockDataset,
        obs={
            'cam_high':
            np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
            'cam_left_wrist':
            np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
            'cam_right_wrist':
            np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
            'qpos':
            np.random.randn(14).astype(np.float32),
            'task_description':
            'pick up the brown bird toy',
            'unnorm_key':
            'private',
        },
        action_shape=(1, 50, 14),
        label='Aloha',
        serializer='protobuf',
    )

    print('\n=== Test 6: Libero-style raw obs (protobuf) ===')
    test_end_to_end(
        port=25555,
        dataset_cls=MockLiberoDataset,
        obs={
            'agentview_image':
            np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
            'robot0_eye_in_hand_image':
            np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
            'robot0_eef_pos':
            np.array([0.1, 0.2, 0.3], dtype=np.float32),
            'robot0_eef_quat':
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            'robot0_gripper_qpos':
            np.array([0.04], dtype=np.float32),
            'task_description':
            'pick up the red cup',
            'unnorm_key':
            'libero_10',
        },
        action_shape=(1, 50, 14),
        label='Libero',
        serializer='protobuf',
    )

    print('\n=== ALL TESTS PASSED ===')


if __name__ == '__main__':
    main()
