"""Tests for TensorBatchInfer RPC and RemoteVLA client proxy."""
from __future__ import annotations

import io
import time
import unittest
from concurrent import futures

import numpy as np
import torch
import grpc

import vla_service_pb2
import vla_service_pb2_grpc
from grpc_server import VLAInferPipeline, VLAServiceServicer, serve

# ---------------------------------------------------------------------------
# Mock VLA model that mimics predict_action interface
# ---------------------------------------------------------------------------

class _MockVLAModel:
    """Minimal VLA stand-in: echoes input shapes back as dummy actions."""

    def __init__(self, n_action_steps=10, action_dim=7):
        self._n = n_action_steps
        self._d = action_dim
        self.norm_stats = {}

    def eval(self):
        return self

    def to(self, *a, **kw):
        return self

    def predict_action(self, images, lang_tokens, states,
                       img_masks=None, lang_masks=None, **kwargs):
        bsize = images.shape[0]
        return torch.randn(bsize, self._n, self._d)


# ---------------------------------------------------------------------------
# Helper: build a test batch that looks like LiberoParquetEvalDataset output
# ---------------------------------------------------------------------------

N_ACTION_STEPS = 10
ACTION_DIM = 7
TEST_PORT = 50098


def _make_batch(device="cpu"):
    return dict(
        images=torch.randn(1, 6, 224, 224, device=device),
        img_masks=torch.tensor([[True, True]], device=device),
        lang_tokens=torch.randint(0, 1000, (1, 48), device=device),
        lang_masks=torch.ones(1, 48, dtype=torch.bool, device=device),
        states=torch.randn(1, 32, device=device).bfloat16(),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTensorSerialization(unittest.TestCase):
    """Verify batch dict survives torch.save/load round-trip."""

    def test_round_trip_dtypes(self):
        batch = _make_batch()
        buf = io.BytesIO()
        torch.save(batch, buf)
        buf.seek(0)
        loaded = torch.load(buf, map_location="cpu", weights_only=True)

        for k in batch:
            self.assertEqual(batch[k].shape, loaded[k].shape,
                             f"Shape mismatch for {k}")
            self.assertEqual(batch[k].dtype, loaded[k].dtype,
                             f"Dtype mismatch for {k}")
            torch.testing.assert_close(batch[k].float(), loaded[k].float(),
                                       msg=f"Value mismatch for {k}")

    def test_bfloat16_preserved(self):
        t = torch.randn(4, 32).bfloat16()
        buf = io.BytesIO()
        torch.save(t, buf)
        buf.seek(0)
        loaded = torch.load(buf, map_location="cpu", weights_only=True)
        self.assertEqual(loaded.dtype, torch.bfloat16)


class TestTensorBatchInferRPC(unittest.TestCase):
    """End-to-end TensorBatchInfer with a mock VLA on a real gRPC server."""

    @classmethod
    def setUpClass(cls):
        mock_vla = _MockVLAModel(n_action_steps=N_ACTION_STEPS,
                                 action_dim=ACTION_DIM)
        cls._vla_pipeline = VLAInferPipeline(
            vla_model=mock_vla,
            device="cpu",
            mixed_precision_dtype=torch.float32,
        )
        cls.server = serve(
            host="localhost", port=TEST_PORT,
            vla_pipeline=cls._vla_pipeline,
        )
        time.sleep(0.3)

    @classmethod
    def tearDownClass(cls):
        cls.server.stop(grace=1)

    def _make_request(self, unnorm_key="libero_10"):
        batch = _make_batch()
        buf = io.BytesIO()
        torch.save(batch, buf)
        return vla_service_pb2.TensorBatchInferRequest(
            batch_data=buf.getvalue(),
            unnorm_key=unnorm_key,
        )

    def _get_stub(self):
        opts = [
            ("grpc.max_send_message_length", 64 * 1024 * 1024),
            ("grpc.max_receive_message_length", 64 * 1024 * 1024),
        ]
        channel = grpc.insecure_channel(f"localhost:{TEST_PORT}", options=opts)
        return vla_service_pb2_grpc.VLAServiceStub(channel), channel

    def test_01_basic_infer(self):
        """Send a batch, get actions back with correct shape."""
        stub, channel = self._get_stub()
        req = self._make_request()
        resp = stub.TensorBatchInfer(req, timeout=10)

        actions = torch.load(io.BytesIO(resp.action_data),
                             map_location="cpu", weights_only=True)
        self.assertEqual(actions.shape, (1, N_ACTION_STEPS, ACTION_DIM))
        self.assertGreater(resp.infer_time, 0.0)
        channel.close()
        print(f"  [PASS] basic_infer: actions shape={actions.shape}, "
              f"infer_time={resp.infer_time:.4f}s")

    def test_02_unnorm_key_passthrough(self):
        """Verify unnorm_key arrives at the pipeline."""
        stub, channel = self._get_stub()
        req = self._make_request(unnorm_key="test_key_42")
        resp = stub.TensorBatchInfer(req, timeout=10)
        # Just ensure no error — the mock VLA ignores unnorm_key
        actions = torch.load(io.BytesIO(resp.action_data),
                             map_location="cpu", weights_only=True)
        self.assertEqual(actions.shape[0], 1)
        channel.close()
        print("  [PASS] unnorm_key_passthrough")

    def test_03_invalid_batch_data(self):
        """Server returns INVALID_ARGUMENT for corrupt batch data."""
        stub, channel = self._get_stub()
        req = vla_service_pb2.TensorBatchInferRequest(
            batch_data=b"not_a_valid_tensor_payload",
            unnorm_key="test",
        )
        with self.assertRaises(grpc.RpcError) as ctx:
            stub.TensorBatchInfer(req, timeout=10)
        self.assertEqual(ctx.exception.code(),
                         grpc.StatusCode.INVALID_ARGUMENT)
        channel.close()
        print(f"  [PASS] invalid_batch_data: {ctx.exception.details()[:60]}...")


class TestRemoteVLA(unittest.TestCase):
    """Test the RemoteVLA proxy class end-to-end."""

    @classmethod
    def setUpClass(cls):
        mock_vla = _MockVLAModel(n_action_steps=N_ACTION_STEPS,
                                 action_dim=ACTION_DIM)
        cls._vla_pipeline = VLAInferPipeline(
            vla_model=mock_vla,
            device="cpu",
            mixed_precision_dtype=torch.float32,
        )
        cls.server = serve(
            host="localhost", port=TEST_PORT + 1,
            vla_pipeline=cls._vla_pipeline,
        )
        time.sleep(0.3)

    @classmethod
    def tearDownClass(cls):
        cls.server.stop(grace=1)

    def test_01_predict_action(self):
        """RemoteVLA.predict_action returns correct tensor."""
        from remote_vla import RemoteVLA

        vla = RemoteVLA(host="localhost", port=TEST_PORT + 1,
                        device="cpu")
        batch = _make_batch()
        batch["unnorm_key"] = "libero_10"

        actions = vla.predict_action(**batch)
        self.assertEqual(actions.shape, (1, N_ACTION_STEPS, ACTION_DIM))
        self.assertTrue(actions.device == torch.device("cpu"))
        vla.close()
        print(f"  [PASS] RemoteVLA.predict_action: shape={actions.shape}")

    def test_02_duck_typing(self):
        """RemoteVLA supports eval(), cuda(), freeze_* attributes."""
        from remote_vla import RemoteVLA

        vla = RemoteVLA(host="localhost", port=TEST_PORT + 1, device="cpu")
        self.assertIs(vla.eval(), vla)
        self.assertIs(vla.cuda(0), vla)
        vla.freeze_vision_backbone = True
        vla.freeze_llm_backbone = True
        vla.norm_stats = {"test": 123}
        self.assertEqual(vla.norm_stats, {"test": 123})
        vla.close()
        print("  [PASS] RemoteVLA duck-typing")


class TestNoVLAPipeline(unittest.TestCase):
    """TensorBatchInfer returns UNIMPLEMENTED when no VLA pipeline is set."""

    @classmethod
    def setUpClass(cls):
        cls.server = serve(host="localhost", port=TEST_PORT + 2)
        time.sleep(0.3)

    @classmethod
    def tearDownClass(cls):
        cls.server.stop(grace=1)

    def test_unimplemented(self):
        opts = [
            ("grpc.max_send_message_length", 64 * 1024 * 1024),
            ("grpc.max_receive_message_length", 64 * 1024 * 1024),
        ]
        channel = grpc.insecure_channel(f"localhost:{TEST_PORT + 2}",
                                        options=opts)
        stub = vla_service_pb2_grpc.VLAServiceStub(channel)
        batch = _make_batch()
        buf = io.BytesIO()
        torch.save(batch, buf)
        req = vla_service_pb2.TensorBatchInferRequest(
            batch_data=buf.getvalue(), unnorm_key="test")
        with self.assertRaises(grpc.RpcError) as ctx:
            stub.TensorBatchInfer(req, timeout=10)
        self.assertEqual(ctx.exception.code(), grpc.StatusCode.UNIMPLEMENTED)
        channel.close()
        print(f"  [PASS] no_vla_pipeline: {ctx.exception.details()[:60]}...")


if __name__ == "__main__":
    unittest.main(verbosity=2)
