"""Tests for ZMQ VLA remote inference: TensorSerializer, VLAPolicy server,
RemoteVLAZmq client, and end-to-end predict_action round-trip."""

import io
import threading
import time
from typing import Any

import numpy as np
import pytest
import torch

from .policy import BasePolicy
from .server_client import PolicyServer
from .vla_server import TensorSerializer, VLAPolicy, VLAInferPipeline, create_vla_server
from .remote_vla_zmq import RemoteVLAZmq

_port = 19700


def _next_port():
    global _port
    _port += 1
    return _port


# =====================================================================
# Mock VLA model for testing (no real GPU needed)
# =====================================================================
class MockVLAModel:
    """Mimics a real VLA model's interface."""

    def __init__(self, action_dim=32, n_action_steps=10):
        self._action_dim = action_dim
        self._n_steps = n_action_steps
        self._last_unnorm_key = None

    def eval(self):
        return self

    def to(self, device):
        return self

    def predict_action(self, **batch):
        self._last_unnorm_key = batch.get("unnorm_key", None)
        bs = 1
        for v in batch.values():
            if isinstance(v, torch.Tensor) and v.dim() >= 1:
                bs = v.shape[0]
                break
        return torch.randn(bs, self._n_steps, self._action_dim) * 0.1


# =====================================================================
# Test TensorSerializer
# =====================================================================
class TestTensorSerializer:
    def test_round_trip_dtypes(self):
        """Batch dict with multiple dtypes survives serialization."""
        batch = {
            "images": torch.randn(1, 6, 224, 224, dtype=torch.float32),
            "lang_tokens": torch.randint(0, 1000, (1, 48), dtype=torch.int64),
            "states": torch.randn(1, 32, dtype=torch.float32),
        }
        data = TensorSerializer.serialize_batch(batch)
        restored = TensorSerializer.deserialize_batch(data)
        for k in batch:
            assert torch.equal(batch[k], restored[k]), f"Mismatch in {k}"
            assert batch[k].dtype == restored[k].dtype, f"Dtype mismatch in {k}"

    def test_bfloat16_preserved(self):
        """bfloat16 tensors should survive round-trip."""
        t = torch.randn(4, 8, dtype=torch.bfloat16)
        data = TensorSerializer.serialize_batch({"t": t})
        restored = TensorSerializer.deserialize_batch(data)
        assert restored["t"].dtype == torch.bfloat16
        assert torch.equal(t, restored["t"])

    def test_action_serialization(self):
        """Action tensor round-trip."""
        actions = torch.randn(1, 10, 32)
        data = TensorSerializer.serialize_actions(actions)
        restored = TensorSerializer.deserialize_actions(data)
        assert torch.equal(actions, restored)


# =====================================================================
# Fixtures
# =====================================================================
@pytest.fixture()
def mock_vla_server():
    """Start a ZMQ VLA server with MockVLAModel in a background thread."""
    port = _next_port()
    model = MockVLAModel()
    pipeline = VLAInferPipeline.__new__(VLAInferPipeline)
    # Bypass real GPU init
    pipeline._vla = model
    pipeline._device = torch.device("cpu")
    pipeline._dtype = torch.float32

    # Override predict_action to work on CPU without autocast
    def cpu_predict_action(self, **batch):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self._device)
        return self._vla.predict_action(**batch)

    import types
    pipeline.predict_action = types.MethodType(cpu_predict_action, pipeline)

    server = create_vla_server(pipeline, host="127.0.0.1", port=port)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    time.sleep(0.3)

    yield port, server, model

    server.close()
    thread.join(timeout=2)


# =====================================================================
# Test VLAPolicy + RemoteVLAZmq end-to-end
# =====================================================================
class TestVLAPolicyE2E:
    def test_01_basic_infer(self, mock_vla_server):
        """RemoteVLAZmq.predict_action returns correct shape."""
        port, server, model = mock_vla_server
        client = RemoteVLAZmq(host="127.0.0.1", port=port,
                               timeout_s=5.0, device="cpu",
                               enable_profiling=False)
        try:
            batch = {
                "images": torch.randn(1, 6, 224, 224),
                "lang_tokens": torch.randint(0, 1000, (1, 48)),
                "states": torch.randn(1, 32),
            }
            actions = client.predict_action(**batch)
            assert isinstance(actions, torch.Tensor)
            assert actions.shape == (1, 10, 32)
        finally:
            client.close()

    def test_02_unnorm_key_passthrough(self, mock_vla_server):
        """unnorm_key is forwarded to the server pipeline."""
        port, server, model = mock_vla_server
        client = RemoteVLAZmq(host="127.0.0.1", port=port,
                               timeout_s=5.0, device="cpu",
                               enable_profiling=False)
        try:
            batch = {
                "images": torch.randn(1, 6, 224, 224),
                "unnorm_key": "libero_10",
            }
            actions = client.predict_action(**batch)
            assert isinstance(actions, torch.Tensor)
            assert model._last_unnorm_key == "libero_10"
        finally:
            client.close()

    def test_03_ping(self, mock_vla_server):
        """Ping health check works."""
        port, server, model = mock_vla_server
        client = RemoteVLAZmq(host="127.0.0.1", port=port,
                               timeout_s=5.0, device="cpu")
        try:
            assert client.ping() is True
        finally:
            client.close()

    def test_04_profiling(self, mock_vla_server):
        """Profiling accumulates stats correctly."""
        port, server, model = mock_vla_server
        client = RemoteVLAZmq(host="127.0.0.1", port=port,
                               timeout_s=5.0, device="cpu",
                               enable_profiling=True)
        try:
            for _ in range(3):
                client.predict_action(images=torch.randn(1, 6, 224, 224))
            stats = client.get_profiling_stats()
            assert stats["call_count"] == 3
            assert stats["avg_total_ms"] > 0
            assert stats["avg_serialize_ms"] > 0
            assert stats["avg_zmq_roundtrip_ms"] > 0
            assert stats["avg_payload_kb"] > 0
        finally:
            client.close()


class TestRemoteVLAZmqDuckTyping:
    def test_duck_typing(self, mock_vla_server):
        """RemoteVLAZmq supports eval(), cuda(), freeze_*, norm_stats."""
        port, server, model = mock_vla_server
        client = RemoteVLAZmq(host="127.0.0.1", port=port,
                               timeout_s=5.0, device="cpu")
        try:
            # eval() returns self
            assert client.eval() is client
            # cuda() returns self
            assert client.cuda(0) is client
            # freeze_* attributes can be set
            client.freeze_vision_backbone = True
            client.freeze_llm_backbone = False
            assert client.freeze_vision_backbone is True
            assert client.freeze_llm_backbone is False
            # norm_stats
            client.norm_stats = {"key": "value"}
            assert client.norm_stats == {"key": "value"}
        finally:
            client.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
