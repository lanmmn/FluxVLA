"""ZMQ VLA Inference Server: wraps a VLA model as a BasePolicy and
serves it via ZMQ REP socket with torch tensor batch serialization.

Usage::

    python zmq_msgpack/serve_vla_zmq.py \
        --config configs/pi05/pi05_paligemma_libero_10_full_finetune.py \
        --ckpt-path /path/to/checkpoint.pt \
        --host 0.0.0.0 --port 5555
"""
from __future__ import annotations

import io
import threading
import time
from typing import Any

import torch

from .policy import BasePolicy
from .server_client import MsgSerializer, PolicyServer


class TensorSerializer:
    """Serialize/deserialize torch tensor batches via torch.save/torch.load,
    transported as bytes inside msgpack messages."""

    @staticmethod
    def serialize_batch(batch: dict) -> bytes:
        """Serialize a dict of tensors to bytes using torch.save."""
        buf = io.BytesIO()
        torch.save(batch, buf)
        return buf.getvalue()

    @staticmethod
    def deserialize_batch(data: bytes, map_location: str = "cpu") -> dict:
        """Deserialize bytes back to a dict of tensors using torch.load."""
        return torch.load(io.BytesIO(data), map_location=map_location,
                          weights_only=True)

    @staticmethod
    def serialize_actions(actions: torch.Tensor) -> bytes:
        """Serialize an action tensor to bytes."""
        buf = io.BytesIO()
        torch.save(actions.cpu(), buf)
        return buf.getvalue()

    @staticmethod
    def deserialize_actions(data: bytes,
                            device: str = "cpu") -> torch.Tensor:
        """Deserialize bytes back to an action tensor."""
        return torch.load(io.BytesIO(data), map_location=device,
                          weights_only=True)


class VLAInferPipeline:
    """Wraps a real VLA model for server-side tensor-batch inference."""

    def __init__(self, vla_model, device: str = "cuda:0",
                 mixed_precision_dtype=torch.bfloat16):
        self._vla = vla_model
        self._device = torch.device(device)
        self._dtype = mixed_precision_dtype
        self._vla.eval()
        self._vla.to(self._device)

    @torch.no_grad()
    def predict_action(self, **batch) -> torch.Tensor:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self._device)
        with torch.autocast("cuda", dtype=self._dtype, enabled=True):
            actions = self._vla.predict_action(**batch)
        return actions


class VLAPolicy(BasePolicy):
    """Wraps VLAInferPipeline as a BasePolicy for use with PolicyServer.

    The ``predict_action`` endpoint receives a msgpack message containing:
    - ``batch_data``: bytes produced by ``TensorSerializer.serialize_batch``
    - ``unnorm_key``: optional string for denormalization

    It returns a msgpack message containing:
    - ``action_data``: bytes produced by ``TensorSerializer.serialize_actions``
    - ``infer_time``: float, server-side inference time in seconds
    """

    def __init__(self, pipeline: VLAInferPipeline):
        super().__init__(strict=False)
        self._pipeline = pipeline
        self._lock = threading.Lock()
        self._total_requests = 0
        self._total_infer_time = 0.0
        self._start_time = time.time()

    def _get_action(self, observation: dict[str, Any],
                    options: dict[str, Any] | None = None
                    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Not used for VLA inference. Use predict_action endpoint instead."""
        return {"error": "Use predict_action endpoint"}, {}

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        return {"status": "ok"}

    def predict_action(self, batch_data: bytes,
                       unnorm_key: str = "") -> dict:
        """Receive serialized torch batch, run inference, return serialized actions."""
        t0 = time.perf_counter()
        batch = TensorSerializer.deserialize_batch(batch_data)
        t_deserialize = time.perf_counter() - t0

        if unnorm_key:
            batch["unnorm_key"] = unnorm_key

        t1 = time.perf_counter()
        actions = self._pipeline.predict_action(**batch)
        infer_time = time.perf_counter() - t1

        t2 = time.perf_counter()
        action_bytes = TensorSerializer.serialize_actions(actions)
        t_serialize = time.perf_counter() - t2

        with self._lock:
            self._total_requests += 1
            self._total_infer_time += infer_time

            if self._total_requests % 50 == 0:
                n = self._total_requests
                avg = self._total_infer_time / n
                print(f"[ZMQ VLAServer] req={n}  "
                      f"deserialize={t_deserialize*1000:.1f}ms  "
                      f"infer={infer_time*1000:.1f}ms  "
                      f"serialize={t_serialize*1000:.1f}ms  "
                      f"avg_infer={avg*1000:.1f}ms",
                      flush=True)

        return {
            "action_data": action_bytes,
            "infer_time": infer_time,
        }

    def get_status(self) -> dict:
        """Return server status."""
        with self._lock:
            total = self._total_requests
            avg = (self._total_infer_time / total) if total > 0 else 0.0
        return {
            "status": "ready",
            "uptime_s": time.time() - self._start_time,
            "total_requests": total,
            "avg_infer_time": avg,
        }


def create_vla_server(pipeline: VLAInferPipeline,
                      host: str = "*",
                      port: int = 5555) -> PolicyServer:
    """Create a ZMQ PolicyServer serving a VLA model.

    Registers custom endpoints:
    - ``predict_action``: tensor batch inference
    - ``get_status``: server status query
    """
    policy = VLAPolicy(pipeline)
    server = PolicyServer(policy, host=host, port=port)
    # Register VLA-specific endpoints
    server.register_endpoint("predict_action", policy.predict_action)
    server.register_endpoint("get_status", policy.get_status,
                             requires_input=False)
    return server
