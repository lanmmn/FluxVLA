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

import numpy as np
import torch

from .policy import BasePolicy
from .server_client import MsgSerializer, ObsSerializer, PolicyServer


class TensorSerializer:
    """Serialize/deserialize action tensors via numpy npy format,
    transported as bytes inside msgpack messages.

    使用 numpy 而非 torch.save: 无 pickle 元数据开销,
    同等数据约小 20-30%, 反序列化更快。"""

    @staticmethod
    def serialize_actions(actions: torch.Tensor) -> bytes:
        """Serialize an action tensor to bytes (numpy npy format)."""
        buf = io.BytesIO()
        np.save(buf, actions.cpu().numpy(), allow_pickle=False)
        return buf.getvalue()

    @staticmethod
    def deserialize_actions(data: bytes,
                            device: str = "cpu") -> torch.Tensor:
        """Deserialize bytes back to an action tensor."""
        arr = np.load(io.BytesIO(data), allow_pickle=False)
        return torch.from_numpy(arr.copy()).to(device)


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
    - ``obs_data``: bytes of raw observation (JPEG images + msgpack)
    - ``unnorm_key``: optional string for denormalization

    The server deserializes the raw observation, runs the dataset
    preprocessing pipeline, then runs model inference.

    It returns a msgpack message containing:
    - ``action_data``: bytes produced by ``TensorSerializer.serialize_actions``
    - ``infer_time``: float, server-side inference time in seconds
    """

    def __init__(self, pipeline: VLAInferPipeline, dataset=None,
                 denormalize_action=None, task_suite_name: str = ""):
        super().__init__()
        self._pipeline = pipeline
        self._dataset = dataset
        self._denormalize_action = denormalize_action
        self._task_suite_name = task_suite_name
        self._lock = threading.Lock()
        self._total_requests = 0
        self._total_infer_time = 0.0
        self._start_time = time.time()

    def _get_action(self, observation, options=None):
        return {"error": "Use predict_action endpoint"}, {}

    def reset(self, options=None):
        return {"status": "ok"}

    def predict_action(self, obs_data: bytes,
                       unnorm_key: str = "") -> dict:
        """Receive raw observation, preprocess, run inference, return actions.

        Args:
            obs_data: Serialized raw observation (JPEG images + msgpack).
            unnorm_key: Optional key for action denormalization.
        """
        # --- Deserialize raw obs ---
        t0 = time.perf_counter()
        obs = ObsSerializer.from_bytes(obs_data)
        t_deserialize = time.perf_counter() - t0

        # --- Preprocess (dataset transforms) ---
        t1 = time.perf_counter()
        if self._dataset is not None:
            result = self._dataset(obs)
            batch = result[0] if isinstance(result, tuple) else result
        else:
            batch = obs
        if unnorm_key:
            batch["unnorm_key"] = unnorm_key
        t_preprocess = time.perf_counter() - t1

        # --- Model inference ---
        t2 = time.perf_counter()
        actions = self._pipeline.predict_action(**batch)
        infer_time = time.perf_counter() - t2

        # --- Denormalize actions (server-side, 批量化) ---
        if self._denormalize_action is not None:
            actions_np = actions.cpu().numpy()
            task_name = self._task_suite_name
            # 传整个 chunk 一次调用,denorm 内部用 numpy broadcast 处理
            # actions_np shape: (1, chunk, dim) 或 (1, dim)
            d = self._denormalize_action(dict(
                action=actions_np[0],          # (chunk, dim) 或 (dim,)
                task_suite_name=task_name))
            actions = torch.from_numpy(d[None].astype(np.float32))

        # --- Serialize actions ---
        t3 = time.perf_counter()
        action_bytes = TensorSerializer.serialize_actions(actions)
        t_serialize = time.perf_counter() - t3

        with self._lock:
            self._total_requests += 1
            self._total_infer_time += infer_time
            n = self._total_requests
            should_print = (n % 50 == 0)
            avg = self._total_infer_time / n if should_print else 0.0

        if should_print:   # I/O 移到锁外,避免阻塞其他线程
            print(f"[ZMQ VLAServer] req={n}  "
                  f"deserialize={t_deserialize*1000:.1f}ms  "
                  f"preprocess={t_preprocess*1000:.1f}ms  "
                  f"infer={infer_time*1000:.1f}ms  "
                  f"serialize={t_serialize*1000:.1f}ms  "
                  f"avg_infer={avg*1000:.1f}ms",
                  flush=True)

        return {
            "action_data": action_bytes,
            "infer_time": infer_time + t_preprocess,
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
                      port: int = 5555,
                      dataset=None,
                      denormalize_action=None,
                      task_suite_name: str = "") -> PolicyServer:
    """Create a ZMQ PolicyServer serving a VLA model.

    Args:
        pipeline: VLA inference pipeline.
        host: Bind address.
        port: Bind port.
        dataset: Optional dataset/transform pipeline for server-side
            preprocessing. When provided, clients send raw observations
            and the server handles preprocessing before inference.
        denormalize_action: Optional denormalization transform for
            server-side action denormalization.
        task_suite_name: Task suite name for denormalization lookup.
    """
    policy = VLAPolicy(pipeline, dataset=dataset,
                       denormalize_action=denormalize_action,
                       task_suite_name=task_suite_name)
    server = PolicyServer(policy, host=host, port=port)
    # Register VLA-specific endpoints
    server.register_endpoint("predict_action", policy.predict_action)
    server.register_endpoint("get_status", policy.get_status,
                             requires_input=False)
    return server
