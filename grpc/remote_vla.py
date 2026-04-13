"""RemoteVLA: drop-in replacement for a local VLA model that delegates
inference to a remote gRPC server via the TensorBatchInfer RPC.

Usage::

    from grpc.remote_vla import RemoteVLA

    vla = RemoteVLA(host="192.168.1.100", port=50051, device="cuda:0")
    actions = vla.predict_action(**batch)   # same interface as real VLA
"""
from __future__ import annotations

import io
import time

import torch
import grpc

import vla_service_pb2
import vla_service_pb2_grpc

_MAX_MSG_BYTES = 64 * 1024 * 1024  # 64 MB


class RemoteVLA:
    """Client-side proxy that mirrors a real VLA model's ``predict_action``
    interface but forwards the call to a remote gRPC server."""

    def __init__(self,
                 host: str = "localhost",
                 port: int = 50051,
                 timeout_s: float = 30.0,
                 device: str = "cuda:0",
                 enable_profiling: bool = True):
        self._address = f"{host}:{port}"
        self._timeout_s = timeout_s
        self._device = torch.device(device)
        self._enable_profiling = enable_profiling

        opts = [
            ("grpc.max_send_message_length", _MAX_MSG_BYTES),
            ("grpc.max_receive_message_length", _MAX_MSG_BYTES),
        ]
        self._channel = grpc.insecure_channel(self._address, options=opts)
        self._stub = vla_service_pb2_grpc.VLAServiceStub(self._channel)

        # Attributes that LiberoEvalRunner / run_setup expect to exist
        self.norm_stats = {}
        self.freeze_vision_backbone = True
        self.freeze_llm_backbone = True
        self.freeze_projector = True
        self.freeze_vlm_backbone = True

        # Profiling accumulators
        self._call_count = 0
        self._t_serialize = 0.0
        self._t_grpc = 0.0
        self._t_deserialize = 0.0
        self._t_total = 0.0
        self._t_server_infer = 0.0
        self._payload_bytes = 0

    # ------------------------------------------------------------------
    # Duck-typing helpers so run_setup() works without branching
    # ------------------------------------------------------------------
    def eval(self):
        return self

    def cuda(self, device=None):
        if device is not None:
            self._device = torch.device(f"cuda:{device}" if isinstance(device, int) else device)
        return self

    def __setattr__(self, name, value):
        # Silently absorb freeze_* assignments from run_setup
        if name.startswith("freeze_"):
            object.__setattr__(self, name, value)
            return
        object.__setattr__(self, name, value)

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------
    def predict_action(self, **kwargs):
        """Serialize *batch*, send to the remote server, return actions tensor.

        Accepts the same keyword arguments as a real VLA's
        ``predict_action(images=..., lang_tokens=..., states=..., ...)``.
        """
        t_total_start = time.perf_counter()

        # Separate non-tensor fields
        unnorm_key = kwargs.pop("unnorm_key", "")

        # --- Serialize ---
        t0 = time.perf_counter()
        batch_cpu = {}
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                batch_cpu[k] = v.cpu()
            else:
                batch_cpu[k] = v
        buf = io.BytesIO()
        torch.save(batch_cpu, buf)
        payload = buf.getvalue()
        t_serialize = time.perf_counter() - t0

        # --- gRPC call ---
        t1 = time.perf_counter()
        request = vla_service_pb2.TensorBatchInferRequest(
            batch_data=payload,
            unnorm_key=str(unnorm_key),
        )
        response = self._stub.TensorBatchInfer(
            request, timeout=self._timeout_s)
        t_grpc = time.perf_counter() - t1

        # --- Deserialize ---
        t2 = time.perf_counter()
        action_buf = io.BytesIO(response.action_data)
        actions = torch.load(action_buf, map_location=self._device,
                             weights_only=True)
        t_deserialize = time.perf_counter() - t2

        t_total = time.perf_counter() - t_total_start

        # --- Profiling ---
        if self._enable_profiling:
            self._call_count += 1
            self._t_serialize += t_serialize
            self._t_grpc += t_grpc
            self._t_deserialize += t_deserialize
            self._t_total += t_total
            self._t_server_infer += response.infer_time
            self._payload_bytes += len(payload)

            if self._call_count % 50 == 0:
                n = self._call_count
                print(f"[RemoteVLA profiling] calls={n}  "
                      f"avg_total={self._t_total/n*1000:.1f}ms  "
                      f"avg_serialize={self._t_serialize/n*1000:.1f}ms  "
                      f"avg_grpc_roundtrip={self._t_grpc/n*1000:.1f}ms  "
                      f"avg_server_infer={self._t_server_infer/n*1000:.1f}ms  "
                      f"avg_deserialize={self._t_deserialize/n*1000:.1f}ms  "
                      f"avg_payload={self._payload_bytes/n/1024:.0f}KB",
                      flush=True)

        return actions

    def get_profiling_stats(self) -> dict:
        """Return accumulated profiling statistics."""
        n = max(self._call_count, 1)
        return {
            "call_count": self._call_count,
            "avg_total_ms": self._t_total / n * 1000,
            "avg_serialize_ms": self._t_serialize / n * 1000,
            "avg_grpc_roundtrip_ms": self._t_grpc / n * 1000,
            "avg_server_infer_ms": self._t_server_infer / n * 1000,
            "avg_deserialize_ms": self._t_deserialize / n * 1000,
            "avg_payload_kb": self._payload_bytes / n / 1024,
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def close(self):
        self._channel.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
