"""RemoteVLAZmq: drop-in replacement for a local VLA model that delegates
inference to a remote ZMQ server.

Usage::

    from zmq_msgpack.remote_vla_zmq import RemoteVLAZmq

    vla = RemoteVLAZmq(host="192.168.1.100", port=5555, device="cuda:0")
    actions = vla.predict_action(**batch)   # same interface as real VLA
"""
from __future__ import annotations

import io
import time

import torch
import zmq
import msgpack


class RemoteVLAZmq:
    """Client-side proxy that mirrors a real VLA model's ``predict_action``
    interface but forwards the call to a remote ZMQ server."""

    def __init__(self,
                 host: str = "localhost",
                 port: int = 5555,
                 timeout_s: float = 30.0,
                 device: str = "cuda:0",
                 enable_profiling: bool = True):
        self._host = host
        self._port = port
        self._address = f"tcp://{host}:{port}"
        self._timeout_ms = int(timeout_s * 1000)
        self._device = torch.device(device)
        self._enable_profiling = enable_profiling

        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
        self._socket.setsockopt(zmq.SNDTIMEO, self._timeout_ms)
        self._socket.connect(self._address)

        # Attributes that LiberoEvalRunner / run_setup expect to exist
        self.norm_stats = {}
        self.freeze_vision_backbone = True
        self.freeze_llm_backbone = True
        self.freeze_projector = True
        self.freeze_vlm_backbone = True

        # Profiling accumulators
        self._call_count = 0
        self._t_serialize = 0.0
        self._t_zmq = 0.0
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
            self._device = torch.device(
                f"cuda:{device}" if isinstance(device, int) else device)
        return self

    def __setattr__(self, name, value):
        if name.startswith("freeze_"):
            object.__setattr__(self, name, value)
            return
        object.__setattr__(self, name, value)

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------
    def predict_action(self, **kwargs):
        """Serialize *batch*, send to the remote server, return actions tensor."""
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

        # --- ZMQ call ---
        t1 = time.perf_counter()
        request = msgpack.packb({
            "endpoint": "predict_action",
            "data": {
                "batch_data": payload,
                "unnorm_key": str(unnorm_key),
            },
        })
        self._socket.send(request)
        raw_response = self._socket.recv()
        response = msgpack.unpackb(raw_response)
        t_zmq = time.perf_counter() - t1

        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"ZMQ server error: {response['error']}")

        # --- Deserialize ---
        t2 = time.perf_counter()
        action_buf = io.BytesIO(response[b"action_data"]
                                if b"action_data" in response
                                else response["action_data"])
        actions = torch.load(action_buf, map_location=self._device,
                             weights_only=True)
        t_deserialize = time.perf_counter() - t2

        t_total = time.perf_counter() - t_total_start

        # Server infer time
        server_infer = (response.get(b"infer_time", 0.0)
                        if b"infer_time" in response
                        else response.get("infer_time", 0.0))

        # --- Profiling ---
        if self._enable_profiling:
            self._call_count += 1
            self._t_serialize += t_serialize
            self._t_zmq += t_zmq
            self._t_deserialize += t_deserialize
            self._t_total += t_total
            self._t_server_infer += server_infer
            self._payload_bytes += len(payload)

            if self._call_count % 50 == 0:
                n = self._call_count
                print(f"[RemoteVLAZmq profiling] calls={n}  "
                      f"avg_total={self._t_total/n*1000:.1f}ms  "
                      f"avg_serialize={self._t_serialize/n*1000:.1f}ms  "
                      f"avg_zmq_roundtrip={self._t_zmq/n*1000:.1f}ms  "
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
            "avg_zmq_roundtrip_ms": self._t_zmq / n * 1000,
            "avg_server_infer_ms": self._t_server_infer / n * 1000,
            "avg_deserialize_ms": self._t_deserialize / n * 1000,
            "avg_payload_kb": self._payload_bytes / n / 1024,
        }

    def ping(self) -> bool:
        """Health check."""
        try:
            request = msgpack.packb({"endpoint": "ping"})
            self._socket.send(request)
            raw = self._socket.recv()
            resp = msgpack.unpackb(raw)
            return (resp.get(b"status") == b"ok"
                    or resp.get("status") == "ok")
        except zmq.error.ZMQError:
            return False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def close(self):
        if hasattr(self, '_socket') and not self._socket.closed:
            self._socket.setsockopt(zmq.LINGER, 0)
            self._socket.close()
        if hasattr(self, '_context'):
            self._context.term()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
