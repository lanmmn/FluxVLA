"""RemoteVLAZmq -- drop-in replacement for a local VLA model that
delegates inference to a remote ZMQ server.

Usage::

    from fluxvla.remote import RemoteVLAZmq

    vla = RemoteVLAZmq(host="192.168.1.100", port=5555, device="cuda:0")
    actions = vla.predict_action(**batch)
"""
from __future__ import annotations
import io
import threading
import time

import msgpack
import numpy as np
import torch
import zmq

from .serializers import (FORMAT_PROTOBUF, decode_predict_response,
                          encode_predict_request)


class RemoteVLAZmq:
    """Client-side proxy that mirrors the ``predict_action`` interface of a
    real VLA model but forwards all inference to a remote ZMQ server.

    Duck-typing compatibility with local models (eval/cuda/freeze_*) allows
    runners to treat this object identically to a local model.
    """

    def __init__(self,
                 host: str = 'localhost',
                 port: int = 5555,
                 timeout_s: float = 30.0,
                 device: str = 'cuda:0',
                 enable_profiling: bool = True,
                 serializer: str = 'msgpack',
                 compress: bool = True):
        """
        Args:
            host: Remote server hostname or IP.
            port: Remote server port.
            timeout_s: ZMQ send/recv timeout in seconds.
            device: Torch device for the returned action tensor.
            enable_profiling: Print average latency every 50 calls.
            serializer: Wire format -- ``'msgpack'`` or ``'protobuf'``.
            compress: JPEG-compress RGB images before sending.  Set False
                for lossless npy transfer (~10x larger payload).
        """
        assert serializer in ('msgpack', 'protobuf'), \
            f"serializer must be 'msgpack' or 'protobuf', got '{serializer}'"
        self._serializer = serializer
        self._compress = compress
        self._host = host
        self._port = port
        self._address = f'tcp://{host}:{port}'
        self._timeout_ms = int(timeout_s * 1000)
        self._device = torch.device(device)
        self._enable_profiling = enable_profiling

        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
        self._socket.setsockopt(zmq.SNDTIMEO, self._timeout_ms)
        self._socket.connect(self._address)

        self._lock = threading.Lock()
        self._closed = False

        self.norm_stats = {}
        self.freeze_vision_backbone = True
        self.freeze_llm_backbone = True
        self.freeze_projector = True
        self.freeze_vlm_backbone = True

        self._last_profile = {}
        self._call_count = 0
        self._t_serialize = 0.0
        self._t_zmq = 0.0
        self._t_deserialize = 0.0
        self._t_total = 0.0
        self._t_server_infer = 0.0
        self._payload_bytes = 0

    # ------------------------------------------------------------------
    # Duck-typing helpers
    # ------------------------------------------------------------------
    def eval(self):
        """No-op; remote model is always in eval mode."""
        return self

    def cuda(self, device=None):
        """Record device for action tensor placement; no actual transfer."""
        if device is not None:
            self._device = torch.device(
                f'cuda:{device}' if isinstance(device, int) else device)
        return self

    def __setattr__(self, name, value):
        """Silently absorb ``freeze_*`` assignments from runner setup."""
        object.__setattr__(self, name, value)

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------
    def predict_action(self, **kwargs):
        """Serialize raw observations, send to the remote server, and return
        the denormalized action tensor on ``self._device``.

        Accepts numpy arrays, torch tensors (auto-converted), and strings.

        Returns:
            torch.Tensor: Action tensor of shape ``(1, n_action_steps, action_dim)``.
        """
        t_total_start = time.perf_counter()

        unnorm_key = kwargs.pop('unnorm_key', '')

        t0 = time.perf_counter()
        obs = {}
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                obs[k] = v.cpu().numpy()
            else:
                obs[k] = v
        request = encode_predict_request(
            obs,
            str(unnorm_key),
            fmt=self._serializer,
            compress=self._compress)
        payload_size = len(request)
        t_serialize = time.perf_counter() - t0

        t1 = time.perf_counter()
        with self._lock:
            self._socket.send(request)
            raw_response = self._socket.recv()
        fmt_tag = FORMAT_PROTOBUF if self._serializer == 'protobuf' else 0
        response = decode_predict_response(raw_response, fmt=fmt_tag)
        t_zmq = time.perf_counter() - t1

        if isinstance(response, dict) and 'error' in response:
            raise RuntimeError(f"ZMQ server error: {response['error']}")

        t2 = time.perf_counter()
        action_buf = io.BytesIO(response['action_data'])
        arr = np.load(action_buf, allow_pickle=False)
        actions = torch.from_numpy(arr.copy()).to(self._device)
        t_deserialize = time.perf_counter() - t2

        t_total = time.perf_counter() - t_total_start
        server_infer = response.get('infer_time', 0.0)

        self._last_profile = {
            'serialize_ms': t_serialize * 1000,
            'zmq_roundtrip_ms': t_zmq * 1000,
            'server_infer_ms': server_infer * 1000,
            'network_ms': (t_zmq - server_infer) * 1000,
            'deserialize_ms': t_deserialize * 1000,
            'total_ms': t_total * 1000,
            'payload_kb': payload_size / 1024,
        }

        if self._enable_profiling:
            self._call_count += 1
            self._t_serialize += t_serialize
            self._t_zmq += t_zmq
            self._t_deserialize += t_deserialize
            self._t_total += t_total
            self._t_server_infer += server_infer
            self._payload_bytes += payload_size

            if self._call_count % 50 == 0:
                n = self._call_count
                print(
                    f'[RemoteVLAZmq profiling] calls={n}  '
                    f'avg_total={self._t_total/n*1000:.1f}ms  '
                    f'avg_serialize={self._t_serialize/n*1000:.1f}ms  '
                    f'avg_zmq_roundtrip={self._t_zmq/n*1000:.1f}ms  '
                    f'avg_server_infer={self._t_server_infer/n*1000:.1f}ms  '
                    f'avg_deserialize={self._t_deserialize/n*1000:.1f}ms  '
                    f'avg_payload={self._payload_bytes/n/1024:.0f}KB',
                    flush=True)

        return actions

    def ping(self) -> bool:
        """Send a health-check ping; return True if the server responds OK."""
        try:
            request = msgpack.packb({'endpoint': 'ping'})
            with self._lock:
                self._socket.send(request)
                raw = self._socket.recv()
            resp = msgpack.unpackb(raw, raw=False)
            return resp.get('status') == 'ok'
        except zmq.error.ZMQError:
            return False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def close(self):
        """Release ZMQ resources.  Idempotent and safe to call multiple times."""
        if getattr(self, '_closed', False):
            return
        self._closed = True
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
