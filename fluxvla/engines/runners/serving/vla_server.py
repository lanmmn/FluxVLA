"""Simplified ZMQ VLA inference server (2-layer architecture).

Layer 1: PolicyServer -- generic ZMQ REP event loop + endpoint routing.
Layer 2: vla_handler  -- obs deserialize -> preprocess -> inference ->
                         denormalize -> serialize action.

Usage::

    python -m fluxvla.engines.runners.serving.serve \\
        --config configs/pi05/pi05_paligemma_libero_10_full_finetune.py \\
        --ckpt-path /path/to/checkpoint.pt
"""
from __future__ import annotations

import io
import time
import threading
from dataclasses import dataclass
from typing import Any, Callable, Optional

import cv2
import msgpack
import numpy as np
import torch
import zmq

from .serializers import (FORMAT_PROTOBUF, decode_predict_request,
                          encode_predict_response, detect_format)


# =========================================================================
# Serialization helpers (kept here so serializers.py can import ObsSerializer)
# =========================================================================
class MsgSerializer:
    """Msgpack serializer with built-in numpy array support."""

    @staticmethod
    def to_bytes(data: Any) -> bytes:
        return msgpack.packb(data, default=MsgSerializer._encode)

    @staticmethod
    def from_bytes(data: bytes) -> Any:
        return msgpack.unpackb(data, object_hook=MsgSerializer._decode)

    @staticmethod
    def _decode(obj):
        if '__ndarray__' in obj:
            return np.load(io.BytesIO(obj['data']), allow_pickle=False)
        return obj

    @staticmethod
    def _encode(obj):
        if isinstance(obj, np.ndarray):
            buf = io.BytesIO()
            np.save(buf, obj, allow_pickle=False)
            return {'__ndarray__': True, 'data': buf.getvalue()}
        raise TypeError(f'Cannot serialize {type(obj)}')


class ObsSerializer:
    """Serialize/deserialize raw observation dicts (JPEG images + npy arrays)."""

    JPEG_QUALITY = 95
    JPEG_KEYS = {
        'cam_high', 'cam_left_wrist', 'cam_right_wrist',
        'agentview_image', 'robot0_eye_in_hand_image',
    }

    @staticmethod
    def to_bytes(obs: dict, compress: bool = True) -> bytes:
        encoded = {}
        for k, v in obs.items():
            if (compress and isinstance(v, np.ndarray) and v.ndim == 3
                    and v.dtype == np.uint8 and k in ObsSerializer.JPEG_KEYS):
                _, jpg = cv2.imencode(
                    '.jpg', v,
                    [cv2.IMWRITE_JPEG_QUALITY, ObsSerializer.JPEG_QUALITY])
                encoded[k] = {'__jpeg__': True, 'data': jpg.tobytes()}
            elif isinstance(v, np.ndarray):
                buf = io.BytesIO()
                np.save(buf, v, allow_pickle=False)
                encoded[k] = {'__ndarray__': True, 'data': buf.getvalue()}
            else:
                encoded[k] = v
        return msgpack.packb(encoded)

    @staticmethod
    def from_bytes(data: bytes) -> dict:
        raw = msgpack.unpackb(data, raw=False)
        obs = {}
        for k, v in raw.items():
            if isinstance(v, dict):
                if '__jpeg__' in v:
                    obs[k] = cv2.imdecode(
                        np.frombuffer(v['data'], np.uint8), cv2.IMREAD_COLOR)
                elif '__ndarray__' in v:
                    obs[k] = np.load(
                        io.BytesIO(v['data']), allow_pickle=False)
                else:
                    obs[k] = v
            elif isinstance(v, bytes):
                obs[k] = v.decode()
            else:
                obs[k] = v
        return obs


# =========================================================================
# Action tensor serialization
# =========================================================================
def serialize_actions(actions: torch.Tensor) -> bytes:
    buf = io.BytesIO()
    np.save(buf, actions.cpu().numpy(), allow_pickle=False)
    return buf.getvalue()


# =========================================================================
# PolicyServer: generic ZMQ REP event loop (Layer 1)
# =========================================================================
@dataclass
class EndpointHandler:
    handler: Callable
    requires_input: bool = True


class PolicyServer:
    """Generic ZMQ REP server with endpoint routing."""

    def __init__(self, host: str = '*', port: int = 5555):
        self.running = True
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f'tcp://{host}:{port}')
        self._endpoints: dict[str, EndpointHandler] = {}

        self.register_endpoint(
            'ping', self._handle_ping, requires_input=False)
        self.register_endpoint(
            'kill', self._kill_server, requires_input=False)

    def register_endpoint(self, name: str, handler: Callable,
                          requires_input: bool = True):
        self._endpoints[name] = EndpointHandler(handler, requires_input)

    def _handle_ping(self) -> dict:
        return {'status': 'ok', 'message': 'Server is running'}

    def _kill_server(self):
        self.running = False
        return {'status': 'ok', 'message': 'Server shutting down'}

    def run(self):
        addr = self.socket.getsockopt_string(zmq.LAST_ENDPOINT)
        print(f'Server is ready and listening on {addr}')
        poller = zmq.Poller()
        poller.register(self.socket, zmq.POLLIN)
        while self.running:
            try:
                socks = dict(poller.poll(timeout=500))
                if self.socket not in socks:
                    continue

                message = self.socket.recv()

                if detect_format(message) == FORMAT_PROTOBUF:
                    self._handle_protobuf_predict(message)
                    continue

                request = MsgSerializer.from_bytes(message)
                endpoint = request.get('endpoint', 'predict_action')
                if endpoint not in self._endpoints:
                    raise ValueError(f'Unknown endpoint: {endpoint}')

                handler = self._endpoints[endpoint]
                result = (
                    handler.handler(**request.get('data', {}))
                    if handler.requires_input else handler.handler())
                self.socket.send(MsgSerializer.to_bytes(result))
            except Exception as e:
                print(f'Error in server: {e}')
                self.socket.send(
                    MsgSerializer.to_bytes({'error': str(e)}))

        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.close()
        self.context.term()

    def _handle_protobuf_predict(self, message: bytes):
        try:
            _, obs, unnorm_key = decode_predict_request(message)
            handler = self._endpoints.get('predict_action')
            if handler is None:
                resp = encode_predict_response(
                    b'', 0.0, FORMAT_PROTOBUF,
                    error='predict_action endpoint not registered')
            else:
                result = handler.handler(
                    obs_data=None, unnorm_key=unnorm_key,
                    _obs_dict=obs, _wire_format=FORMAT_PROTOBUF)
                resp = encode_predict_response(
                    result.get('action_data', b''),
                    result.get('infer_time', 0.0),
                    FORMAT_PROTOBUF,
                    error=result.get('error', ''))
            self.socket.send(resp)
        except Exception as e:
            print(f'Error in protobuf handler: {e}')
            self.socket.send(encode_predict_response(
                b'', 0.0, FORMAT_PROTOBUF, error=str(e)))

    def close(self):
        self.running = False


# =========================================================================
# VLA handler + server factory (Layer 2)
# =========================================================================
def create_server(
    vla,
    dataset=None,
    denormalize_action=None,
    task_suite_name: str = '',
    host: str = '*',
    port: int = 5555,
    device: str = 'cuda:0',
    mixed_precision_dtype=torch.bfloat16,
) -> PolicyServer:
    """Create a ZMQ server that wraps a VLA model.

    Args:
        vla: VLA model (already loaded with weights).
        dataset: Optional dataset transform pipeline for preprocessing.
        denormalize_action: Optional denormalization transform.
        task_suite_name: Task suite name for denormalization lookup.
        host: Bind address.
        port: Bind port.
        device: CUDA device.
        mixed_precision_dtype: Dtype for autocast.
    """
    torch_device = torch.device(device)
    vla.eval()
    vla.to(torch_device)

    lock = threading.Lock()
    total_requests = 0
    total_infer_time = 0.0
    start_time = time.time()

    def predict_action(obs_data: bytes = None, unnorm_key: str = '',
                       _obs_dict: dict = None,
                       _wire_format: int = 0) -> dict:
        nonlocal total_requests, total_infer_time

        if _obs_dict is not None:
            obs = _obs_dict
        else:
            obs = ObsSerializer.from_bytes(obs_data)

        if dataset is not None:
            result = dataset(obs)
            batch = result[0] if isinstance(result, tuple) else result
        else:
            batch = obs
        if unnorm_key:
            batch['unnorm_key'] = unnorm_key

        t0 = time.perf_counter()
        with torch.no_grad(), torch.autocast(
                'cuda', dtype=mixed_precision_dtype, enabled=True):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(torch_device)
            actions = vla.predict_action(**batch)
        infer_time = time.perf_counter() - t0

        if denormalize_action is not None:
            actions_np = actions.cpu().numpy()
            d = denormalize_action(dict(
                action=actions_np[0],
                task_suite_name=task_suite_name))
            actions = torch.from_numpy(d[None].astype(np.float32))

        action_bytes = serialize_actions(actions)

        with lock:
            total_requests += 1
            total_infer_time += infer_time
            n = total_requests
            should_print = (n % 50 == 0)
            avg = total_infer_time / n if should_print else 0.0

        if should_print:
            print(f'[VLAServer] req={n}  '
                  f'infer={infer_time*1000:.1f}ms  '
                  f'avg_infer={avg*1000:.1f}ms', flush=True)

        return {'action_data': action_bytes, 'infer_time': infer_time}

    def reset() -> dict:
        return {'status': 'ok'}

    def get_status() -> dict:
        with lock:
            n = total_requests
            avg = (total_infer_time / n) if n > 0 else 0.0
        return {
            'status': 'ready',
            'uptime_s': time.time() - start_time,
            'total_requests': n,
            'avg_infer_time': avg,
        }

    server = PolicyServer(host=host, port=port)
    server.register_endpoint('predict_action', predict_action)
    server.register_endpoint('reset', reset, requires_input=False)
    server.register_endpoint(
        'get_status', get_status, requires_input=False)
    return server
