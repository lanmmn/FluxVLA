"""Wire-format serializers for ZMQ transport.

Two formats are supported for the predict_action hot path:

- **msgpack** (default): flexible dict-based, zero schema
- **protobuf**: schema-driven, optimal for cross-language clients (C++/Rust)

Format detection
~~~~~~~~~~~~~~~~
The first byte of every ZMQ message indicates the format:

- ``0x01`` → protobuf
- anything else → msgpack (legacy clients without prefix are also handled)

Non-predict endpoints (ping, kill, get_status) always use msgpack
regardless of the client's serializer choice.
"""
from __future__ import annotations
import io
from typing import Literal

import cv2
import msgpack
import numpy as np

FORMAT_MSGPACK: int = 0x00
FORMAT_PROTOBUF: int = 0x01


def detect_format(raw: bytes) -> int:
    """Detect wire format from the first byte of a ZMQ message."""
    if raw and raw[0] == FORMAT_PROTOBUF:
        return FORMAT_PROTOBUF
    return FORMAT_MSGPACK


class ObsSerializerProto:
    """Serialize/deserialize raw observation dicts using protobuf.

    Same JPEG + npy payload encoding as :class:`ObsSerializer`, but the
    container is a protobuf ``Observation`` message instead of msgpack.
    """

    JPEG_QUALITY = 95
    JPEG_KEYS = {
        'cam_high',
        'cam_left_wrist',
        'cam_right_wrist',
        'agentview_image',
        'robot0_eye_in_hand_image',
    }

    @staticmethod
    def obs_to_proto(obs: dict, compress: bool = True):
        """Convert an observation dict to a protobuf ``Observation``.

        Args:
            obs: observation dict with numpy arrays and strings.
            compress: if True (default), JPEG-compress RGB images in
                :attr:`JPEG_KEYS`; if False, all arrays use lossless npy.
        """
        from .proto import vla_service_pb2 as pb

        msg = pb.Observation()
        for k, v in obs.items():
            if (compress and isinstance(v, np.ndarray) and v.ndim == 3
                    and v.dtype == np.uint8
                    and k in ObsSerializerProto.JPEG_KEYS):
                _, jpg = cv2.imencode('.jpg', v, [
                    cv2.IMWRITE_JPEG_QUALITY, ObsSerializerProto.JPEG_QUALITY
                ])
                msg.images[k] = jpg.tobytes()
            elif isinstance(v, np.ndarray):
                buf = io.BytesIO()
                np.save(buf, v, allow_pickle=False)
                msg.arrays[k] = buf.getvalue()
            elif isinstance(v, str):
                msg.strings[k] = v
        return msg

    @staticmethod
    def obs_from_proto(msg) -> dict:
        """Convert a protobuf ``Observation`` back to an obs dict."""
        obs: dict = {}
        for k, v in msg.images.items():
            obs[k] = cv2.imdecode(np.frombuffer(v, np.uint8), cv2.IMREAD_COLOR)
        for k, v in msg.arrays.items():
            obs[k] = np.load(io.BytesIO(v), allow_pickle=False)
        for k, v in msg.strings.items():
            obs[k] = v
        return obs


def encode_predict_request(
    obs: dict,
    unnorm_key: str,
    fmt: Literal['msgpack', 'protobuf'] = 'msgpack',
    compress: bool = True,
) -> bytes:
    """Client : Encode a predict_action request in the chosen wire format.

    Args:
        obs: observation dict.
        unnorm_key: denormalization dataset key.
        fmt: wire format, 'msgpack' or 'protobuf'.
        compress: if True, JPEG-compress RGB images; if False, lossless npy.
    """
    if fmt == 'protobuf':
        from .proto import vla_service_pb2 as pb

        req = pb.PredictActionRequest()
        req.obs.CopyFrom(
            ObsSerializerProto.obs_to_proto(obs, compress=compress))
        req.unnorm_key = unnorm_key
        return bytes([FORMAT_PROTOBUF]) + req.SerializeToString()

    from .server_client import ObsSerializer
    payload = ObsSerializer.to_bytes(obs, compress=compress)
    return msgpack.packb({
        'endpoint': 'predict_action',
        'data': {
            'obs_data': payload,
            'unnorm_key': unnorm_key
        },
    })


def decode_predict_request(raw: bytes) -> tuple[int, dict, str]:
    """Server : Decode a predict_action request, get data, obs.

    Returns:
        (format_tag, obs_dict, unnorm_key)
    """
    fmt = detect_format(raw)
    if fmt == FORMAT_PROTOBUF:
        from .proto import vla_service_pb2 as pb

        req = pb.PredictActionRequest()
        req.ParseFromString(raw[1:])
        obs = ObsSerializerProto.obs_from_proto(req.obs)
        return FORMAT_PROTOBUF, obs, req.unnorm_key

    from .server_client import ObsSerializer
    parsed = msgpack.unpackb(raw, raw=False)
    data = parsed.get('data', {})
    obs = ObsSerializer.from_bytes(data['obs_data'])
    return FORMAT_MSGPACK, obs, data.get('unnorm_key', '')


def encode_predict_response(
    action_data: bytes,
    infer_time: float,
    fmt: int = FORMAT_MSGPACK,
    error: str = '',
) -> bytes:
    """Server : Encode a predict_action response in the given wire format."""
    if fmt == FORMAT_PROTOBUF:
        from .proto import vla_service_pb2 as pb

        resp = pb.PredictActionResponse()
        if error:
            resp.error = error
        else:
            resp.action_data = action_data
            resp.infer_time = infer_time
        return bytes([FORMAT_PROTOBUF]) + resp.SerializeToString()

    if error:
        return msgpack.packb({'error': error})
    return msgpack.packb({
        'action_data': action_data,
        'infer_time': infer_time
    })


def decode_predict_response(
    raw: bytes,
    fmt: int = FORMAT_MSGPACK,
) -> dict:
    """Client : `Decode a predict_action response.

    Returns:
        dict with 'action_data' (bytes), 'infer_time' (float),
        and optionally 'error' (str).
    """
    if fmt == FORMAT_PROTOBUF:
        from .proto import vla_service_pb2 as pb

        resp = pb.PredictActionResponse()
        resp.ParseFromString(raw[1:])
        if resp.error:
            return {'error': resp.error}
        return {
            'action_data': resp.action_data,
            'infer_time': resp.infer_time,
        }

    return msgpack.unpackb(raw, raw=False)
