"""gRPC client for VLA inference service."""
from __future__ import annotations

import time
from typing import Generator, Iterator

import grpc

import vla_service_pb2
import vla_service_pb2_grpc


class GRPCInferClient:
    """Client for VLAService gRPC server.

    Supports unary inference, bidirectional streaming, and status monitoring.
    """

    def __init__(self, host: str = "localhost", port: int = 50051, timeout_s: float = 10.0,
                 camera_names: list[str] | None = None):
        self._address = f"{host}:{port}"
        self._timeout_s = timeout_s
        self._channel = grpc.insecure_channel(self._address)
        self._stub = vla_service_pb2_grpc.VLAServiceStub(self._channel)
        self._camera_names: list[str] | None = camera_names

    def close(self):
        self._channel.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @property
    def camera_names(self) -> list[str] | None:
        return self._camera_names

    def get_camera_config(self) -> list[str]:
        """Fetch the server's expected camera names."""
        req = vla_service_pb2.CameraConfigRequest()
        resp = self._stub.GetCameraConfig(req, timeout=self._timeout_s)
        return list(resp.camera_names)

    def sync_camera_config(self) -> list[str]:
        """Fetch server camera config and store it locally. Returns camera names."""
        self._camera_names = self.get_camera_config()
        return self._camera_names

    def _validate_cameras(self, image_keys: list[str]):
        if self._camera_names is None:
            return
        expected = set(self._camera_names)
        received = set(image_keys)
        missing = expected - received
        extra = received - expected
        if missing or extra:
            parts = []
            if missing:
                parts.append(f"missing cameras: {sorted(missing)}")
            if extra:
                parts.append(f"unexpected cameras: {sorted(extra)}")
            raise ValueError(
                f"Camera mismatch: {'; '.join(parts)}. Expected: {sorted(expected)}"
            )

    @staticmethod
    def _build_request(images: dict[str, bytes], action: list[list[float]],
                       state_delta: int = 0, timestamp: float = 0.0) -> vla_service_pb2.InferRequest:
        req = vla_service_pb2.InferRequest()
        for name, data in images.items():
            req.images[name] = data
        for act in action:
            av = vla_service_pb2.ActionVector()
            av.values.extend(act)
            req.action.append(av)
        req.state_delta = state_delta
        req.timestamp = timestamp
        return req

    @staticmethod
    def _parse_response(response: vla_service_pb2.InferResponse) -> dict:
        return {
            "action_list": [[v for v in av.values] for av in response.action_list],
            "raw_action_list": [[v for v in av.values] for av in response.raw_action_list],
            "infer_time": response.infer_time,
            "sequence_id": response.sequence_id,
        }

    # --- Unary RPC ---

    def infer(self, images: dict[str, bytes], action: list[list[float]],
              state_delta: int = 0, timestamp: float = 0.0) -> dict:
        """Single request-response inference call."""
        self._validate_cameras(list(images.keys()))
        req = self._build_request(images, action, state_delta, timestamp)
        resp = self._stub.Infer(req, timeout=self._timeout_s)
        return self._parse_response(resp)

    # --- Bidirectional Streaming ---

    def infer_stream(self, request_generator: Generator) -> Iterator[dict]:
        """Bidirectional streaming inference.

        Args:
            request_generator: yields (images, action, state_delta, timestamp) tuples.

        Yields:
            Parsed response dicts for each frame.
        """
        def _gen():
            for images, action, state_delta, ts in request_generator:
                self._validate_cameras(list(images.keys()))
                yield self._build_request(images, action, state_delta, ts)

        responses = self._stub.InferStream(_gen())
        for resp in responses:
            yield self._parse_response(resp)

    # --- Server Streaming ---

    def status_stream(self, interval_s: float = 1.0, max_updates: int = 0) -> Iterator[dict]:
        """Subscribe to server status updates.

        Args:
            interval_s: desired update interval.
            max_updates: stop after this many updates (0 = unlimited).

        Yields:
            Status dicts.
        """
        req = vla_service_pb2.StatusRequest(interval_s=interval_s)
        count = 0
        for status in self._stub.StatusStream(req):
            yield {
                "status": status.status,
                "uptime_s": status.uptime_s,
                "total_requests": status.total_requests,
                "avg_infer_time": status.avg_infer_time,
                "timestamp": status.timestamp,
            }
            count += 1
            if 0 < max_updates <= count:
                break


if __name__ == "__main__":
    import numpy as np

    with GRPCInferClient() as client:
        # Unary test
        images = {"high": b"\xff\xd8" + b"\x00" * 100}
        action = [list(np.zeros(14))]
        result = client.infer(images, action, state_delta=0, timestamp=time.time())
        print(f"Unary result: {len(result['action_list'])} actions, infer_time={result['infer_time']:.4f}s")

        # Bidirectional streaming test
        def gen_requests():
            for i in range(3):
                yield ({"high": b"\xff\xd8" + b"\x00" * 50},
                       [list(np.zeros(14))], 0, time.time())

        for resp in client.infer_stream(gen_requests()):
            print(f"Stream response seq={resp['sequence_id']}, actions={len(resp['action_list'])}")
