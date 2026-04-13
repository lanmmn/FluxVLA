"""gRPC server for VLA inference service."""
from __future__ import annotations
import io
import threading
import time
from concurrent import futures

import numpy as np
import torch
import vla_service_pb2
import vla_service_pb2_grpc

import grpc

_MAX_MSG_BYTES = 64 * 1024 * 1024  # 64 MB


class MockInferPipeline:
    """Mock inference pipeline that generates dummy actions for testing."""

    def __init__(self, action_dim: int = 14, action_horizon: int = 50):
        self._action_dim = action_dim
        self._action_horizon = action_horizon

    def infer(self, images: dict[str, bytes], action: list[list[float]],
              state_delta: int, timestamp: float) -> dict:
        time.sleep(0.01)  # simulate inference latency
        raw_actions = np.random.randn(self._action_horizon,
                                      self._action_dim) * 0.1
        optimized_actions = raw_actions * 0.95  # mock optimization
        return {
            "action_list": optimized_actions.tolist(),
            "raw_action_list": raw_actions.tolist(),
        }


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
    def predict_action(self, **batch):
        # Move all tensors to the model device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self._device)

        with torch.autocast("cuda", dtype=self._dtype, enabled=True):
            actions = self._vla.predict_action(**batch)
        return actions


class VLAServiceServicer(vla_service_pb2_grpc.VLAServiceServicer):
    """Implements the VLAService gRPC interface."""

    def __init__(self, pipeline=None, camera_names: list[str] | None = None,
                 vla_pipeline: VLAInferPipeline | None = None):
        self._pipeline = pipeline or MockInferPipeline()
        self._vla_pipeline = vla_pipeline
        self._camera_names = camera_names or [
            "high", "left_hand", "right_hand"
        ]
        self._start_time = time.time()
        self._total_requests = 0
        self._total_infer_time = 0.0
        self._lock = threading.Lock()

    def _validate_cameras(self, image_keys: list[str], context) -> bool:
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
            msg = f"Camera mismatch: {'; '.join(parts)}. Expected: {sorted(expected)}"
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, msg)
            return False
        return True

    def _parse_request(self, request):
        images = dict(request.images)
        action = [[v for v in av.values] for av in request.action]
        return images, action, request.state_delta, request.timestamp

    def _build_response(self,
                        result: dict,
                        sequence_id: int = 0) -> vla_service_pb2.InferResponse:
        response = vla_service_pb2.InferResponse()
        for action in result["action_list"]:
            av = vla_service_pb2.ActionVector()
            av.values.extend(action)
            response.action_list.append(av)
        for action in result.get("raw_action_list", []):
            av = vla_service_pb2.ActionVector()
            av.values.extend(action)
            response.raw_action_list.append(av)
        response.infer_time = result.get("infer_time", 0.0)
        response.sequence_id = sequence_id
        return response

    def GetCameraConfig(self, request, context):
        """Return the server's expected camera configuration."""
        return vla_service_pb2.CameraConfig(camera_names=self._camera_names)

    def TensorBatchInfer(self, request, context):
        """Tensor-batch inference: receive serialized batch, return actions."""
        if self._vla_pipeline is None:
            context.abort(
                grpc.StatusCode.UNIMPLEMENTED,
                "TensorBatchInfer requires a VLAInferPipeline; "
                "start the server with --config and --ckpt-path.")
            return vla_service_pb2.TensorBatchInferResponse()

        t0 = time.perf_counter()
        try:
            batch = torch.load(io.BytesIO(request.batch_data),
                               map_location="cpu", weights_only=True)
        except Exception as e:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT,
                          f"Failed to deserialize batch: {e}")
            return vla_service_pb2.TensorBatchInferResponse()
        t_deserialize = time.perf_counter() - t0

        if request.unnorm_key:
            batch["unnorm_key"] = request.unnorm_key

        t1 = time.perf_counter()
        actions = self._vla_pipeline.predict_action(**batch)
        infer_time = time.perf_counter() - t1

        t2 = time.perf_counter()
        buf = io.BytesIO()
        torch.save(actions.cpu(), buf)
        t_serialize = time.perf_counter() - t2

        with self._lock:
            self._total_requests += 1
            self._total_infer_time += infer_time

            if self._total_requests % 50 == 0:
                n = self._total_requests
                avg = self._total_infer_time / n
                print(f"[TensorBatchInfer] req={n}  "
                      f"deserialize={t_deserialize*1000:.1f}ms  "
                      f"infer={infer_time*1000:.1f}ms  "
                      f"serialize={t_serialize*1000:.1f}ms  "
                      f"avg_infer={avg*1000:.1f}ms",
                      flush=True)

        return vla_service_pb2.TensorBatchInferResponse(
            action_data=buf.getvalue(),
            infer_time=infer_time,
        )

    def Infer(self, request, context):
        """Unary RPC: single request-response."""
        images, action, state_delta, timestamp = self._parse_request(request)
        self._validate_cameras(list(images.keys()), context)

        start = time.time()
        result = self._pipeline.infer(images, action, state_delta, timestamp)
        infer_time = time.time() - start
        result["infer_time"] = infer_time

        with self._lock:
            self._total_requests += 1
            self._total_infer_time += infer_time

        return self._build_response(result)

    def InferStream(self, request_iterator, context):
        """Bidirectional streaming: receive observation frames, yield action frames."""
        seq_id = 0
        for request in request_iterator:
            if context.is_active():
                images, action, state_delta, timestamp = self._parse_request(
                    request)
                self._validate_cameras(list(images.keys()), context)

                start = time.time()
                result = self._pipeline.infer(images, action, state_delta,
                                              timestamp)
                infer_time = time.time() - start
                result["infer_time"] = infer_time

                with self._lock:
                    self._total_requests += 1
                    self._total_infer_time += infer_time

                yield self._build_response(result, sequence_id=seq_id)
                seq_id += 1

    def StatusStream(self, request, context):
        """Server streaming: push periodic status updates."""
        interval = max(0.1,
                       request.interval_s) if request.interval_s > 0 else 1.0

        while context.is_active():
            with self._lock:
                total = self._total_requests
                avg = (self._total_infer_time / total) if total > 0 else 0.0

            status = vla_service_pb2.ServerStatus(
                status="ready",
                uptime_s=time.time() - self._start_time,
                total_requests=total,
                avg_infer_time=avg,
                timestamp=time.time(),
            )
            yield status
            time.sleep(interval)


def serve(host: str = "0.0.0.0",
          port: int = 50051,
          max_workers: int = 4,
          pipeline=None,
          camera_names: list[str] | None = None,
          vla_pipeline: VLAInferPipeline | None = None) -> grpc.Server:
    """Create and start a gRPC server. Returns the server instance."""
    opts = [
        ("grpc.max_send_message_length", _MAX_MSG_BYTES),
        ("grpc.max_receive_message_length", _MAX_MSG_BYTES),
    ]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers),
                         options=opts)
    servicer = VLAServiceServicer(pipeline=pipeline,
                                  camera_names=camera_names,
                                  vla_pipeline=vla_pipeline)
    vla_service_pb2_grpc.add_VLAServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"{host}:{port}")
    server.start()
    print(f"[grpc_server] listening on {host}:{port}")
    return server


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=None,
        help="expected camera names, e.g. --cameras high left_hand right_hand")
    args = parser.parse_args()

    server = serve(
        host=args.host,
        port=args.port,
        max_workers=args.workers,
        camera_names=args.cameras)
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(grace=2)
        print("Server stopped.")
