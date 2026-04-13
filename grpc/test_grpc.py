"""End-to-end tests for VLA gRPC service (unary, bidirectional stream, status stream, concurrent)."""
from __future__ import annotations
import threading
import time
import unittest
from concurrent import futures

import numpy as np
import vla_service_pb2
import vla_service_pb2_grpc
from grpc_client import GRPCInferClient
from grpc_server import MockInferPipeline, VLAServiceServicer

import grpc

TEST_PORT = 50099
ACTION_DIM = 14
ACTION_HORIZON = 50
DEFAULT_CAMERAS = ["high", "left_hand", "right_hand"]


def _start_test_server(port: int,
                       camera_names: list[str] | None = None) -> grpc.Server:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    pipeline = MockInferPipeline(
        action_dim=ACTION_DIM, action_horizon=ACTION_HORIZON)
    servicer = VLAServiceServicer(pipeline=pipeline, camera_names=camera_names)
    vla_service_pb2_grpc.add_VLAServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"localhost:{port}")
    server.start()
    return server


class TestVLAGRPC(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.server = _start_test_server(TEST_PORT)
        time.sleep(0.3)  # wait for server ready
        cls.client = GRPCInferClient(
            host="localhost", port=TEST_PORT, timeout_s=10.0)

    @classmethod
    def tearDownClass(cls):
        cls.client.close()
        cls.server.stop(grace=1)

    def _make_images(self) -> dict[str, bytes]:
        return {
            "high": b"\xff\xd8" + b"\x00" * 200,
            "left_hand": b"\xff\xd8" + b"\x00" * 100,
            "right_hand": b"\xff\xd8" + b"\x00" * 100,
        }

    def _make_action(self, n: int = 3) -> list[list[float]]:
        return [list(np.random.randn(ACTION_DIM) * 0.01) for _ in range(n)]

    # ---- Test 1: Unary RPC ----

    def test_01_unary_infer(self):
        """Unary inference: single request-response with data integrity check."""
        images = self._make_images()
        action = self._make_action(5)
        ts = time.time()

        result = self.client.infer(images, action, state_delta=2, timestamp=ts)

        self.assertIn("action_list", result)
        self.assertIn("raw_action_list", result)
        self.assertIn("infer_time", result)
        self.assertEqual(len(result["action_list"]), ACTION_HORIZON)
        self.assertEqual(len(result["raw_action_list"]), ACTION_HORIZON)
        self.assertEqual(len(result["action_list"][0]), ACTION_DIM)
        self.assertGreater(result["infer_time"], 0.0)
        print(
            f"  [PASS] unary_infer: {ACTION_HORIZON} actions, infer_time={result['infer_time']:.4f}s"
        )

    # ---- Test 2: Bidirectional Streaming ----

    def test_02_bidirectional_stream(self):
        """Bidirectional streaming: send 5 frames, receive 5 action sets."""
        num_frames = 5
        received = []

        def request_gen():
            for i in range(num_frames):
                yield (self._make_images(), self._make_action(3), i,
                       time.time())

        for resp in self.client.infer_stream(request_gen()):
            received.append(resp)

        self.assertEqual(len(received), num_frames)
        for i, resp in enumerate(received):
            self.assertEqual(resp["sequence_id"], i)
            self.assertEqual(len(resp["action_list"]), ACTION_HORIZON)
            self.assertEqual(len(resp["action_list"][0]), ACTION_DIM)
            self.assertGreater(resp["infer_time"], 0.0)

        print(
            f"  [PASS] bidirectional_stream: sent {num_frames} frames, received {len(received)} responses"
        )

    # ---- Test 3: Server Streaming (Status) ----

    def test_03_status_stream(self):
        """Server streaming: receive 3 status updates."""
        statuses = []
        for status in self.client.status_stream(interval_s=0.2, max_updates=3):
            statuses.append(status)

        self.assertEqual(len(statuses), 3)
        for s in statuses:
            self.assertIn(s["status"], ("ready", "busy", "error"))
            self.assertGreater(s["uptime_s"], 0.0)
            self.assertGreater(s["timestamp"], 0.0)

        # uptime should be increasing
        self.assertGreaterEqual(statuses[2]["uptime_s"],
                                statuses[0]["uptime_s"])
        print(
            f"  [PASS] status_stream: received {len(statuses)} status updates")

    # ---- Test 4: Concurrent Unary Requests ----

    def test_04_concurrent_unary(self):
        """Concurrent unary requests from multiple threads."""
        num_threads = 4
        results = [None] * num_threads
        errors = []

        def worker(idx):
            try:
                client = GRPCInferClient(
                    host="localhost", port=TEST_PORT, timeout_s=10.0)
                result = client.infer(
                    self._make_images(),
                    self._make_action(2),
                    state_delta=idx,
                    timestamp=time.time())
                results[idx] = result
                client.close()
            except Exception as e:
                errors.append((idx, str(e)))

        threads = [
            threading.Thread(target=worker, args=(i, ))
            for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15.0)

        self.assertEqual(
            len(errors), 0, f"Errors in concurrent requests: {errors}")
        for i, r in enumerate(results):
            self.assertIsNotNone(r, f"Thread {i} returned None")
            self.assertEqual(len(r["action_list"]), ACTION_HORIZON)

        print(
            f"  [PASS] concurrent_unary: {num_threads} threads completed successfully"
        )

    # ---- Test 5: Get Camera Config ----

    def test_05_get_camera_config(self):
        """GetCameraConfig: server returns its configured camera list."""
        cameras = self.client.get_camera_config()
        self.assertEqual(sorted(cameras), sorted(DEFAULT_CAMERAS))
        print(f"  [PASS] get_camera_config: {cameras}")

    # ---- Test 6: Camera Mismatch (server-side) ----

    def test_06_camera_mismatch_server(self):
        """Server rejects request with wrong camera set."""
        images = {
            "high": b"\xff\xd8" + b"\x00" * 100
        }  # missing left_hand, right_hand
        action = self._make_action(1)

        with self.assertRaises(grpc.RpcError) as ctx:
            self.client.infer(
                images, action, state_delta=0, timestamp=time.time())
        self.assertEqual(ctx.exception.code(),
                         grpc.StatusCode.INVALID_ARGUMENT)
        print(f"  [PASS] camera_mismatch_server: {ctx.exception.details()}")

    # ---- Test 7: Client-side Camera Validation ----

    def test_07_client_camera_validation(self):
        """Client validates cameras locally before sending."""
        client = GRPCInferClient(
            host="localhost",
            port=TEST_PORT,
            timeout_s=10.0,
            camera_names=["cam_a", "cam_b"],
        )
        images = {"cam_a": b"\xff\xd8\x00", "cam_c": b"\xff\xd8\x00"}
        with self.assertRaises(ValueError) as ctx:
            client.infer(images, self._make_action(1))
        self.assertIn("Camera mismatch", str(ctx.exception))
        client.close()
        print(f"  [PASS] client_camera_validation: {ctx.exception}")

    # ---- Test 8: Sync Camera Config ----

    def test_08_sync_camera_config(self):
        """Client syncs camera config from server, then validates locally."""
        client = GRPCInferClient(
            host="localhost", port=TEST_PORT, timeout_s=10.0)
        self.assertIsNone(client.camera_names)

        names = client.sync_camera_config()
        self.assertEqual(sorted(names), sorted(DEFAULT_CAMERAS))
        self.assertEqual(sorted(client.camera_names), sorted(DEFAULT_CAMERAS))

        # after sync, sending wrong cameras should raise locally
        with self.assertRaises(ValueError):
            client.infer({"wrong": b"\xff\xd8\x00"}, self._make_action(1))

        client.close()
        print(f"  [PASS] sync_camera_config: synced {names}, validation works")


class TestCustomCameraConfig(unittest.TestCase):
    """Test server with a custom (non-default) camera configuration."""

    @classmethod
    def setUpClass(cls):
        cls.custom_cameras = ["front", "back"]
        cls.server = _start_test_server(
            TEST_PORT + 1, camera_names=cls.custom_cameras)
        time.sleep(0.3)
        cls.client = GRPCInferClient(
            host="localhost", port=TEST_PORT + 1, timeout_s=10.0)

    @classmethod
    def tearDownClass(cls):
        cls.client.close()
        cls.server.stop(grace=1)

    def test_01_custom_cameras_infer(self):
        """Infer succeeds with custom 2-camera setup."""
        images = {
            name: b"\xff\xd8" + b"\x00" * 50
            for name in self.custom_cameras
        }
        action = [list(np.random.randn(ACTION_DIM) * 0.01)]
        result = self.client.infer(
            images, action, state_delta=0, timestamp=time.time())
        self.assertEqual(len(result["action_list"]), ACTION_HORIZON)
        print(
            f"  [PASS] custom_cameras_infer: 2 cameras, {ACTION_HORIZON} actions"
        )

    def test_02_custom_cameras_config(self):
        """GetCameraConfig returns custom camera list."""
        cameras = self.client.get_camera_config()
        self.assertEqual(sorted(cameras), sorted(self.custom_cameras))
        print(f"  [PASS] custom_cameras_config: {cameras}")

    def test_03_custom_cameras_reject_default(self):
        """Server rejects the default 3-camera set when configured for 2 cameras."""
        images = {
            "high": b"\xff\xd8" + b"\x00" * 50,
            "left_hand": b"\xff\xd8" + b"\x00" * 50,
            "right_hand": b"\xff\xd8" + b"\x00" * 50,
        }
        action = [list(np.random.randn(ACTION_DIM) * 0.01)]
        with self.assertRaises(grpc.RpcError) as ctx:
            self.client.infer(
                images, action, state_delta=0, timestamp=time.time())
        self.assertEqual(ctx.exception.code(),
                         grpc.StatusCode.INVALID_ARGUMENT)
        print(
            f"  [PASS] custom_cameras_reject_default: {ctx.exception.details()}"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
