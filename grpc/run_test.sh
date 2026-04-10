#!/bin/bash
# Standalone build & test script for VLA gRPC service.
# Usage: cd grpc && bash run_test.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Step 1: Install dependencies ==="
pip install -r requirements.txt -q

echo "=== Step 2: Compile protobuf ==="
rm -f vla_service_pb2.py vla_service_pb2_grpc.py
python -m grpc_tools.protoc \
    -I. \
    --python_out=. \
    --grpc_python_out=. \
    vla_service.proto
echo "  Generated: vla_service_pb2.py, vla_service_pb2_grpc.py"

echo "=== Step 3: Run tests ==="
python -m pytest test_grpc.py -v --tb=short 2>/dev/null || python -m unittest test_grpc.TestVLAGRPC -v

echo ""
echo "=== All tests passed! ==="
