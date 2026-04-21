#!/usr/bin/env bash
# Build FluxVLA Docker images.
#
# Usage:
#   bash docker/scripts/build.sh a100            # A100 full environment (train+eval+infer)
#   bash docker/scripts/build.sh server          # x86 GPU inference server only
#   bash docker/scripts/build.sh orin            # Orin full local inference
#   bash docker/scripts/build.sh orin-client     # Orin lightweight ZMQ client
#   bash docker/scripts/build.sh all             # build all images
#
# Optional environment variables:
#   JETPACK_TAG    JetPack base image tag (default: r35.4.1)
#   TORCH_WHL_URL  PyTorch wheel URL for Jetson

set -euo pipefail
cd "$(dirname "$0")/../.."

JETPACK_TAG="${JETPACK_TAG:-r35.4.1}"
TAG_SUFFIX="${TAG_SUFFIX:-latest}"

build_server() {
    echo "==> Building fluxvla-server (x86 GPU) ..."
    docker build \
        -f docker/Dockerfile.server \
        -t "fluxvla-server:${TAG_SUFFIX}" \
        .
    echo "==> Done: fluxvla-server:${TAG_SUFFIX}"
}

build_orin() {
    echo "==> Building fluxvla-orin (full inference, JetPack=${JETPACK_TAG}) ..."
    local extra_args=()
    [ -n "${TORCH_WHL_URL:-}" ] && extra_args+=(--build-arg "TORCH_WHL_URL=${TORCH_WHL_URL}")

    docker build \
        -f docker/Dockerfile.orin \
        --build-arg "JETPACK_TAG=${JETPACK_TAG}" \
        "${extra_args[@]}" \
        -t "fluxvla-orin:${TAG_SUFFIX}" \
        .
    echo "==> Done: fluxvla-orin:${TAG_SUFFIX}"
}

build_orin_client() {
    echo "==> Building fluxvla-orin-client (lightweight ZMQ client, JetPack=${JETPACK_TAG}) ..."
    local extra_args=()
    [ -n "${TORCH_WHL_URL:-}" ] && extra_args+=(--build-arg "TORCH_WHL_URL=${TORCH_WHL_URL}")

    docker build \
        -f docker/Dockerfile.orin-client \
        --build-arg "JETPACK_TAG=${JETPACK_TAG}" \
        "${extra_args[@]}" \
        -t "fluxvla-orin-client:${TAG_SUFFIX}" \
        .
    echo "==> Done: fluxvla-orin-client:${TAG_SUFFIX}"
}

build_a100() {
    echo "==> Building fluxvla-a100 (full train+eval+infer environment) ..."
    docker build \
        -f docker/Dockerfile.a100 \
        -t "fluxvla-a100:${TAG_SUFFIX}" \
        .
    echo "==> Done: fluxvla-a100:${TAG_SUFFIX}"
}

case "${1:-help}" in
    a100)         build_a100 ;;
    server)       build_server ;;
    orin)         build_orin ;;
    orin-client)  build_orin_client ;;
    all)
        build_a100
        build_server
        build_orin
        build_orin_client
        ;;
    *)
        echo "Usage: $0 {a100|server|orin|orin-client|all}"
        echo ""
        echo "  a100         A100 full environment (train + eval + inference)"
        echo "  server       x86 GPU inference server (ZMQ PolicyServer)"
        echo "  orin         Jetson Orin full local inference"
        echo "  orin-client  Jetson Orin lightweight remote client"
        echo "  all          Build all images"
        exit 1
        ;;
esac
