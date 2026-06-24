#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Override the image with FLUXVLA_IMAGE, for example:
#   FLUXVLA_IMAGE=fluxvla:orin-ros-fa-1.0.0 ./run_docker.sh
IMAGE="${FLUXVLA_IMAGE:-fluxvla:orin-ros-fa}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
TRANSFORMERS_ATTN_IMPLEMENTATION="${TRANSFORMERS_ATTN_IMPLEMENTATION:-${ATTN_IMPLEMENTATION}}"
ROBOTIQ_PY_PKG="${ROBOTIQ_PY_PKG:-${HOME}/sober/robotiq_pkg/robotiq}"

DOCKER_ENV_ARGS=(
    -e PYTHONPATH=/workspace/FluxVLA
    -e WANDB_MODE=disabled
    -e "ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION}"
    -e "TRANSFORMERS_ATTN_IMPLEMENTATION=${TRANSFORMERS_ATTN_IMPLEMENTATION}"
)

if [ -n "${ROS_MASTER_URI:-}" ]; then
    DOCKER_ENV_ARGS+=(-e "ROS_MASTER_URI=${ROS_MASTER_URI}")
fi
if [ -n "${ROS_IP:-}" ]; then
    DOCKER_ENV_ARGS+=(-e "ROS_IP=${ROS_IP}")
fi
if [ -n "${ROS_HOSTNAME:-}" ]; then
    DOCKER_ENV_ARGS+=(-e "ROS_HOSTNAME=${ROS_HOSTNAME}")
fi

DOCKER_VOLUME_ARGS=(
    -v "${REPO_ROOT}:/workspace/FluxVLA"
    -v /mnt/nvme:/mnt/nvme
)

if [ -d "${ROBOTIQ_PY_PKG}" ]; then
    DOCKER_VOLUME_ARGS+=(
        -v "${ROBOTIQ_PY_PKG}:/opt/ros/noetic/lib/python3/dist-packages/robotiq:ro"
    )
fi

if ! docker images --format '{{.Repository}}:{{.Tag}}' | grep -q "^${IMAGE}$"; then
    echo "Error: ${IMAGE} image not found. Build it first with ./build_docker.sh"
    exit 1
fi

docker run --rm -it \
    --runtime=nvidia \
    --ipc=host \
    --network=host \
    --shm-size=16g \
    "${DOCKER_ENV_ARGS[@]}" \
    "${DOCKER_VOLUME_ARGS[@]}" \
    -w /workspace/FluxVLA \
    "${IMAGE}" "${@:-bash}"
