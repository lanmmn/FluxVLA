#!/usr/bin/env bash
# Launch the original four-step done-dim teacher for all eight WBT prompts.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLUXVLA_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export CONFIG="${CONFIG:-$FLUXVLA_ROOT/configs/gr00t/gr00t_hud04_rtc_done_rtc_kernel_inference.py}"
export MODEL_DIR="${MODEL_DIR:-/data/ckpts/gr00t_rtc_wbt_june_task7_0630_latesttrash_8gpu_20260630_134743_epoch30}"
export CKPT_PATH="${CKPT_PATH:-$MODEL_DIR/checkpoints/step-490560-epoch-30-loss=0.0123.safetensors}"
export TASK_SWITCH="${TASK_SWITCH:-1}"
export META_MODE="${META_MODE:-0}"
export ONLINE_DONE_PLOT="${ONLINE_DONE_PLOT:-0}"
export MROS_IP_LIST="${MROS_IP_LIST:-10.192.1.185}"

exec "$SCRIPT_DIR/inference_gr00t_rtc_wbt_done.sh" "$@"
