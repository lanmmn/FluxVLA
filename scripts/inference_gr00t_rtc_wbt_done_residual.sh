#!/usr/bin/env bash
# Launch the selected done-dim residual student for all eight WBT prompts.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLUXVLA_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export CONFIG="${CONFIG:-$FLUXVLA_ROOT/configs/gr00t/gr00t_hud04_rtc_done_rtc_kernel_inference_residual.py}"
export MODEL_DIR="${MODEL_DIR:-/data/ckpts/gr00t_rtc_wbt_june_done_4to2_residual_stage1_20260727_1205}"
export CKPT_PATH="${CKPT_PATH:-$MODEL_DIR/checkpoints/step-000350-epoch-00-loss=0.0038.safetensors}"
export TASK_SWITCH="${TASK_SWITCH:-1}"
export META_MODE="${META_MODE:-0}"
export ONLINE_DONE_PLOT="${ONLINE_DONE_PLOT:-0}"
export MROS_IP_LIST="${MROS_IP_LIST:-10.192.1.185}"

exec "$SCRIPT_DIR/inference_gr00t_rtc_wbt_done.sh" "$@"
