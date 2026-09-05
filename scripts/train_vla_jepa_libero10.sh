#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
cd "${REPO_ROOT}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "fluxvla" ]]; then
  if command -v conda >/dev/null 2>&1; then
    exec conda run --no-capture-output -n fluxvla bash "$0" "$@"
  fi
  echo "Error: activate the fluxvla conda environment or make conda available." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export MASTER_PORT="${MASTER_PORT:-29502}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

CONFIG="${CONFIG:-configs/vla_jepa/vla_jepa_qwen3vl_2b_libero_10_finetune.py}"
DATA_ROOT="${DATA_ROOT:-/mnt/data/stable/users/sober/fluxvla_source_data/libero_10_no_noops_lerobotv2.1}"
PRETRAIN_ROOT="${PRETRAIN_ROOT:-/mnt/data/stable/users/sober/pretrain_checkpoint}"
QWEN3VL_PATH="${QWEN3VL_PATH:-${PRETRAIN_ROOT}/Qwen3-VL-2B-Instruct}"
VJEPA2_PATH="${VJEPA2_PATH:-${PRETRAIN_ROOT}/vjepa2-vitl-fpc64-256}"
RUN_NAME="${RUN_NAME:-vla_jepa_qwen3vl_2b_libero_10_$(date +%Y%m%d_%H%M%S)}"
WORK_DIR="${WORK_DIR:-work_dirs/${RUN_NAME}}"

PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-4}"
PER_DEVICE_NUM_WORKERS="${PER_DEVICE_NUM_WORKERS:-4}"
GRAD_ACCUMULATION_STEPS="${GRAD_ACCUMULATION_STEPS:-1}"
MAX_STEPS="${MAX_STEPS:-30000}"
SAVE_ITER_INTERVAL="${SAVE_ITER_INTERVAL:-10000}"
MAX_KEEP_CKPTS="${MAX_KEEP_CKPTS:-3}"

if [[ ! -f "${QWEN3VL_PATH}/config.json" ]]; then
  echo "Error: Qwen3-VL checkpoint not found at ${QWEN3VL_PATH}" >&2
  exit 1
fi
if [[ ! -f "${VJEPA2_PATH}/config.json" ]]; then
  echo "Error: V-JEPA2 checkpoint not found at ${VJEPA2_PATH}" >&2
  exit 1
fi

mkdir -p "${WORK_DIR}"

echo "Starting VLA-JEPA LIBERO-10 training"
echo "  config: ${CONFIG}"
echo "  data: ${DATA_ROOT}"
echo "  qwen3vl: ${QWEN3VL_PATH}"
echo "  vjepa2: ${VJEPA2_PATH}"
echo "  work_dir: ${WORK_DIR}"
echo "  cuda: ${CUDA_VISIBLE_DEVICES}"
echo "  nproc_per_node: ${NPROC_PER_NODE}"
echo "  per_device_batch_size: ${PER_DEVICE_BATCH_SIZE}"
echo "  grad_accumulation_steps: ${GRAD_ACCUMULATION_STEPS}"
echo "  max_steps: ${MAX_STEPS}"

exec bash scripts/train.sh \
  "${CONFIG}" \
  "${WORK_DIR}" \
  --cfg-options \
  "train_dataloader.dataset.datasets.data_root_path=${DATA_ROOT}" \
  "model.tokenizer.model_path=${QWEN3VL_PATH}" \
  "model.vlm_backbone.vlm_path=${QWEN3VL_PATH}" \
  "model.vj_encoder_path=${VJEPA2_PATH}" \
  "inference_model.tokenizer.model_path=${QWEN3VL_PATH}" \
  "inference_model.vlm_backbone.vlm_path=${QWEN3VL_PATH}" \
  "inference_model.vj_encoder_path=${VJEPA2_PATH}" \
  "train_dataloader.dataset.datasets.transforms.1.video_processor_path=${VJEPA2_PATH}" \
  "train_dataloader.dataset.datasets.transforms.3.tokenizer.model_path=${QWEN3VL_PATH}" \
  "eval.dataset.transforms.4.tokenizer.model_path=${QWEN3VL_PATH}" \
  "train_dataloader.per_device_batch_size=${PER_DEVICE_BATCH_SIZE}" \
  "train_dataloader.per_device_num_workers=${PER_DEVICE_NUM_WORKERS}" \
  "runner.tokenizer.model_path=${QWEN3VL_PATH}" \
  "runner.grad_accumulation_steps=${GRAD_ACCUMULATION_STEPS}" \
  "runner.max_steps=${MAX_STEPS}" \
  "runner.save_iter_interval=${SAVE_ITER_INTERVAL}" \
  "runner.max_keep_ckpts=${MAX_KEEP_CKPTS}" \
  "runner.metric.active_trackers=[jsonl]" \
  "$@"
