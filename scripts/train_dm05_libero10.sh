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

OPENDM_ROOT="${OPENDM_ROOT:-/root/code/opendm}"
if [[ ! -f "${OPENDM_ROOT}/script/dm05_launcher.sh" ]]; then
  echo "Error: OpenDM checkout not found at ${OPENDM_ROOT}" >&2
  exit 1
fi

export PYTHONPATH="${OPENDM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"

DATA_ROOT="${DATA_ROOT:-/mnt/data/stable/users/sober/fluxvla_source_data/libero_10_no_noops_lerobotv2.1}"
DM05_JSONL_DIR="${DM05_JSONL_DIR:-/mnt/data/stable/users/sober/fluxvla_source_data/libero_10_no_noops_opendm}"
DM05_IMAGE_DIR="${DM05_IMAGE_DIR:-${DATA_ROOT}/videos}"
DM05_CKPT="${DM05_CKPT:-/mnt/data/stable/users/sober/pretrain_checkpoint/DM05}"
NORM_STATS_ROOT="${NORM_STATS_ROOT:-/mnt/data/stable/users/sober/fluxvla_work_dirs/dm05_norm_stats}"

RUN_NAME="${RUN_NAME:-dm05_libero10_lora_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/data/stable/users/sober/fluxvla_work_dirs}"
WORK_DIR="${WORK_DIR:-${OUTPUT_ROOT}/${RUN_NAME}}"
TRAINABLE_SUMMARY_PATH="${TRAINABLE_SUMMARY_PATH:-${WORK_DIR}/trainable_summary.json}"

NPROC_PER_NODE="${NPROC_PER_NODE:-${MLP_WORKER_GPU:-1}}"
NNODES="${NNODES:-${WORLD_SIZE:-${MLP_WORKER_NUM:-1}}}"
NODE_RANK="${NODE_RANK:-${RANK:-${MLP_ROLE_INDEX:-0}}}"
MASTER_ADDR="${MASTER_ADDR:-${MLP_WORKER_0_HOST:-localhost}}"
MASTER_PORT="${MASTER_PORT:-${MLP_WORKER_0_PORT:-29503}}"

PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-4}"
GRAD_ACCUMULATION_STEPS="${GRAD_ACCUMULATION_STEPS:-1}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-50000}"
SAVE_STEPS="${SAVE_STEPS:-10000}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
CHUNK_SIZE="${CHUNK_SIZE:-10}"

if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "Error: LIBERO-10 LeRobot dataset not found at ${DATA_ROOT}" >&2
  exit 1
fi
if [[ ! -f "${DM05_CKPT}/config.json" ]]; then
  echo "Error: DM05 checkpoint not found at ${DM05_CKPT}" >&2
  echo "Download it with: hf download Dexmal/DM05 --local-dir ${DM05_CKPT}" >&2
  exit 1
fi

mkdir -p "${WORK_DIR}" "${NORM_STATS_ROOT}"

python scripts/prepare_dm05_libero10_data.py \
  --source-root "${DATA_ROOT}" \
  --output-root "${DM05_JSONL_DIR}"

echo "Starting DM05 LIBERO-10 LoRA training"
echo "  OpenDM root: ${OPENDM_ROOT}"
echo "  LeRobot data: ${DATA_ROOT}"
echo "  OpenDM jsonl: ${DM05_JSONL_DIR}"
echo "  OpenDM image/video dir: ${DM05_IMAGE_DIR}"
echo "  DM05 checkpoint: ${DM05_CKPT}"
echo "  work_dir: ${WORK_DIR}"
echo "  nproc_per_node: ${NPROC_PER_NODE}"
echo "  nnodes: ${NNODES}"
echo "  node_rank: ${NODE_RANK}"
echo "  master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "  per_device_batch_size: ${PER_DEVICE_BATCH_SIZE}"
echo "  grad_accumulation_steps: ${GRAD_ACCUMULATION_STEPS}"
echo "  num_train_steps: ${NUM_TRAIN_STEPS}"

exec bash "${OPENDM_ROOT}/script/dm05_launcher.sh" \
  --exp "${OPENDM_ROOT}/playground/dm05_libero_lora.py" \
  --nproc_per_node "${NPROC_PER_NODE}" \
  --nnodes "${NNODES}" \
  --node_rank "${NODE_RANK}" \
  --master_addr "${MASTER_ADDR}" \
  --master_port "${MASTER_PORT}" \
  -- \
  --task train \
  --data-config.dataset-name libero_pi0_all \
  --data-config.jsonl-dir "${DM05_JSONL_DIR}" \
  --data-config.image-dir "${DM05_IMAGE_DIR}" \
  --data-config.norm-stats-root "${NORM_STATS_ROOT}" \
  --model-config.model-name-or-path "${DM05_CKPT}" \
  --model-config.chunk-size "${CHUNK_SIZE}" \
  --model-config.lora-config.dump-trainable-path "${TRAINABLE_SUMMARY_PATH}" \
  --trainer-config.output-dir "${WORK_DIR}" \
  --trainer-config.num-train-steps "${NUM_TRAIN_STEPS}" \
  --trainer-config.save-steps "${SAVE_STEPS}" \
  --trainer-config.per-device-train-batch-size "${PER_DEVICE_BATCH_SIZE}" \
  --trainer-config.gradient-accumulation-steps "${GRAD_ACCUMULATION_STEPS}" \
  --trainer-config.dataloader-num-workers "${DATALOADER_NUM_WORKERS}" \
  "$@"
