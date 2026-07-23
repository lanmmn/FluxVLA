#!/usr/bin/env bash

# Profile a short VLA-JEPA training window with Nsight Systems and generate
# reusable CSV/text analysis artifacts.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

MODE="${1:-all}"
case "${MODE}" in
  all|profile|analyze|help|-h|--help)
    shift || true
    ;;
  *)
    printf 'Unknown mode: %s\n\n' "${MODE}" >&2
    MODE="help"
    ;;
esac

CONFIG="${CONFIG:-configs/vla_jepa/vla_jepa_qwen3vl_2b_libero_10_finetune.py}"
WORK_DIR="${WORK_DIR:-/tmp/vla_jepa_nsys_profile_run}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/fluxvla_nsys_vla_jepa}"
PYTHON="${PYTHON:-/root/miniconda3/bin/python}"
NSYS_BIN="${NSYS_BIN:-nsys}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-32}"
GRAD_ACCUMULATION_STEPS="${GRAD_ACCUMULATION_STEPS:-1}"
START_STEP="${START_STEP:-10}"
NUM_STEPS="${NUM_STEPS:-3}"
POST_CAPTURE_STEPS="${POST_CAPTURE_STEPS:-1}"
ALL_RANKS="${ALL_RANKS:-0}"
MAX_STEPS="${MAX_STEPS:-$((START_STEP + NUM_STEPS + POST_CAPTURE_STEPS))}"
SAVE_ITER_INTERVAL="${SAVE_ITER_INTERVAL:-10000}"

WANDB_MODE="${WANDB_MODE:-disabled}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
CPU_IO_TRACE="${CPU_IO_TRACE:-0}"
if [[ "${CPU_IO_TRACE}" == "1" ]]; then
  NSYS_SAMPLE="${NSYS_SAMPLE:-process-tree}"
  NSYS_CPUCTXSW="${NSYS_CPUCTXSW:-process-tree}"
  NSYS_PYTHON_SAMPLING="${NSYS_PYTHON_SAMPLING:-true}"
  NSYS_PYTHON_BACKTRACE="${NSYS_PYTHON_BACKTRACE:-cuda}"
  NSYS_OSRT_FILE_ACCESS="${NSYS_OSRT_FILE_ACCESS:-true}"
  NSYS_SYSCALL="${NSYS_SYSCALL:-process-tree}"
else
  NSYS_SAMPLE="${NSYS_SAMPLE:-none}"
  NSYS_CPUCTXSW="${NSYS_CPUCTXSW:-none}"
  NSYS_PYTHON_SAMPLING="${NSYS_PYTHON_SAMPLING:-false}"
  NSYS_PYTHON_BACKTRACE="${NSYS_PYTHON_BACKTRACE:-none}"
  NSYS_OSRT_FILE_ACCESS="${NSYS_OSRT_FILE_ACCESS:-false}"
  NSYS_SYSCALL="${NSYS_SYSCALL:-none}"
fi
NSYS_PYTHON_SAMPLING_FREQUENCY="${NSYS_PYTHON_SAMPLING_FREQUENCY:-100}"
NSYS_OSRT_THRESHOLD_NS="${NSYS_OSRT_THRESHOLD_NS:-1000}"
NSYS_PYTORCH_TRACE="${NSYS_PYTORCH_TRACE:-none}"
NSYS_INLINE_STATS="${NSYS_INLINE_STATS:-false}"
FORCE_EXPORT="${FORCE_EXPORT:-0}"
FORCE_STATS="${FORCE_STATS:-0}"
RUN_ANALYZE_RULES="${RUN_ANALYZE_RULES:-1}"
DRY_RUN="${DRY_RUN:-0}"
TOP_N="${TOP_N:-20}"
RUN_TAG="${RUN_TAG:-vla_jepa_train_$(date +%Y%m%d_%H%M%S)}"

STAT_NAMES=(
  cuda_api_sum
  cuda_gpu_kern_sum
  cuda_kern_exec_sum
  nvtx_pushpop_sum
  nvtx_gpu_proj_sum
  nvtx_kern_sum
  cuda_gpu_kern_by_nvtx
  osrt_sum
  syscall_sum
)
STAT_SPECS=(
  cuda_api_sum
  cuda_gpu_kern_sum:base
  cuda_kern_exec_sum:base
  nvtx_pushpop_sum
  nvtx_gpu_proj_sum
  nvtx_kern_sum:base
  cuda_gpu_kern_sum:nvtx-name:base
  osrt_sum
  syscall_sum
)
ANALYZE_RULES=(
  cuda_api_sync
  cuda_memcpy_async
  cuda_memcpy_sync
  cuda_memset_sync
  gpu_gaps
  gpu_time_util
)

usage() {
  cat <<'EOF'
Usage:
  scripts/profile_vla_jepa_nsys.sh all [CFG_KEY=VALUE ...]
  scripts/profile_vla_jepa_nsys.sh profile [CFG_KEY=VALUE ...]
  scripts/profile_vla_jepa_nsys.sh analyze [REPORT_OR_DIRECTORY ...]

Modes:
  all      Run short training capture, then analyze generated reports (default).
  profile  Run short training capture only.
  analyze  Analyze existing .nsys-rep files. With no path, scan OUTPUT_DIR.

Main environment overrides:
  CONFIG, WORK_DIR, OUTPUT_DIR, PYTHON, NSYS_BIN
  NPROC_PER_NODE=8, CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  PER_DEVICE_BATCH_SIZE=32, GRAD_ACCUMULATION_STEPS=1
  START_STEP=10, NUM_STEPS=3, POST_CAPTURE_STEPS=1, ALL_RANKS=0
  MAX_STEPS=14, SAVE_ITER_INTERVAL=10000
  CPU_IO_TRACE=0, NSYS_PYTORCH_TRACE=none
  TOP_N=20, FORCE_EXPORT=0, FORCE_STATS=0, RUN_ANALYZE_RULES=1, DRY_RUN=0

Examples:
  scripts/profile_vla_jepa_nsys.sh all

  PER_DEVICE_BATCH_SIZE=16 GRAD_ACCUMULATION_STEPS=4 NUM_STEPS=2 \
    scripts/profile_vla_jepa_nsys.sh all

  scripts/profile_vla_jepa_nsys.sh analyze \
    /tmp/fluxvla_nsys_vla_jepa/example.nsys-rep

  DRY_RUN=1 scripts/profile_vla_jepa_nsys.sh profile \
    train_dataloader.per_device_num_workers=8

  CPU_IO_TRACE=1 NUM_STEPS=1 \
    scripts/profile_vla_jepa_nsys.sh all
EOF
}

log() {
  printf '[vla-jepa-nsys] %s\n' "$*"
}

warn() {
  printf '[vla-jepa-nsys] WARNING: %s\n' "$*" >&2
}

die() {
  printf '[vla-jepa-nsys] ERROR: %s\n' "$*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

require_nonnegative_integer() {
  local name="$1"
  local value="$2"
  [[ "${value}" =~ ^[0-9]+$ ]] || die "${name} must be a non-negative integer: ${value}"
}

require_boolean_flag() {
  local name="$1"
  local value="$2"
  [[ "${value}" == "0" || "${value}" == "1" ]] || \
    die "${name} must be 0 or 1: ${value}"
}

require_true_false() {
  local name="$1"
  local value="$2"
  [[ "${value}" == "true" || "${value}" == "false" ]] || \
    die "${name} must be true or false: ${value}"
}

validate_analysis_settings() {
  require_nonnegative_integer TOP_N "${TOP_N}"
  (( TOP_N > 0 )) || die 'TOP_N must be greater than zero.'
  require_boolean_flag FORCE_EXPORT "${FORCE_EXPORT}"
  require_boolean_flag FORCE_STATS "${FORCE_STATS}"
  require_boolean_flag RUN_ANALYZE_RULES "${RUN_ANALYZE_RULES}"
}

print_command() {
  local arg
  printf '  '
  for arg in "$@"; do
    printf '%q ' "${arg}"
  done
  printf '\n'
}

resolve_repo_path() {
  local path="$1"
  if [[ "${path}" = /* ]]; then
    printf '%s\n' "${path}"
  else
    printf '%s\n' "${REPO_ROOT}/${path}"
  fi
}

validate_profile_settings() {
  require_command "${NSYS_BIN}"
  [[ -x "${PYTHON}" ]] || die "Python executable not found: ${PYTHON}"
  [[ -f "$(resolve_repo_path "${CONFIG}")" ]] || die "Config not found: ${CONFIG}"

  require_nonnegative_integer START_STEP "${START_STEP}"
  require_nonnegative_integer NUM_STEPS "${NUM_STEPS}"
  require_nonnegative_integer POST_CAPTURE_STEPS "${POST_CAPTURE_STEPS}"
  require_nonnegative_integer MAX_STEPS "${MAX_STEPS}"
  require_nonnegative_integer NPROC_PER_NODE "${NPROC_PER_NODE}"
  require_nonnegative_integer PER_DEVICE_BATCH_SIZE "${PER_DEVICE_BATCH_SIZE}"
  require_nonnegative_integer GRAD_ACCUMULATION_STEPS "${GRAD_ACCUMULATION_STEPS}"
  require_nonnegative_integer SAVE_ITER_INTERVAL "${SAVE_ITER_INTERVAL}"
  require_nonnegative_integer NSYS_PYTHON_SAMPLING_FREQUENCY \
    "${NSYS_PYTHON_SAMPLING_FREQUENCY}"
  require_nonnegative_integer NSYS_OSRT_THRESHOLD_NS "${NSYS_OSRT_THRESHOLD_NS}"
  require_boolean_flag CPU_IO_TRACE "${CPU_IO_TRACE}"
  require_true_false NSYS_PYTHON_SAMPLING "${NSYS_PYTHON_SAMPLING}"
  require_true_false NSYS_OSRT_FILE_ACCESS "${NSYS_OSRT_FILE_ACCESS}"

  (( NUM_STEPS > 0 )) || die 'NUM_STEPS must be greater than zero.'
  (( NPROC_PER_NODE > 0 )) || die 'NPROC_PER_NODE must be greater than zero.'
  (( MAX_STEPS >= START_STEP + NUM_STEPS )) || \
    die 'MAX_STEPS must reach START_STEP + NUM_STEPS so profiler.stop() runs.'
  [[ "${ALL_RANKS}" == "0" || "${ALL_RANKS}" == "1" ]] || \
    die 'ALL_RANKS must be 0 or 1.'
  [[ "${NSYS_SAMPLE}" =~ ^(none|process-tree|system-wide)$ ]] || \
    die "Invalid NSYS_SAMPLE: ${NSYS_SAMPLE}"
  [[ "${NSYS_CPUCTXSW}" =~ ^(none|process-tree|system-wide)$ ]] || \
    die "Invalid NSYS_CPUCTXSW: ${NSYS_CPUCTXSW}"
  [[ "${NSYS_SYSCALL}" =~ ^(none|process-tree|pid-namespace)$ ]] || \
    die "Invalid NSYS_SYSCALL: ${NSYS_SYSCALL}"
  [[ "${NSYS_PYTHON_BACKTRACE}" =~ ^(none|cuda)$ ]] || \
    die "Invalid NSYS_PYTHON_BACKTRACE: ${NSYS_PYTHON_BACKTRACE}"
}

collect_reports_from_target() {
  local target="$1"
  local report
  if [[ -f "${target}" ]]; then
    [[ "${target}" == *.nsys-rep ]] || die "Not an .nsys-rep file: ${target}"
    REPORTS+=("$(cd -- "$(dirname -- "${target}")" && pwd)/$(basename -- "${target}")")
  elif [[ -d "${target}" ]]; then
    while IFS= read -r -d '' report; do
      REPORTS+=("${report}")
    done < <(find "${target}" -maxdepth 1 -type f -name '*.nsys-rep' -print0 | sort -z)
  else
    die "Report path does not exist: ${target}"
  fi
}

write_filtered_csv() {
  local source="$1"
  local destination="$2"
  local pattern="$3"
  if [[ ! -s "${source}" ]]; then
    : > "${destination}"
    return 0
  fi
  {
    sed -n '1p' "${source}"
    sed -n '2,$p' "${source}" | grep -Ei "${pattern}" || true
  } > "${destination}"
}

generate_optimizer_step_breakdown() {
  local source="$1"
  local destination="$2"

  if [[ ! -s "${source}" ]]; then
    printf 'No NVTX CPU-side data is available.\n' > "${destination}"
    return 0
  fi

  awk -F',' '
    BEGIN {
      phase_count = 12
      phase[1] = "DATA_WAIT"
      phase[2] = "LR_PREPARE"
      phase[3] = "H2D_AND_PREPARE"
      phase[4] = "FORWARD"
      phase[5] = "BACKWARD"
      phase[6] = "GRAD_CLIP"
      phase[7] = "ADAMW_STEP"
      phase[8] = "REINIT_OPTIMIZER"
      phase[9] = "ADAMW_STEP_RETRY"
      phase[10] = "LR_SCHEDULER"
      phase[11] = "ZERO_GRAD"
      phase[12] = "CUSTOM_TRAINING_STEP"
    }
    NR > 1 {
      range = $NF
      sub(/^:/, "", range)
      total = $2 + 0
      instances = $3 + 0
      average = $4 + 0

      if (range ~ /^OPTIMIZER_STEP_/) {
        optimizer_total += total
        optimizer_instances += instances
        optimizer_average_sum += average
        if (instances > inferred_ranks) {
          inferred_ranks = instances
        }
        if (!(range in seen_optimizer_step)) {
          seen_optimizer_step[range] = 1
          logical_steps++
        }
      }
      if (range ~ /^MICRO_[0-9]+\/TRAIN_SYNC_/) {
        train_sync_total += total
        train_sync_instances += instances
      }

      key = range
      if (range ~ /\/DATA_WAIT$/) {
        key = "DATA_WAIT"
      }
      for (i = 1; i <= phase_count; i++) {
        if (key == phase[i]) {
          phase_total[key] += total
          phase_instances[key] += instances
        }
      }
    }
    END {
      if (optimizer_instances == 0) {
        print "No OPTIMIZER_STEP_* NVTX ranges were found."
        exit
      }
      expected_rank_steps = inferred_ranks * logical_steps
      optimizer_average = optimizer_average_sum / logical_steps
      normalized_optimizer_total = optimizer_average * expected_rank_steps

      printf "CPU-side optimizer-step breakdown\n"
      printf "=================================\n"
      printf "Captured logical optimizer steps: %d\n", logical_steps
      printf "Observed outer rank-step instances: %d\n", optimizer_instances
      printf "Expected rank-step instances: %d\n", expected_rank_steps
      printf "Inferred ranks: %d\n", inferred_ranks
      if (optimizer_instances != expected_rank_steps) {
        printf "WARNING: %d outer OPTIMIZER_STEP range(s) were missed at the capture boundary.\n", \
          expected_rank_steps - optimizer_instances
      }
      printf "Average OPTIMIZER_STEP wall time per rank: %.3f ms\n", \
        optimizer_average / 1000000
      printf "Average enclosing TRAIN_SYNC time per rank-step: %.3f ms\n\n", \
        train_sync_total / expected_rank_steps / 1000000
      printf "OPTIMIZER_STEP includes all gradient-accumulation micro-batches;\n"
      printf "ADAMW_STEP is only the actual optimizer.step() call.\n"
      printf "Values below are CPU-side range wall times averaged per rank-step.\n\n"
      printf "%-24s %14s %12s %11s\n", \
        "Phase", "ms/rank-step", "calls/step", "% step wall"
      printf "%-24s %14s %12s %11s\n", \
        "------------------------", "--------------", "------------", "-----------"

      leaf_total = 0
      for (i = 1; i <= phase_count; i++) {
        key = phase[i]
        value_ms = phase_total[key] / expected_rank_steps / 1000000
        calls_per_step = phase_instances[key] / expected_rank_steps
        percentage = 100 * phase_total[key] / normalized_optimizer_total
        leaf_total += phase_total[key]
        printf "%-24s %14.3f %12.2f %10.2f%%\n", \
          key, value_ms, calls_per_step, percentage
      }

      unattributed = normalized_optimizer_total - leaf_total
      printf "%-24s %14.3f %12s %10.2f%%\n", \
        "UNATTRIBUTED/GAPS", unattributed / expected_rank_steps / 1000000, \
        "-", 100 * unattributed / normalized_optimizer_total
      printf "\nNested parent ranges are intentionally excluded from the phase sum.\n"
    }
  ' "${source}" > "${destination}"
}

run_stat_csv() {
  local sqlite="$1"
  local stat_name="$2"
  local stat_spec="$3"
  local destination="$4"
  local temporary="${destination}.tmp"

  if "${NSYS_BIN}" stats \
      --quiet \
      --report "${stat_spec}" \
      --format=csv \
      --timeunit=nsec \
      "${sqlite}" > "${temporary}"; then
    mv -f "${temporary}" "${destination}"
  else
    rm -f "${temporary}"
    warn "Failed to generate ${stat_name} for ${sqlite}"
    return 1
  fi
}

append_csv_preview() {
  local source="$1"
  local title="$2"
  local overview="$3"
  local preview_lines=$((TOP_N + 1))

  {
    printf '\n================================================================================\n'
    printf '%s\n' "${title}"
    printf '================================================================================\n'
  } >> "${overview}"

  if [[ -s "${source}" ]]; then
    sed -n "1,${preview_lines}p" "${source}" >> "${overview}"
  else
    printf 'No rows generated.\n' >> "${overview}"
  fi
}

append_rule_analysis() {
  local sqlite="$1"
  local rule_name="$2"
  local destination="$3"
  local overview="$4"
  local temporary="${destination}.tmp"

  if [[ -e "${destination}" && "${FORCE_STATS}" != "1" \
      && "${FORCE_EXPORT}" != "1" ]]; then
    log "Reusing rule analysis: ${destination}"
  elif "${NSYS_BIN}" analyze \
        --quiet \
        --rule "${rule_name}" \
        --format=table \
        --timeunit=msec \
        "${sqlite}" > "${temporary}"; then
    mv -f "${temporary}" "${destination}"
  else
    rm -f "${temporary}"
    warn "nsys analyze rule failed: ${rule_name}"
    return 1
  fi

  if [[ -s "${destination}" ]]; then
    {
      printf '\n================================================================================\n'
      printf 'NSYS ANALYZE RULE: %s\n' "${rule_name}"
      printf '================================================================================\n'
      cat "${destination}"
    } >> "${overview}"
  fi
}

analyze_report() {
  local report="$1"
  local report_base="${report%.nsys-rep}"
  local sqlite="${report_base}.sqlite"
  local analysis_dir="${report_base}_analysis"
  local overview="${analysis_dir}/overview.txt"
  local manifest="${analysis_dir}/manifest.txt"
  local stat_name
  local stat_spec
  local stat_csv
  local rule_name
  local index

  validate_analysis_settings
  mkdir -p "${analysis_dir}"
  find "${analysis_dir}" -maxdepth 1 -type f -name '*.tmp' -delete
  log "Analyzing ${report}"

  if [[ ! -s "${sqlite}" || "${FORCE_EXPORT}" == "1" ]]; then
    log "Exporting SQLite: ${sqlite}"
    "${NSYS_BIN}" export \
      --quiet=true \
      --type=sqlite \
      --force-overwrite=true \
      --output="${sqlite}" \
      "${report}"
  else
    log "Reusing SQLite: ${sqlite}"
  fi

  {
    printf 'report=%s\n' "${report}"
    printf 'sqlite=%s\n' "${sqlite}"
    printf 'analysis_dir=%s\n' "${analysis_dir}"
    printf 'generated_at=%s\n' "$(date --iso-8601=seconds)"
    printf 'nsys_version=%s\n' "$("${NSYS_BIN}" --version | head -n 1)"
  } > "${manifest}"

  for index in "${!STAT_NAMES[@]}"; do
    stat_name="${STAT_NAMES[${index}]}"
    stat_spec="${STAT_SPECS[${index}]}"
    stat_csv="${analysis_dir}/${stat_name}.csv"
    if [[ -e "${stat_csv}" && "${FORCE_STATS}" != "1" \
        && "${FORCE_EXPORT}" != "1" ]]; then
      log "Reusing stats CSV: ${stat_csv}"
    else
      run_stat_csv \
        "${sqlite}" "${stat_name}" "${stat_spec}" "${stat_csv}" || true
    fi
  done

  write_filtered_csv \
    "${analysis_dir}/nvtx_pushpop_sum.csv" \
    "${analysis_dir}/nvtx_training_phases_cpu.csv" \
    'OPTIMIZER_STEP_|MICRO_[0-9]+/|DATA_WAIT|TRAIN_SYNC|LR_PREPARE|H2D_AND_PREPARE|FORWARD|BACKWARD|GRAD_CLIP|ADAMW_STEP|LR_SCHEDULER|ZERO_GRAD|CUSTOM_TRAINING_STEP'
  write_filtered_csv \
    "${analysis_dir}/nvtx_gpu_proj_sum.csv" \
    "${analysis_dir}/nvtx_training_phases_gpu.csv" \
    'OPTIMIZER_STEP_|MICRO_[0-9]+/|DATA_WAIT|TRAIN_SYNC|LR_PREPARE|H2D_AND_PREPARE|FORWARD|BACKWARD|GRAD_CLIP|ADAMW_STEP|LR_SCHEDULER|ZERO_GRAD|CUSTOM_TRAINING_STEP'
  write_filtered_csv \
    "${analysis_dir}/cuda_gpu_kern_sum.csv" \
    "${analysis_dir}/nccl_kernels.csv" \
    'nccl'
  write_filtered_csv \
    "${analysis_dir}/osrt_sum.csv" \
    "${analysis_dir}/osrt_file_io.csv" \
    ',(read|pread|pread64|open|open64|openat|fopen|fread|mmap|mmap64|munmap|stat|fstat|lseek|close|write|pwrite|fsync|fdatasync|io_uring[^,]*)$'
  write_filtered_csv \
    "${analysis_dir}/osrt_sum.csv" \
    "${analysis_dir}/osrt_blocking.csv" \
    ',(poll|select|epoll_wait|futex|pthread_cond_wait|pthread_cond_timedwait|sem_wait|sem_timedwait|nanosleep)$'
  generate_optimizer_step_breakdown \
    "${analysis_dir}/nvtx_pushpop_sum.csv" \
    "${analysis_dir}/optimizer_step_breakdown.txt"

  {
    printf 'VLA-JEPA TRAINING NSIGHT SYSTEMS ANALYSIS\n'
    printf 'Report: %s\n' "${report}"
    printf 'SQLite: %s\n' "${sqlite}"
    printf 'Generated: %s\n' "$(date --iso-8601=seconds)"
    printf '\nInterpretation guide:\n'
    printf '  * nvtx_pushpop_sum: CPU-side range duration; nested ranges overlap.\n'
    printf '  * nvtx_gpu_proj_sum: GPU work launched inside each NVTX range.\n'
    printf '  * Autograd/FSDP worker-thread launches may not inherit Python-thread NVTX.\n'
    printf '    A small BACKWARD GPU projection does not prove backward GPU work is small.\n'
    printf '  * DATA_WAIT + GPU gaps: input/data pipeline bottleneck signal.\n'
    printf '  * osrt_file_io.csv: aggregate file API time; deep mode also records paths.\n'
    printf '  * osrt_blocking.csv: blocking/wait APIs aggregated across ranks/threads.\n'
    printf '  * BACKWARD: includes checkpoint recomputation and FSDP/NCCL work.\n'
    printf '  * Queue time is not automatically bad; it can indicate a busy GPU.\n'
    printf '  * NCCL kernel time may overlap compute and is not pure overhead.\n'
  } > "${overview}"

  {
    printf '\n================================================================================\n'
    printf 'OPTIMIZER STEP CPU-SIDE BREAKDOWN\n'
    printf '================================================================================\n'
    cat "${analysis_dir}/optimizer_step_breakdown.txt"
  } >> "${overview}"

  append_csv_preview \
    "${analysis_dir}/nvtx_pushpop_sum.csv" \
    'NVTX CPU TRAINING RANGES (nanoseconds)' \
    "${overview}"
  append_csv_preview \
    "${analysis_dir}/nvtx_gpu_proj_sum.csv" \
    'NVTX GPU PROJECTION (nanoseconds)' \
    "${overview}"
  append_csv_preview \
    "${analysis_dir}/cuda_gpu_kern_sum.csv" \
    'TOP CUDA KERNELS (nanoseconds)' \
    "${overview}"
  append_csv_preview \
    "${analysis_dir}/cuda_kern_exec_sum.csv" \
    'CUDA KERNEL LAUNCH AND QUEUE TIME (nanoseconds)' \
    "${overview}"
  append_csv_preview \
    "${analysis_dir}/cuda_gpu_kern_by_nvtx.csv" \
    'CUDA KERNELS GROUPED BY INNERMOST NVTX RANGE (nanoseconds)' \
    "${overview}"

  if [[ "${RUN_ANALYZE_RULES}" == "1" ]]; then
    for rule_name in "${ANALYZE_RULES[@]}"; do
      append_rule_analysis \
        "${sqlite}" \
        "${rule_name}" \
        "${analysis_dir}/rule_${rule_name}.txt" \
        "${overview}" || true
    done
  fi

  if ! grep -Eq 'OPTIMIZER_STEP_|FORWARD|BACKWARD' \
      "${analysis_dir}/nvtx_pushpop_sum.csv" 2>/dev/null; then
    warn "No FluxVLA training NVTX ranges found in ${report}."
    {
      printf '\nWARNING: No OPTIMIZER_STEP/FORWARD/BACKWARD ranges were found.\n'
      printf 'Confirm FLUXVLA_NSYS_START_STEP/NUM_STEPS reached the runner.\n'
    } >> "${overview}"
  fi

  {
    printf '\nartifacts:\n'
    find "${analysis_dir}" -maxdepth 1 -type f -printf '  %f\n' | sort
  } >> "${manifest}"

  log "Analysis overview: ${overview}"
  log "CSV directory: ${analysis_dir}"
}

run_profile() {
  local -a cfg_options
  local -a env_args
  local -a profile_command
  local config_path
  local marker
  local command_log
  local report
  local profile_prefix

  validate_profile_settings
  mkdir -p "${OUTPUT_DIR}" "${WORK_DIR}"
  config_path="$(resolve_repo_path "${CONFIG}")"
  profile_prefix="${OUTPUT_DIR}/${RUN_TAG}_%h_%p"
  marker="${OUTPUT_DIR}/.${RUN_TAG}.capture-start"
  command_log="${OUTPUT_DIR}/${RUN_TAG}.command.txt"

  cfg_options=(
    "train_dataloader.per_device_batch_size=${PER_DEVICE_BATCH_SIZE}"
    "runner.grad_accumulation_steps=${GRAD_ACCUMULATION_STEPS}"
    "runner.max_steps=${MAX_STEPS}"
    "runner.save_iter_interval=${SAVE_ITER_INTERVAL}"
  )
  if (( $# > 0 )); then
    cfg_options+=("$@")
  fi

  env_args=(
    "PYTHONPATH=${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
    "HF_ENDPOINT=${HF_ENDPOINT}"
    "WANDB_MODE=${WANDB_MODE}"
    "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    "OMP_NUM_THREADS=${OMP_NUM_THREADS}"
    "PYTHONUNBUFFERED=1"
    "TOKENIZERS_PARALLELISM=false"
    "FLUXVLA_NSYS_START_STEP=${START_STEP}"
    "FLUXVLA_NSYS_NUM_STEPS=${NUM_STEPS}"
    "FLUXVLA_NSYS_ALL_RANKS=${ALL_RANKS}"
  )

  profile_command=(
    "${NSYS_BIN}" profile
    --trace=cuda,nvtx,nccl,osrt
    --capture-range=cudaProfilerApi
    --capture-range-end=stop
    --cuda-trace-scope=process-tree
    "--sample=${NSYS_SAMPLE}"
    "--cpuctxsw=${NSYS_CPUCTXSW}"
    "--python-sampling=${NSYS_PYTHON_SAMPLING}"
    "--python-sampling-frequency=${NSYS_PYTHON_SAMPLING_FREQUENCY}"
    "--python-backtrace=${NSYS_PYTHON_BACKTRACE}"
    "--osrt-file-access=${NSYS_OSRT_FILE_ACCESS}"
    "--osrt-threshold=${NSYS_OSRT_THRESHOLD_NS}"
    "--syscall=${NSYS_SYSCALL}"
    "--pytorch=${NSYS_PYTORCH_TRACE}"
    --cuda-memory-usage=false
    --force-overwrite=true
    "--stats=${NSYS_INLINE_STATS}"
    --export=sqlite
    "--output=${profile_prefix}"
    /usr/bin/env
    "${env_args[@]}"
    "${PYTHON}"
    -m torch.distributed.run
    --standalone
    --nnodes=1
    "--nproc-per-node=${NPROC_PER_NODE}"
    "${REPO_ROOT}/scripts/train.py"
    --config "${config_path}"
    --work-dir "${WORK_DIR}"
    --cfg-options
    "${cfg_options[@]}"
  )

  {
    printf 'mode=%s\n' "${MODE}"
    printf 'start_step=%s\n' "${START_STEP}"
    printf 'num_steps=%s\n' "${NUM_STEPS}"
    printf 'max_steps=%s\n' "${MAX_STEPS}"
    printf 'all_ranks=%s\n' "${ALL_RANKS}"
    printf 'cpu_io_trace=%s\n' "${CPU_IO_TRACE}"
    printf 'nsys_sample=%s\n' "${NSYS_SAMPLE}"
    printf 'nsys_cpuctxsw=%s\n' "${NSYS_CPUCTXSW}"
    printf 'nsys_python_sampling=%s\n' "${NSYS_PYTHON_SAMPLING}"
    printf 'nsys_python_backtrace=%s\n' "${NSYS_PYTHON_BACKTRACE}"
    printf 'nsys_osrt_file_access=%s\n' "${NSYS_OSRT_FILE_ACCESS}"
    printf 'nsys_syscall=%s\n' "${NSYS_SYSCALL}"
    printf 'command=' 
    printf '%q ' "${profile_command[@]}"
    printf '\n'
  } > "${command_log}"

  log "Capture optimizer steps ${START_STEP}..$((START_STEP + NUM_STEPS - 1))"
  log "Output prefix: ${profile_prefix}"
  log 'Command:'
  print_command "${profile_command[@]}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    log "DRY_RUN=1; command not executed. Command log: ${command_log}"
    return 0
  fi

  log "Imported FluxVLA: $(
    /usr/bin/env "PYTHONPATH=${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
      "${PYTHON}" -c 'import fluxvla; print(fluxvla.__file__)')"

  touch "${marker}"
  (
    cd "${REPO_ROOT}"
    "${profile_command[@]}"
  ) 2>&1 | tee "${OUTPUT_DIR}/${RUN_TAG}.profile.log"

  GENERATED_REPORTS=()
  while IFS= read -r -d '' report; do
    GENERATED_REPORTS+=("${report}")
  done < <(find "${OUTPUT_DIR}" -maxdepth 1 -type f \
    -name "${RUN_TAG}_*.nsys-rep" -newer "${marker}" -print0 | sort -z)
  rm -f "${marker}"

  (( ${#GENERATED_REPORTS[@]} > 0 )) || die \
    "No .nsys-rep generated. Inspect ${OUTPUT_DIR}/${RUN_TAG}.profile.log"
  for report in "${GENERATED_REPORTS[@]}"; do
    log "Generated report: ${report}"
  done
}

if [[ "${MODE}" == "help" || "${MODE}" == "-h" || "${MODE}" == "--help" ]]; then
  usage
  exit 0
fi

cd "${REPO_ROOT}"
REPORTS=()
GENERATED_REPORTS=()

case "${MODE}" in
  profile)
    run_profile "$@"
    ;;
  all)
    run_profile "$@"
    if [[ "${DRY_RUN}" != "1" ]]; then
      for report in "${GENERATED_REPORTS[@]}"; do
        analyze_report "${report}"
      done
    fi
    ;;
  analyze)
    require_command "${NSYS_BIN}"
    if (( $# == 0 )); then
      collect_reports_from_target "${OUTPUT_DIR}"
    else
      for target in "$@"; do
        collect_reports_from_target "${target}"
      done
    fi
    (( ${#REPORTS[@]} > 0 )) || die 'No .nsys-rep files found to analyze.'
    for report in "${REPORTS[@]}"; do
      analyze_report "${report}"
    done
    ;;
esac
