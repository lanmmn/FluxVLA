# VLA-JEPA Nsight Systems and NVTX Profiling

This note records how to profile the current VLA-JEPA LIBERO-10 training run
with Nsight Systems. The training runner emits NVTX ranges and can trigger a
short `cudaProfilerApi` capture window after warm-up steps.

For a concise Chinese change summary and quick-start guide, see
[`vla_jepa_nsys_quickstart.md`](vla_jepa_nsys_quickstart.md).

## What The Runner Marks

NVTX ranges are enabled when either of these environment variables is set:

```bash
FLUXVLA_NVTX=1
FLUXVLA_NSYS_NUM_STEPS=3
```

The step-based training loop emits these ranges:

```text
OPTIMIZER_STEP_000010
  MICRO_00/DATA_WAIT
  MICRO_00/TRAIN_SYNC_0
    LR_PREPARE
    H2D_AND_PREPARE
    FORWARD
    BACKWARD
  MICRO_01/DATA_WAIT
  MICRO_01/TRAIN_SYNC_0
    ...
  MICRO_03/DATA_WAIT
  MICRO_03/TRAIN_SYNC_1
    LR_PREPARE
    H2D_AND_PREPARE
    FORWARD
    BACKWARD
    GRAD_CLIP
    ADAMW_STEP
    LR_SCHEDULER
    ZERO_GRAD
```

For FSDP full-shard, NCCL communication is mostly embedded inside `FORWARD`,
`BACKWARD`, and sometimes `GRAD_CLIP`. Do not treat `ADAMW_STEP` alone as the
total communication cost.

## Capture Controls

Use these environment variables:

```bash
FLUXVLA_NSYS_START_STEP=10
FLUXVLA_NSYS_NUM_STEPS=3
FLUXVLA_NSYS_ALL_RANKS=0
```

Meaning:

- finish 10 optimizer steps as warm-up;
- capture the next 3 optimizer steps;
- trigger capture from rank 0 only.

Set `FLUXVLA_NSYS_ALL_RANKS=1` only when comparing rank imbalance. Rank-0
capture is usually enough for first-pass overhead attribution.

## One-Command Profile And Analysis

The preferred entry point is the repository script. It embeds the required
`FLUXVLA_NSYS_*` variables in the launched training process, captures a short
training window, exports SQLite, and generates CSV/text analysis artifacts:

```bash
cd /root/projects/fluxvla
bash scripts/profile_vla_jepa_nsys.sh all
```

The default run uses 8 GPUs with `per_device_batch_size=32` and
`grad_accumulation_steps=1`, warms up through optimizer step 9, captures steps
10-12, and exits after step 13. Common overrides are environment variables:

```bash
PER_DEVICE_BATCH_SIZE=8 \
GRAD_ACCUMULATION_STEPS=4 \
START_STEP=10 \
NUM_STEPS=2 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash scripts/profile_vla_jepa_nsys.sh all \
  train_dataloader.per_device_num_workers=8
```

Supported modes are:

```bash
# Capture only.
bash scripts/profile_vla_jepa_nsys.sh profile

# Analyze one existing report.
bash scripts/profile_vla_jepa_nsys.sh analyze /path/to/run.nsys-rep

# Analyze every top-level report in the default output directory.
bash scripts/profile_vla_jepa_nsys.sh analyze

# Print the exact training command without executing it.
DRY_RUN=1 bash scripts/profile_vla_jepa_nsys.sh profile
```

Each report gets a sibling analysis directory named `<report>_analysis` with:

```text
overview.txt                         human-readable top reports and diagnostics
manifest.txt                         report paths, Nsight version, artifact list
nvtx_training_phases_cpu.csv         filtered CPU-side training ranges
nvtx_training_phases_gpu.csv         filtered GPU-projected training ranges
cuda_gpu_kern_sum.csv                top CUDA/NCCL kernels
cuda_gpu_kern_by_nvtx.csv            kernels grouped by innermost NVTX range
cuda_kern_exec_sum.csv               launch/API/queue/kernel timing
nccl_kernels.csv                     NCCL-only kernel rows
rule_gpu_gaps.txt                    automatic long GPU-idle-gap analysis
rule_gpu_time_util.txt               automatic GPU utilization analysis
rule_cuda_api_sync.txt               blocking CUDA synchronization calls
```

Set `ALL_RANKS=1` only for short rank-imbalance investigations. Use
`FORCE_EXPORT=1` to rebuild a stale SQLite export, `FORCE_STATS=1` to recompute
existing CSV/rule outputs, and `TOP_N=50` to increase the number of rows
included in `overview.txt`. Existing analysis artifacts are reused by default,
so an interrupted large-report analysis can be resumed. Set
`RUN_ANALYZE_RULES=0` when only CSV summaries are needed.

## Recommended Short Profiling Run

The script above is preferred. The expanded command below is retained for
debugging or environments where a standalone shell script cannot be used. Run
it from a terminal, not from VS Code Debug. It profiles a short run using the
same VLA-JEPA config and the current profiling defaults
`per_device_batch_size=32, grad_accumulation_steps=1`.

```bash
cd /root/projects/fluxvla

export PYTHONPATH=/root/projects/fluxvla
export HF_ENDPOINT=https://hf-mirror.com
export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export FLUXVLA_NSYS_START_STEP=10
export FLUXVLA_NSYS_NUM_STEPS=3
export FLUXVLA_NSYS_ALL_RANKS=0

mkdir -p /tmp/fluxvla_nsys_vla_jepa

# This must print /root/projects/fluxvla/fluxvla/__init__.py. If it prints
# /root/projects/fluxvla-qwen/fluxvla/__init__.py, the VLA-JEPA dataset config
# will fail because that older editable install lacks frame_window_size support.
/root/miniconda3/bin/python -c \
  'import fluxvla; print(fluxvla.__file__)'

# The FLUXVLA_NSYS_* variables are repeated in the launched environment so the
# nsys block still works if it is copied without the export lines above.

nsys profile \
  --trace=cuda,nvtx,nccl,osrt \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --cuda-trace-scope=process-tree \
  --sample=none \
  --cpuctxsw=none \
  --cuda-memory-usage=false \
  --force-overwrite=true \
  --stats=true \
  --export=sqlite \
  --output=/tmp/fluxvla_nsys_vla_jepa/vla_jepa_rank0_%h_%p \
  /usr/bin/env \
    PYTHONPATH=/root/projects/fluxvla:${PYTHONPATH:-} \
    WANDB_MODE=disabled \
    FLUXVLA_NSYS_START_STEP=10 \
    FLUXVLA_NSYS_NUM_STEPS=3 \
    FLUXVLA_NSYS_ALL_RANKS=0 \
  /root/miniconda3/bin/python \
    -m torch.distributed.run \
    --standalone \
    --nnodes 1 \
    --nproc-per-node 8 \
    scripts/train.py \
    --config configs/vla_jepa/vla_jepa_qwen3vl_2b_libero_10_finetune.py \
    --work-dir /tmp/vla_jepa_nsys_profile_run \
    --cfg-options \
    train_dataloader.per_device_batch_size=32 \
    runner.grad_accumulation_steps=1 \
    runner.max_steps=14 \
    runner.save_iter_interval=10000
```

With `START_STEP=10`, `NUM_STEPS=3`, and `max_steps=14`, the run warms up for
steps 0-9, captures steps 10-12, then exits cleanly shortly after. Keeping this
as a short run avoids producing huge reports and avoids waiting for a full
30,000-step training job before the report is finalized.

If Nsight Systems prints `No reports were generated`, first check that the
training process actually received `FLUXVLA_NSYS_START_STEP` and
`FLUXVLA_NSYS_NUM_STEPS`. With `--capture-range=cudaProfilerApi`, the runner
must call `torch.cuda.profiler.start()` and `torch.cuda.profiler.stop()`;
without those environment variables, the capture window never opens and no
`.nsys-rep` file is written.

## Post-Processing Reports

Nsight Systems writes `.nsys-rep` and, because `--export=sqlite` is enabled,
also writes a `.sqlite` file. Prefer the automated analyzer:

```bash
bash scripts/profile_vla_jepa_nsys.sh analyze /path/to/run.nsys-rep
```

The analyzer uses `nsys stats --quiet` so status messages do not contaminate
CSV output. To generate the original minimal report set manually, use:

```bash
cd /tmp/fluxvla_nsys_vla_jepa

for report in *.nsys-rep; do
  base="${report%.nsys-rep}"

  nsys stats \
    --report cuda_api_sum,cuda_gpu_kern_sum,cuda_kern_exec_sum,nvtx_pushpop_sum,nvtx_gpu_proj_sum,osrt_sum \
    "${report}" \
    > "${base}.summary.txt" 2>&1

  for report_name in \
    cuda_api_sum \
    cuda_gpu_kern_sum \
    cuda_kern_exec_sum \
    nvtx_pushpop_sum \
    nvtx_gpu_proj_sum \
    osrt_sum
  do
    nsys stats \
      --report "${report_name}" \
      --format=csv \
      "${report}" \
      > "${base}.${report_name}.csv"
  done
done
```

Keep at least these artifacts:

```text
*.nsys-rep
*.sqlite
*.summary.txt
*.cuda_gpu_kern_sum.csv
*.cuda_kern_exec_sum.csv
*.nvtx_pushpop_sum.csv
*.nvtx_gpu_proj_sum.csv
*.osrt_sum.csv
```

## How To Read The Result

Start with these files:

| Question | Report |
| --- | --- |
| Is the main process waiting for data? | `nvtx_pushpop_sum.csv`, `DATA_WAIT` |
| How much GPU work falls under each training phase? | `nvtx_gpu_proj_sum.csv` |
| Which CUDA kernels dominate? | `cuda_gpu_kern_sum.csv` |
| Are CUDA launches delayed on CPU? | `cuda_kern_exec_sum.csv` |
| Is NCCL prominent? | `cuda_gpu_kern_sum.csv`, search `nccl` |
| Are OS/runtime calls expensive? | `osrt_sum.csv` |

Important interpretation rules:

- `nvtx_pushpop_sum` is CPU-side range time.
- `nvtx_gpu_proj_sum` projects GPU kernels into NVTX ranges and is better for
  GPU phase attribution.
- GPU projection only includes operations whose launch calls are enclosed by
  the range. PyTorch autograd/FSDP worker-thread launches may not inherit a
  Python-thread `BACKWARD` range, so a small projected `BACKWARD` value does
  not prove that backward GPU work is small. Cross-check `OPTIMIZER_STEP`,
  `cuda_gpu_kern_by_nvtx.csv`, and the timeline.
- CUDA is asynchronous, so CPU range time and GPU execution time are not the
  same thing.
- NCCL kernel time is not always pure overhead because communication may
  overlap with compute.
- If `DATA_WAIT` is large and GPU streams have idle gaps, focus on data loading
  and video processing first.
- If `BACKWARD` is much larger than `FORWARD`, consider gradient checkpointing
  recomputation and FSDP communication.

For timeline inspection, open the `.nsys-rep` in Nsight Systems GUI or import
the `.sqlite` into custom analysis scripts.

## Profiling All Ranks

If rank 0 does not explain the slowdown, repeat the short run with:

```bash
export FLUXVLA_NSYS_ALL_RANKS=1
```

This makes all ranks call `cudaProfilerStart/Stop`. The single process-tree
report still contains all torchrun children, and the NVTX ranges can be
compared by process/rank in the timeline. Use this only for short runs because
the report size and profiling overhead are higher.