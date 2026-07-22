# VLA-JEPA Nsight Systems and NVTX Profiling

This note records how to profile the current VLA-JEPA LIBERO-10 training run
with Nsight Systems. The training runner emits NVTX ranges and can trigger a
short `cudaProfilerApi` capture window after warm-up steps.

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

## Recommended Short Profiling Run

Run this from a terminal, not from VS Code Debug. The command profiles a short
run using the same VLA-JEPA config and the current larger-batch test setting
`per_device_batch_size=16, grad_accumulation_steps=4`.

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
  /root/miniconda3/envs/fluxvla-test/bin/python \
    -m torch.distributed.run \
    --standalone \
    --nnodes 1 \
    --nproc-per-node 8 \
    scripts/train.py \
    --config configs/vla_jepa/vla_jepa_qwen3vl_2b_libero_10_finetune.py \
    --work-dir /tmp/vla_jepa_nsys_profile_run \
    --cfg-options \
    train_dataloader.per_device_batch_size=16 \
    runner.grad_accumulation_steps=4 \
    runner.max_steps=14 \
    runner.save_iter_interval=10000
```

With `START_STEP=10`, `NUM_STEPS=3`, and `max_steps=14`, the run warms up for
steps 0-9, captures steps 10-12, then exits cleanly shortly after. Keeping this
as a short run avoids producing huge reports and avoids waiting for a full
30,000-step training job before the report is finalized.

## Post-Processing Reports

Nsight Systems writes `.nsys-rep` and, because `--export=sqlite` is enabled,
also writes a `.sqlite` file. Generate text and CSV summaries with:

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