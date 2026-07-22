# VLA-JEPA FSDP Runtime Notes

This note records how VLA-JEPA is distributed and executed during the
LIBERO-10 training recipe.

## Training Setup

The VLA-JEPA LIBERO-10 recipe uses `FSDPTrainRunner` with:

- `sharding_strategy='full-shard'`, mapped to PyTorch
  `ShardingStrategy.FULL_SHARD`.
- BF16 mixed precision training.
- Gradient checkpointing enabled.
- One torchrun worker per GPU.

With eight GPUs, the process layout is effectively:

```text
rank 0 -> GPU 0
rank 1 -> GPU 1
rank 2 -> GPU 2
rank 3 -> GPU 3
rank 4 -> GPU 4
rank 5 -> GPU 5
rank 6 -> GPU 6
rank 7 -> GPU 7
```

For the default recipe, the logged batch shape is:

```text
batch=256 (4x8x8)
```

which means:

```text
per_device_batch_size x world_size x grad_accumulation_steps
```

## What Is Sharded

This is not ordinary data parallel training where every GPU permanently owns a
complete copy of the model.

Under FSDP full-shard, the following states are partitioned across ranks:

- model parameters
- gradients
- optimizer state

Each rank keeps only its local shard of these states while the model is idle
between layer executions. This is why a large graph such as Qwen3-VL plus the
V-JEPA2 world-model path can fit with a much lower per-GPU persistent memory
footprint than a fully replicated setup.

## What Happens During Forward

At rest, each GPU owns only a shard of each FSDP-managed parameter group. When
execution reaches an FSDP-wrapped module, FSDP temporarily reconstructs the
complete parameters needed for that module:

```text
all-gather current module parameters
-> run the module forward computation
-> reshard or release the full parameters
-> move to the next wrapped module
```

So the model is not permanently replicated, but each wrapped module may be
temporarily materialized in full while that module is being computed.

For VLA-JEPA, forward computation includes the Qwen3-VL path, the V-JEPA2 video
encoder/world-model path used during training, the action-conditioned predictor,
and the flow-matching action head. FSDP communication such as parameter
all-gather is therefore interleaved with the forward compute rather than
appearing as a single isolated communication block.

## What Happens During Backward

Backward follows the same on-demand reconstruction pattern:

```text
all-gather parameters needed for the current backward segment
-> run autograd backward computation
-> reduce-scatter gradients
-> keep only the local gradient shard
```

For non-final gradient-accumulation micro-steps, the runner enters FSDP
`no_sync()` when available, so gradient synchronization is suppressed until the
final micro-step in the accumulation window. On the final micro-step, FSDP
performs the required communication and each rank keeps only the shard of the
gradients it owns.

After backward completes for the final micro-step, the optimizer updates only
the parameter shards local to that rank.

## Which VLA-JEPA Components Participate In FSDP

The complete `VLAJEPA` module is wrapped by FSDP in `FSDPTrainRunner.run_setup()`.
The runner uses the model-provided FSDP wrapping policy unless `no-shard` is
selected.

The Qwen3-VL backbone exposes an FSDP wrapping policy based on
`Qwen3VLTextDecoderLayer`, so Qwen3-VL text decoder blocks are wrapped as
transformer units. The remaining VLA-JEPA modules are still under the top-level
FSDP wrapper, including:

- Qwen3-VL vision-language backbone
- frozen V-JEPA2 encoder path used for training loss
- action-conditioned world predictor
- VLA-JEPA flow-matching action head
- tokenizer-extended embedding parameters owned by the VLA model

Gradient checkpointing is applied after FSDP wrapping. This reduces activation
memory but adds recomputation during backward, so a low persistent memory
footprint does not imply that the run is compute-free or communication-free.

## Practical Interpretation

When profiling this run, the main FSDP communication appears inside forward and
backward ranges:

- `all_gather` / `all-gather` around module parameter materialization
- `reduce_scatter` / `reduce-scatter` around gradient sharding
- possible collectives during gradient clipping

The optimizer step itself updates local shards, so optimizer time alone should
not be treated as the total communication overhead. For FSDP full-shard, inspect
the whole forward/backward timeline and NCCL kernels to understand the real
communication cost.