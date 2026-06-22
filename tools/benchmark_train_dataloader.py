#!/usr/bin/env python3
"""Benchmark training DataLoader throughput without model forward/backward."""

from __future__ import annotations

import argparse
import os
import statistics
import time
from typing import List

from mmengine import Config, DictAction
from torch.utils.data import DataLoader

from fluxvla.engines import build_dataset_from_cfg
from fluxvla.engines.utils import build_collator_from_cfg
from fluxvla.engines.utils.torch_utils import worker_init_function


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--cfg-options", nargs="+", action=DictAction)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--cache-size", type=int, default=None)
    return parser.parse_args()


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = min(len(sorted_values) - 1, max(0, round((len(values) - 1) * pct)))
    return sorted_values[index]


def main() -> None:
    args = parse_args()
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    if args.cache_size is not None:
        os.environ["TORCHCODEC_DECODER_CACHE_SIZE"] = str(args.cache_size)

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    batch_size = args.batch_size or cfg.train_dataloader.per_device_batch_size
    num_workers = (args.num_workers if args.num_workers is not None else
                   getattr(cfg.train_dataloader, "per_device_num_workers", 0))
    use_workers = num_workers > 0

    dataset = build_dataset_from_cfg(cfg.train_dataloader.dataset)
    collator = build_collator_from_cfg(cfg.runner.collator)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=num_workers,
        worker_init_fn=worker_init_function,
        pin_memory=True,
        prefetch_factor=2 if use_workers else None,
        persistent_workers=use_workers,
    )

    iterator = iter(dataloader)
    warmup_times = []
    measured_times = []
    total_steps = args.warmup + args.steps

    for step in range(total_steps):
        start = time.perf_counter()
        batch = next(iterator)
        elapsed = time.perf_counter() - start
        _ = batch.keys() if hasattr(batch, "keys") else batch
        if step < args.warmup:
            warmup_times.append(elapsed)
        else:
            measured_times.append(elapsed)

    mean_s = statistics.mean(measured_times)
    median_s = statistics.median(measured_times)
    print("DataLoader benchmark")
    print(f"  config={args.config}")
    print(f"  batch_size={batch_size}")
    print(f"  num_workers={num_workers}")
    print(f"  cache_size={os.environ.get('TORCHCODEC_DECODER_CACHE_SIZE')}")
    print(f"  warmup_steps={args.warmup}")
    print(f"  measured_steps={args.steps}")
    print(f"  warmup_mean_s={statistics.mean(warmup_times):.6f}")
    print(f"  mean_s={mean_s:.6f}")
    print(f"  median_s={median_s:.6f}")
    print(f"  p90_s={percentile(measured_times, 0.90):.6f}")
    print(f"  p99_s={percentile(measured_times, 0.99):.6f}")
    print(f"  batches_per_s={1.0 / mean_s:.3f}")
    print(f"  samples_per_s={batch_size / mean_s:.3f}")
    print(f"  min_s={min(measured_times):.6f}")
    print(f"  max_s={max(measured_times):.6f}")


if __name__ == "__main__":
    main()
