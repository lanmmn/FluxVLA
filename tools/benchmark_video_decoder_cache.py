#!/usr/bin/env python3
"""Benchmark torchcodec decoder cache effectiveness.

Examples:
  python tools/benchmark_video_decoder_cache.py \
    --glob 'datasets/SARM_manual_test_10Episodes_lerobotv3.0/videos/*/chunk-000/*.mp4' \
    --max-videos 80 --rounds 4 --cache-size 32 --include-pyav
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path
from typing import List
import os

from fluxvla.datasets.utils.video_decode import (
    clear_torchcodec_decoder_cache,
    decode_video_frames,
    get_torchcodec_decoder_cache_stats,
    reset_torchcodec_decoder_cache_stats,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--glob",
        default="datasets/SARM_manual_test_10Episodes_lerobotv3.0/videos/*/chunk-000/*.mp4",
        help="Glob pattern for video files.",
    )
    parser.add_argument("--max-videos", type=int, default=40)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmup-videos", type=int, default=3)
    parser.add_argument(
        "--timestamps",
        default="0.0,0.0333,0.0667,0.1,0.1333,0.1667,0.2,0.2333,0.2667,0.3",
        help="Comma-separated timestamps in seconds.",
    )
    parser.add_argument(
        "--cache-size",
        type=int,
        default=32,
        help="Cache size for torchcodec ON mode.",
    )
    parser.add_argument(
        "--include-pyav",
        action="store_true",
        help="Also run pyav backend as baseline.",
    )
    return parser.parse_args()


def parse_timestamps(raw: str) -> List[float]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("--timestamps is empty")
    return [float(v) for v in values]


def bench_case(
    *,
    case_name: str,
    backend: str,
    videos: List[Path],
    timestamps: List[float],
    rounds: int,
    warmup_videos: int,
    torchcodec_cache_size: int | None,
) -> dict:
    if backend == "torchcodec":
        if torchcodec_cache_size is None:
            raise ValueError("torchcodec_cache_size must be set for torchcodec")
        os.environ["TORCHCODEC_DECODER_CACHE_SIZE"] = str(torchcodec_cache_size)
        clear_torchcodec_decoder_cache()
        reset_torchcodec_decoder_cache_stats()

    # Warmup to reduce one-time overhead noise.
    for path in videos[: min(warmup_videos, len(videos))]:
        _ = decode_video_frames(path, timestamps, backend=backend)

    if backend == "torchcodec":
        reset_torchcodec_decoder_cache_stats()

    durations_s = []
    for _ in range(rounds):
        start = time.perf_counter()
        for path in videos:
            out = decode_video_frames(path, timestamps, backend=backend)
            _ = out.shape
        durations_s.append(time.perf_counter() - start)

    mean_s = statistics.mean(durations_s)
    num_videos = len(videos)
    num_frames = len(videos) * len(timestamps)
    cache_stats = (get_torchcodec_decoder_cache_stats()
                   if backend == "torchcodec" else None)
    hit_rate = None
    if cache_stats is not None:
        cached_requests = cache_stats["hits"] + cache_stats["misses"]
        hit_rate = (cache_stats["hits"] / cached_requests
                    if cached_requests else 0.0)

    return {
        "case": case_name,
        "backend": backend,
        "cache_size": torchcodec_cache_size,
        "rounds": rounds,
        "videos": num_videos,
        "timestamps_per_video": len(timestamps),
        "times_s": [round(x, 4) for x in durations_s],
        "mean_s": round(mean_s, 4),
        "videos_per_s": round(num_videos / mean_s, 2),
        "frames_per_s": round(num_frames / mean_s, 2),
        "ms_per_video": round(mean_s / num_videos * 1000, 2),
        "cache_stats": cache_stats,
        "cache_hit_rate": round(hit_rate, 4) if hit_rate is not None else None,
    }


def print_result(result: dict) -> None:
    print(
        f"[{result['case']}] backend={result['backend']} cache_size={result['cache_size']}"
    )
    print(
        f"  rounds={result['rounds']} videos/round={result['videos']} timestamps/video={result['timestamps_per_video']}"
    )
    print(f"  times_s={result['times_s']}")
    print(
        f"  mean_s={result['mean_s']} videos_per_s={result['videos_per_s']} "
        f"frames_per_s={result['frames_per_s']} ms_per_video={result['ms_per_video']}"
    )
    if result["cache_stats"] is not None:
        print(
            f"  cache_hit_rate={result['cache_hit_rate']} stats={result['cache_stats']}"
        )


def main() -> None:
    args = parse_args()
    timestamps = parse_timestamps(args.timestamps)

    videos = sorted(Path().glob(args.glob))
    if not videos:
        raise FileNotFoundError(f"No videos matched glob: {args.glob}")
    videos = videos[: args.max_videos]

    print("Benchmark setup")
    print(f"  glob={args.glob}")
    print(f"  videos={len(videos)}")
    print(f"  first_video={videos[0]}")
    print(f"  rounds={args.rounds}")
    print(f"  timestamps={timestamps}")
    print()

    results = []
    results.append(
        bench_case(
            case_name="torchcodec_cache_off",
            backend="torchcodec",
            videos=videos,
            timestamps=timestamps,
            rounds=args.rounds,
            warmup_videos=args.warmup_videos,
            torchcodec_cache_size=0,
        ))
    results.append(
        bench_case(
            case_name="torchcodec_cache_on",
            backend="torchcodec",
            videos=videos,
            timestamps=timestamps,
            rounds=args.rounds,
            warmup_videos=args.warmup_videos,
            torchcodec_cache_size=args.cache_size,
        ))

    if args.include_pyav:
        results.append(
            bench_case(
                case_name="pyav_baseline",
                backend="pyav",
                videos=videos,
                timestamps=timestamps,
                rounds=args.rounds,
                warmup_videos=args.warmup_videos,
                torchcodec_cache_size=None,
            ))

    for result in results:
        print_result(result)

    off = next(result for result in results
               if result["case"] == "torchcodec_cache_off")
    on = next(result for result in results
              if result["case"] == "torchcodec_cache_on")
    speedup = off["mean_s"] / on["mean_s"]
    delta_pct = (off["mean_s"] - on["mean_s"]) / off["mean_s"] * 100
    print()
    print("Torchcodec cache summary")
    print(f"  speedup_x={speedup:.4f}")
    print(f"  latency_delta_pct={delta_pct:.2f}%")


if __name__ == "__main__":
    main()
