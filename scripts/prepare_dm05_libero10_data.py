#!/usr/bin/env python3
"""Prepare FluxVLA LeRobot LIBERO-10 data for OpenDM/DM05 training.

The FluxVLA LIBERO datasets are stored as LeRobot v2.1 parquet episodes plus
mp4 videos. OpenDM's DM05 trainer expects one JSONL file per episode, where
image fields point either to image files or to a video file and frame index.
This script builds that JSONL/index layer while reusing the existing mp4 files.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from tqdm import tqdm


DEFAULT_SOURCE_ROOT = (
    "/mnt/data/stable/users/sober/fluxvla_source_data/"
    "libero_10_no_noops_lerobotv2.1"
)
DEFAULT_OUTPUT_ROOT = (
    "/mnt/data/stable/users/sober/fluxvla_source_data/"
    "libero_10_no_noops_opendm"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert LeRobot LIBERO-10 metadata to OpenDM JSONL."
    )
    parser.add_argument(
        "--source-root",
        default=DEFAULT_SOURCE_ROOT,
        help="LeRobot v2.1 dataset root containing data/, videos/, and meta/.",
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where OpenDM JSONL files and index_cache.json are written.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rewrite JSONL files even if a complete index_cache.json already exists.",
    )
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_tasks(source_root: Path) -> dict[int, str]:
    tasks_path = source_root / "meta" / "tasks.jsonl"
    tasks: dict[int, str] = {}
    for record in _load_jsonl(tasks_path):
        tasks[int(record["task_index"])] = str(record["task"])
    return tasks


def _episode_prompt(
    source_root: Path,
    parquet_record: dict[str, Any],
    tasks: dict[int, str],
) -> str:
    task_index = parquet_record.get("task_index")
    if task_index is not None and int(task_index) in tasks:
        return tasks[int(task_index)]

    episode_index = int(parquet_record["episode_index"])
    episodes_path = source_root / "meta" / "episodes.jsonl"
    for record in _load_jsonl(episodes_path):
        if int(record["episode_index"]) == episode_index:
            episode_tasks = record.get("tasks") or []
            if episode_tasks:
                return str(episode_tasks[0])
    raise KeyError(f"Could not resolve task prompt for episode {episode_index}")


def _as_list(value: Any, *, field: str, row: int) -> list[float]:
    if value is None:
        raise ValueError(f"{field} is missing at row {row}")
    return [float(x) for x in value]


def _episode_video_url(parquet_path: Path, source_root: Path, video_key: str) -> str:
    rel = parquet_path.relative_to(source_root / "data")
    episode_name = parquet_path.with_suffix(".mp4").name
    video_path = source_root / "videos" / rel.parent / video_key / episode_name
    if not video_path.exists():
        raise FileNotFoundError(f"Missing video for {parquet_path}: {video_path}")
    return os.fspath(video_path.relative_to(source_root / "videos"))


def _convert_episode(
    parquet_path: Path,
    source_root: Path,
    output_root: Path,
    tasks: dict[int, str],
) -> tuple[Path, int]:
    table = pq.read_table(
        parquet_path,
        columns=[
            "observation.state",
            "action",
            "frame_index",
            "episode_index",
            "task_index",
        ],
    )
    records = table.to_pylist()
    if not records:
        raise ValueError(f"Episode parquet is empty: {parquet_path}")

    prompt = _episode_prompt(source_root, records[0], tasks)
    head_video = _episode_video_url(
        parquet_path, source_root, "observation.images.image"
    )
    wrist_video = _episode_video_url(
        parquet_path, source_root, "observation.images.wrist_image"
    )

    episode_index = int(records[0]["episode_index"])
    out_path = output_root / f"episode_{episode_index:06d}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for row_idx, row in enumerate(records):
            frame_idx = int(row.get("frame_index", row_idx))
            payload = {
                "images_1": {
                    "type": "video",
                    "url": f"./{head_video}",
                    "frame_idx": frame_idx,
                },
                "images_2": {
                    "type": "video",
                    "url": f"./{wrist_video}",
                    "frame_idx": frame_idx,
                },
                "state": _as_list(row["observation.state"], field="state", row=row_idx),
                "action": _as_list(row["action"], field="action", row=row_idx),
                "prompt": prompt,
            }
            f.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")

    return out_path, len(records)


def _write_index_cache(output_root: Path, counts: dict[Path, int]) -> None:
    index_payload = {
        "data": {os.fspath(path): count for path, count in sorted(counts.items())}
    }
    with (output_root / "index_cache.json").open("w", encoding="utf-8") as f:
        json.dump(index_payload, f, indent=2)


def _is_existing_conversion_complete(output_root: Path) -> bool:
    index_path = output_root / "index_cache.json"
    if not index_path.exists():
        return False
    try:
        with index_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except json.JSONDecodeError:
        return False
    data = payload.get("data")
    return isinstance(data, dict) and bool(data)


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    if not source_root.exists():
        raise FileNotFoundError(f"Source dataset root does not exist: {source_root}")
    if _is_existing_conversion_complete(output_root) and not args.force:
        print(f"Using existing OpenDM JSONL conversion: {output_root}")
        return

    tasks = _load_tasks(source_root)
    parquet_files = sorted((source_root / "data").glob("chunk-*/episode_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet episodes found under {source_root / 'data'}")

    output_root.mkdir(parents=True, exist_ok=True)
    counts: dict[Path, int] = {}
    for parquet_path in tqdm(parquet_files, desc="Converting LIBERO-10 episodes"):
        out_path, num_rows = _convert_episode(parquet_path, source_root, output_root, tasks)
        counts[out_path] = num_rows

    _write_index_cache(output_root, counts)
    print(
        f"Wrote {len(counts)} OpenDM JSONL episodes "
        f"({sum(counts.values())} frames) to {output_root}"
    )
    print(f"Use image_dir={source_root / 'videos'}")


if __name__ == "__main__":
    main()
