#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from fluxvla.datasets.parquet_dataset import ParquetDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        default="/limx_embop/tos/limx_mani_data/raw_data/RealRobot_UR3_lerobot_v2_bbox_debug/20250904_yologripper_debug/20250904_yologripper_debug",
    )
    parser.add_argument("--index", type=int, default=0)
    return parser.parse_args()


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def check_raw_dataset(root):
    info = json.load(open(root / "meta" / "info.json", encoding="utf-8"))
    tasks = load_jsonl(root / "meta" / "tasks.jsonl")
    episodes = load_jsonl(root / "meta" / "episodes.jsonl")
    class_map = json.load(open(root / "meta" / "bbox2d_class_map.json", encoding="utf-8"))
    parquet_files = sorted((root / "data").glob("*/*.parquet"))
    assert parquet_files, f"No parquet files found under {root / 'data'}"

    total_rows = 0
    for parquet_file in parquet_files:
        table = pq.read_table(parquet_file)
        total_rows += table.num_rows
        state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        bbox_count = np.asarray(table["observation.bbox2d.count"].to_pylist()).reshape(-1)
        class_id = np.asarray(table["observation.bbox2d.class_id"].to_pylist())

        assert state.shape[-1] == 7, state.shape
        assert action.shape[-1] == 7, action.shape
        if len(state) > 1:
            assert np.allclose(action[:-1], state[1:]), parquet_file
        assert np.allclose(action[-1], state[-1]), parquet_file
        assert np.all(bbox_count > 0), parquet_file
        assert set(np.unique(class_id)).issubset({-1, 0}), parquet_file

    print("RAW_OK")
    print(f"root={root}")
    print(f"episodes={info['total_episodes']} rows={total_rows} tasks={len(tasks)}")
    print(f"episode_lengths={[ep['length'] for ep in episodes]}")
    print(f"bbox_class_map={class_map}")


def check_fluxvla_dataset(root, index):
    dataset = ParquetDataset(
        data_root_path=str(root),
        transforms=[
            dict(
                type="ProcessParquetInputs",
                parquet_keys=[
                    "observation.state",
                    "observation.eepose",
                    "timestamp",
                    "action",
                    "observation.bbox2d.boxes",
                    "observation.bbox2d.mask",
                    "observation.bbox2d.class_id",
                    "observation.bbox2d.count",
                    "info",
                    "stats",
                    "action_masks",
                ],
                video_keys=[
                    "observation.images.cam_high",
                    "observation.images.cam_wrist",
                ],
                name_mappings={
                    "observation.state": ["states", "proprio"],
                    "action": ["actions_from_parquet"],
                },
            ),
        ],
        action_window_size=4,
        action_key="action",
        window_start_idx=0,
    )

    stats = {
        "private": {
            "observation.state": dataset.stats[0]["stats"]["observation.state"],
            "action": dataset.stats[0]["stats"]["action"],
            "observation.eepose": dataset.stats[0]["stats"]["observation.eepose"],
        }
    }
    sample = dataset.__getitem__(index, stats)

    print("FLUXVLA_OK")
    print(f"len={len(dataset)} index={index}")
    for key in [
        "states",
        "proprio",
        "actions_from_parquet",
        "observation.eepose",
        "observation.bbox2d.boxes",
        "observation.bbox2d.mask",
        "observation.bbox2d.class_id",
        "observation.bbox2d.count",
        "images",
        "img_masks",
        "task_description",
    ]:
        value = sample[key]
        if hasattr(value, "shape"):
            print(f"{key}: shape={value.shape} dtype={getattr(value, 'dtype', type(value))}")
        elif isinstance(value, list):
            first_shape = getattr(value[0], "shape", None) if value else None
            print(f"{key}: list_len={len(value)} first_shape={first_shape}")
        else:
            print(f"{key}: {value}")

    assert sample["states"].shape == (7,)
    assert sample["actions_from_parquet"].shape == (7,)
    assert sample["observation.bbox2d.boxes"].shape == (4, 4)
    assert np.asarray(sample["observation.bbox2d.count"]).reshape(-1)[0] > 0


def main():
    args = parse_args()
    root = Path(args.data_root)
    check_raw_dataset(root)
    check_fluxvla_dataset(root, args.index)


if __name__ == "__main__":
    main()
