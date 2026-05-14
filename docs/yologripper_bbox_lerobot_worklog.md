# YOLOGripper BBox LeRobot Conversion Worklog

## Goal

Generate a UR3 LeRobot dataset that keeps only YOLOGripper bbox annotations, filters frames by existing bbox frames, writes 7-D `action` from shifted `observation.state`, and verifies the output with FluxVLA dataset loading.

## Environment

Server:

```text
LimVLA-3
root@14.103.233.39:57705
```

Conda environment:

```bash
conda activate base
```

## Code Files

Conversion directory:

```text
/root/code/DataLoop/convert/parquet2lerobot/ur3
```

Files used:

```text
convert_npy_to_lerobot_v2_with_bbox.py
run_yologripper_bbox_debug.py
converted_20250904_yologripper_debug.json
```

FluxVLA smoke-load script:

```text
/root/projects/fluxvla/scripts/smoke_load_yologripper_bbox_dataset.py
```

## Output Dataset

The debug runner writes to:

```text
/limx_embop/tos/limx_mani_data/raw_data/RealRobot_UR3_lerobot_v2_bbox_debug/20250904_yologripper_debug/20250904_yologripper_debug
```

This path comes from:

```python
HF_LEROBOT_HOME = "/limx_embop/tos/limx_mani_data/raw_data/RealRobot_UR3_lerobot_v2_bbox_debug/20250904_yologripper_debug"
repo_id = "20250904_yologripper_debug"
```

## Conversion Behavior

BBox source:

```text
bbox2d_rgb_yologripper_threshold0_2.json
```

BBox filtering:

```text
bbox_file_keyword = "yologripper"
bbox_class_keywords = ["gripper"]
```

The resulting bbox class map contains only:

```json
{
  "left gripper": 0
}
```

Frame filtering rule:

```text
keep frame if task != "empty" and frame has YOLOGripper gripper bbox
```

Action rule:

```text
action[t] = next kept frame's observation.state
action[-1] = last kept frame's observation.state
```

So the written parquet satisfies:

```text
action[:-1] == observation.state[1:]
action[-1] == observation.state[-1]
```

## Commands Run

Generate debug dataset:

```bash
cd /root/code/DataLoop/convert/parquet2lerobot/ur3
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
python run_yologripper_bbox_debug.py
```

FluxVLA load test:

```bash
cd /root/projects/fluxvla
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
python scripts/smoke_load_yologripper_bbox_dataset.py \
  --data-root /limx_embop/tos/limx_mani_data/raw_data/RealRobot_UR3_lerobot_v2_bbox_debug/20250904_yologripper_debug/20250904_yologripper_debug \
  --index 0
```

## Generated Dataset Summary

The generated debug dataset contains:

```text
episodes: 2
rows: 230
episode_lengths: [123, 107]
tasks: 3
bbox_class_map: {"left gripper": 0}
```

## FluxVLA Load Test Result

The smoke-load script performs two checks:

1. Raw LeRobot parquet check.
2. FluxVLA `ParquetDataset` load check.

Observed successful output:

```text
RAW_OK
episodes=2 rows=230 tasks=3
episode_lengths=[123, 107]
bbox_class_map={'id_to_class': {'0': 'left gripper'}, 'class_to_id': {'left gripper': 0}}

FLUXVLA_OK
len=230 index=0
states: shape=(7,)
proprio: shape=(7,)
actions_from_parquet: shape=(7,)
observation.eepose: shape=(7,)
observation.bbox2d.boxes: shape=(4, 4)
observation.bbox2d.mask: shape=(4,)
observation.bbox2d.class_id: shape=(4,)
observation.bbox2d.count: 1
images: list_len=2 first_shape=(3, 480, 640)
img_masks: shape=(2,)
task_description: pick up the gray bowl
```

TensorFlow, robosuite, and Gym warnings were printed during import, but they did not block loading.

## Notes

FluxVLA has no existing dedicated script for this exact bbox dataset load check, so `scripts/smoke_load_yologripper_bbox_dataset.py` was added.

The smoke script intentionally does not modify FluxVLA training code. It only imports `fluxvla.datasets.parquet_dataset.ParquetDataset`, builds the dataset, and verifies sample fields and shapes.
