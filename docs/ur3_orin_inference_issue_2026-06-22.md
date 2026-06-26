# UR3 Orin Inference Startup Issue - 2026-06-22

## Context

Target command:

```bash
FLUXVLA_IMAGE=fluxvla:orin-ros-fa docker/run_docker.sh \
  python3 scripts/inference_real_robot.py \
    --config configs/gr00t/gr00t_eagle_3b_ur3_full_finetune.py \
    --ckpt-path /mnt/nvme/gr00t-ur3-checkpoint/fab7d175_gr00t_eagle_3b_ur3_full_finetune_bs64/checkpoints/step-042600-epoch-06-loss=0.0238.pt
```

The goal was to run UR3 real-robot inference inside `fluxvla:orin-ros-fa` until the process reached the interactive task-selection stage.

## Symptoms

The first failure happened during package import:

```text
ModuleNotFoundError: No module named 'robosuite'
```

After fixing that, the startup exposed several Orin/ROS environment issues in sequence:

```text
AssertionError: Transformers==4.53.2 is used but incompatible. Please install transformers>=5.3.0, <5.3.1.
ModuleNotFoundError: No module named 'libero'
ModuleNotFoundError: No module named 'torch._C._distributed_c10d'; 'torch._C' is not a package
rospy.exceptions.ROSException: timeout exceeded while waiting for message on topic /wrist_camera/color/camera_info
```

Starting `roscore` in the current image also failed:

```text
ModuleNotFoundError: No module named 'defusedxml'
RLException: Invalid <param> tag: Cannot load command parameter [rosversion]: no such command [['rosversion', 'roslaunch']].
```

## Root Causes

1. `fluxvla/__init__.py` treated `robosuite` as a mandatory top-level dependency. UR3 inference does not require RoboCasa/robosuite, so this blocked non-RoboCasa deployment paths.

2. The top-level package version check only accepted `transformers==5.3.0`, while the Orin-specific Docker dependency file pins `transformers==4.53.2`:

```text
requirements_orin_notorch.txt: transformers==4.53.2
requirements.txt: transformers==5.3.0
```

3. `fluxvla/engines/utils/eval_utils.py` imported LIBERO at module import time even though UR3 inference only needed general utilities such as `quat2axisangle`.

4. `scripts/inference_real_robot.py` imported `fluxvla.models` wholesale. That forced registration/import of training and FSDP-only modules even though the UR3 GR00T inference config only needs:

- `LlavaVLA`
- `EagleBackbone` / `EagleInferenceBackbone`
- `FlowMatchingHead` / `FlowMatchingInferenceHead`

5. The Orin PyTorch build does not include `torch._C._distributed_c10d`, so package-level imports of FSDP-related training modules can fail even when the real-robot inference path does not use distributed training.

6. `UROperator` required `/wrist_camera/color/camera_info` and `/front_camera/color/camera_info` during initialization. These camera-info topics may be absent or late on the real robot, and the collected `cam_info_dict` is not used elsewhere in the current code path.

7. The current `fluxvla:orin-ros-fa` image can run ROS Python modules, but `roscore` is incomplete in this runtime: `defusedxml` was missing and `rosversion` was not available for `roslaunch`.

## Code Fixes Applied Locally

The following files were modified in the local `FluxVLA` worktree:

- `fluxvla/__init__.py`

  - Made `robosuite` optional at top level.
  - Kept robosuite version checks when it is installed.
  - Accepted both `transformers==4.53.x` and `transformers==5.3.0`.

- `fluxvla/engines/utils/eval_utils.py`

  - Removed top-level LIBERO imports.
  - Kept LIBERO imports inside `get_libero_env()`, where they are actually needed.

- `scripts/inference_real_robot.py`

  - Replaced full `import fluxvla.models` with targeted imports for the UR3 GR00T inference components.

- `fluxvla/engines/runners/__init__.py`

  - Fixed the skipped-import warning message so it prints the real exception instead of the literal `{e}`.

- `fluxvla/models/backbones/__init__.py`

- `fluxvla/models/backbones/vlms/__init__.py`

- `fluxvla/models/heads/__init__.py`

- `fluxvla/models/vlas/__init__.py`

  - Added narrow handling for optional `torch.distributed` / `torch._C._distributed_c10d` import failures.
  - Kept non-distributed import errors visible.

- `fluxvla/engines/operators/ur_operator.py`

  - Made camera-info collection non-fatal.
  - Missing camera-info topics now log warnings and inference initialization continues.

## ROS Master Workaround Used For Validation

Because `roscore` failed in the current image, validation used `rosmaster` directly:

```bash
FLUXVLA_IMAGE=fluxvla:orin-ros-fa docker/run_docker.sh \
  bash -lc 'python3 -m pip install -q defusedxml >/dev/null 2>&1 || true; exec rosmaster --core -p 11311'
```

This opened port `11311` and allowed `rospy` registration to proceed.

Longer-term image fix: make sure the ROS layer includes the runtime dependency for `defusedxml` and a complete ROS command environment including `rosversion`, so plain `roscore` works.

## Validation Result

Syntax checks passed for the changed Python files:

```bash
python -m py_compile \
  fluxvla/__init__.py \
  fluxvla/engines/utils/eval_utils.py \
  fluxvla/models/backbones/__init__.py \
  fluxvla/models/backbones/vlms/__init__.py \
  fluxvla/models/heads/__init__.py \
  fluxvla/models/vlas/__init__.py \
  fluxvla/engines/operators/ur_operator.py \
  scripts/inference_real_robot.py
```

The full inference command progressed through:

```text
[Startup] load_config: 0.0s
[*] [Startup] build_dataset: 0.6s
[*] [Startup] build_vla: ~43s
[*] [Startup] load_checkpoint_to_cpu: ~93s
[*] [Startup] load_state_dict: ~1.4s
topicmanager initialized
```

Camera-info topics were skipped as warnings:

```text
Skip camera info topic /wrist_camera/color/camera_info during initialization: timeout exceeded while waiting for message on topic /wrist_camera/color/camera_info
Skip camera info topic /front_camera/color/camera_info during initialization: timeout exceeded while waiting for message on topic /front_camera/color/camera_info
```

The process then reached the interactive real-robot task prompt:

```text
[Startup] build_runner_from_cfg: 148.8s
[Startup] run_setup_ros: 4.6s
[Startup] total_before_interactive_loop: 153.4s
Enter task ID (or press Enter for default):
```

No task ID was entered automatically because that can trigger real-robot motion.

## Operational Notes

- To run again, start a ROS master first, then launch the inference command.
- If camera drivers are running normally, `/wrist_camera/color/camera_info` and `/front_camera/color/camera_info` should ideally publish before inference starts, but they are no longer hard blockers.
- If running on the real robot, enter task IDs only after checking arm, gripper, camera, and workspace safety.
- Validation containers created during debugging were stopped after the prompt was reached.

## Current Container cv_bridge Fix

The running `fluxvla:orin-ros-fa` container also hit this runtime error when converting ROS images:

```text
ImportError: libboost_python310.so.1.74.0: cannot open shared object file: No such file or directory
```

Temporary fix applied inside the current container:

```bash
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  libboost-python1.74.0 libboost-regex1.74.0
```

Validation result:

```text
libboost_python310.so.1.74.0 => /usr/lib/aarch64-linux-gnu/libboost_python310.so.1.74.0
libboost_regex.so.1.74.0 => /usr/lib/aarch64-linux-gnu/libboost_regex.so.1.74.0
cv_bridge conversion ok (1, 1, 3) [[[1, 2, 3]]]
```

This fix only affects the current container. Rebuild `fluxvla:orin-ros-fa` from the updated Dockerfile to make it persistent.

## Temporary Checkpoint Conversion Helper

The UR3 checkpoint used in this issue is a 29GB PyTorch zip checkpoint. Inference startup spends most of its time in `torch.load(..., map_location='cpu')` before GPU execution starts.

A temporary helper script was added locally:

```bash
scripts/convert_checkpoint_to_safetensors.py
```

Use it after stopping any running inference process, so the system does not load the 29GB checkpoint twice at the same time:

```bash
FLUXVLA_IMAGE=fluxvla:orin-ros-fa docker/run_docker.sh \
  python3 scripts/convert_checkpoint_to_safetensors.py \
    /mnt/nvme/gr00t-ur3-checkpoint/fab7d175_gr00t_eagle_3b_ur3_full_finetune_bs64/checkpoints/step-042600-epoch-06-loss=0.0238.pt
```

Default output:

```text
/mnt/nvme/gr00t-ur3-checkpoint/fab7d175_gr00t_eagle_3b_ur3_full_finetune_bs64/checkpoints/step-042600-epoch-06-loss=0.0238.model.safetensors
```

After conversion, use the `.model.safetensors` path as `--ckpt-path` to avoid repeatedly loading optimizer/scheduler training state.

Actual conversion result on 2026-06-22:

```text
torch.load: 492.0s
tensors: 899
skipped non-tensors: 0
save_file: 9.2s
output size: 11G
```

The generated file was validated with `safe_open` metadata/key listing without loading all tensors into memory.
