# Jetson Orin Docker and Runtime Test Guide

> Date: 2026-06-22
> Scope: Jetson AGX Orin / JetPack 6.2 / L4T R36.4.x / FluxVLA Docker runtime environment
> Prerequisites: Orin has completed JetPack initialization according to the flashing guide, and CUDA, Docker, and the NVIDIA runtime have been verified.

This document covers FluxVLA Docker build steps after flashing, container runtime, basic tests, ROS/UR3 real-robot tests, and Q&A from recent debugging.

## 1. Recommended Path

Use the Orin host as the stable runtime base. The host should own JetPack, Docker, NVMe storage, source code, and model data. Put the FluxVLA Python, PyTorch, Triton, FlashAttention, and ROS Noetic runtime environment inside Docker images whenever possible.

Recommended source path:

```bash
/home/<user>/FluxVLA
```

Recommended data and model path:

```bash
/mnt/nvme
```

Docker builds are managed by one entry point:

```bash
docker/build_docker.sh
```

The default build creates the recommended layered images:

- `fluxvla:orin-base` without flash attention and ROS 1
- `fluxvla:orin-fa` with flash attention
- `fluxvla:orin-ros` with ROS 1 Noetic
- `fluxvla:orin-ros-fa`

This avoids recompiling FlashAttention and ROS together every time.

For real-robot inference, prefer:

```text
fluxvla:orin-ros-fa
```

For non-ROS inference or benchmark tests, use:

```text
fluxvla:orin-fa
```

## 2. Host Prechecks

After logging into the Orin, first check the system, NVMe, and Docker state:

```bash
cat /etc/nv_tegra_release
nvcc --version
df -h /mnt/nvme
docker --version
docker info | grep -i runtime || true
docker ps
```

If `docker ps` requires `sudo`, fix Docker user permissions first:

```bash
sudo usermod -aG docker "$USER"
newgrp docker
docker ps
```

Put the Docker image store on NVMe to avoid filling the system disk with image layers:

```bash
docker info | grep "Docker Root Dir"
```

If it is still `/var/lib/docker`, migrate it before the first large build:

```bash
sudo apt update
sudo apt install -y rsync
sudo systemctl stop docker
sudo mkdir -p /mnt/nvme/docker
# sudo apt install rsync
sudo rsync -aHAX /var/lib/docker/ /mnt/nvme/docker/

sudo mkdir -p /etc/docker
sudo tee /etc/docker/daemon.json <<'EOF'
{
  "data-root": "/mnt/nvme/docker"
}
EOF

sudo systemctl start docker
docker info | grep "Docker Root Dir"
```

Expected output:

```text
Docker Root Dir: /mnt/nvme/docker
```

## 3. Build Images

Run all commands from the repository root by default:

```bash
cd /home/<user>/FluxVLA
```

Use the unified entry point:

```bash
docker/build_docker.sh
```

The default target is `all`, which builds layers in this order: `base -> flash-attn wheel -> fa -> ros -> ros-fa`. The ROS layer uses China mirrors by default. To disable China mirrors, set `FLUXVLA_USE_CN_MIRRORS=0` explicitly.

Build outputs:

```text
fluxvla:orin-base
fluxvla:orin-ros-fa
```

The FlashAttention wheel is written to:

```text
/mnt/nvme/fluxvla-wheels
```

If you need to rebuild the FlashAttention SM87 wheel:

```bash
docker/build_docker.sh wheel
docker/build_docker.sh fa
docker/build_docker.sh ros-fa
```

If you need to rebuild the ROS layer:

```bash
docker/build_docker.sh ros
docker/build_docker.sh ros-fa
```

FluxVLA source code is mounted at runtime. You do not need to rebuild the image or container after source edits. `docker/run_docker.sh` bind-mounts the current source tree to `/workspace/FluxVLA`. Editing the local Orin source, such as `/workspace/FluxVLA`, updates the code seen inside the container.

The layer relationship is:

```mermaid
flowchart TB
  subgraph BUILD["Layered build"]
    direction LR
    L4T["L4T JetPack<br/>r36.4.0"] --> BASE["orin-base<br/>system / PyTorch / Triton"]
    BASE --> FA["orin-fa<br/>FlashAttention"]
    BASE --> ROS["orin-ros<br/>ROS Noetic"]
    FA --> ROS_FA["orin-ros-fa<br/>recommended real-robot image"]
    ROS --> ROS_FA
  end

  subgraph BOTTOM[" "]
    direction LR

    subgraph CACHE["Cache"]
      direction TB
      WHEEL_CMD["build_docker.sh wheel"] --> WHEEL["/mnt/nvme/fluxvla-wheels<br/>flash_attn-*.whl"]
    end

    subgraph RUN["Runtime mounts"]
      direction TB
      SRC["Source<br/>/home/&lt;user&gt;/FluxVLA"] -.-> RUNTIME["Container<br/>/workspace/FluxVLA"]
      NVME["NVMe<br/>/mnt/nvme"] -.-> RUNTIME
    end
  end

  WHEEL -. "install wheel" .-> FA
  ROS_FA --> RUNTIME
```

View full help:

```bash
docker/build_docker.sh --help
```

## 4. Enter the Container

Use the repository script:

```bash
cd /home/<user>/FluxVLA
docker/run_docker.sh
```

Specify an image:

```bash
FLUXVLA_IMAGE=fluxvla:orin-ros-fa docker/run_docker.sh
```

Run a command directly inside the container:

```bash
FLUXVLA_IMAGE=fluxvla:orin-ros-fa docker/run_docker.sh \
  python3 -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

`docker/run_docker.sh` fills in runtime arguments, environment variables, and common mounts:

| Category                       | Automatic handling                                                                                                                 |
| ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------- |
| Docker runtime args            | `--runtime=nvidia`, `--network=host`, `--ipc=host`, `--shm-size=16g`                                                               |
| Python / inference environment | `PYTHONPATH=/workspace/FluxVLA`, `WANDB_MODE=disabled`                                                                             |
| Attention configuration        | `ATTN_IMPLEMENTATION` defaults to `flash_attention_2`; `TRANSFORMERS_ATTN_IMPLEMENTATION` follows `ATTN_IMPLEMENTATION` by default |
| Directory mounts               | Repository root mounted to `/workspace/FluxVLA`                                                                                    |
| Robotiq message package        | Auto-detects `~/robotiq_pkg/robotiq`; override with `ROBOTIQ_PY_PKG=/path/to/robotiq`. It is mounted read-only to `/opt/ros/noetic/lib/python3/dist-packages/robotiq`                  |
| ROS network variables          | If the host has `ROS_MASTER_URI`, `ROS_IP`, or `ROS_HOSTNAME`, they are passed through to the container                            |

Temporarily fall back to eager attention:

```bash
ATTN_IMPLEMENTATION=eager TRANSFORMERS_ATTN_IMPLEMENTATION=eager \
  FLUXVLA_IMAGE=fluxvla:orin-ros-fa docker/run_docker.sh
```

Confirm that host source code is really mounted into the container:

```bash
# Host
cd /home/<user>/FluxVLA
echo "MARK_$(date +%s)" > _mount_probe.txt
cat _mount_probe.txt

# Inside the container
cat /workspace/FluxVLA/_mount_probe.txt
```

Matching content means the source tree you are editing is the one mounted inside the container. Remove the probe after validation:

```bash
rm -f _mount_probe.txt /workspace/FluxVLA/_mount_probe.txt
```

## 5. Basic Acceptance Tests

### 5.1 Image and Container Basics

```bash
cd /home/<user>/FluxVLA
docker images fluxvla

FLUXVLA_IMAGE=fluxvla:orin-ros-fa docker/run_docker.sh \
  bash -lc 'pwd; python3 --version; echo "$PYTHONPATH"; ls /mnt/nvme >/dev/null && echo NVME_OK'
```

### 5.2 PyTorch / CUDA

```bash
FLUXVLA_IMAGE=fluxvla:orin-ros-fa docker/run_docker.sh python3 - <<'PY'
import torch
print('torch:', torch.__version__)
print('cuda_available:', torch.cuda.is_available())
print('torch_cuda:', torch.version.cuda)
if torch.cuda.is_available():
    print('device:', torch.cuda.get_device_name(0))
PY
```

Pass criteria: `cuda_available: True`, and the device name shows the Orin GPU.

### 5.3 Triton / FlashAttention / FluxVLA

```bash
FLUXVLA_IMAGE=fluxvla:orin-ros-fa docker/run_docker.sh python3 - <<'PY'
import triton
print('triton:', triton.__version__)

import flash_attn
print('flash_attn:', flash_attn.__version__)

import fluxvla
print('fluxvla: OK')
PY
```

If you are using `fluxvla:orin-base` or another image without FA, `flash_attn` failing is expected. In that case, use `ATTN_IMPLEMENTATION=eager` for the basic path.

### 5.4 ROS Python Environment

```bash
FLUXVLA_IMAGE=fluxvla:orin-ros-fa docker/run_docker.sh \
  python3 -c "import rospy, cv_bridge; print('ROS PY OK')"
```

Validate the custom Robotiq message package:

```bash
FLUXVLA_IMAGE=fluxvla:orin-ros-fa docker/run_docker.sh \
  bash -lc 'source /opt/ros/noetic/setup.bash && python3 -c "from robotiq.msg import StampedFloat32; print(StampedFloat32._type)"'
```

Pass criteria: output is `robotiq/StampedFloat32`.

### 5.5 GR00T Checkpoint Dummy Test

First confirm that the checkpoint file can be read by `torch.load`:

```bash
CKPT=.../gr00t_eagle_3b_ur3_full_finetune_orin/checkpoints/step-000100-epoch-00-loss=1.2629.pt

FLUXVLA_IMAGE=fluxvla:orin-ros-fa docker/run_docker.sh \
  python3 - <<PY
import torch
p = '${CKPT}'
obj = torch.load(p, map_location='cpu')
print(type(obj).__name__)
print(sorted(obj.keys())[:20] if isinstance(obj, dict) else 'non-dict')
PY
```

Then run one short inference:

```bash
FLUXVLA_IMAGE=fluxvla:orin-ros-fa docker/run_docker.sh \
  python3 scripts/test_gr00t_with_embodiment_fixed.py \
    --config configs/gr00t/gr00t_eagle_3b_ur3_full_finetune.py \
    --ckpt "$CKPT" \
    --warmup 0 \
    --predict-runs 1
```

Pass criteria: `predict_action output shape` appears and the command exits normally.

### 5.7 Pi0.5 Triton Test

An existing measured script can be run directly:

```bash
FLUXVLA_IMAGE=fluxvla:orin-fa docker/run_docker.sh \
  python3 /home/<user>/FluxVLA/test/test_models/pi05_triton_bench_real_100.py
```

The first run may record a CUDA Graph and take longer. Historical records show 100-run mean latency around `583-585 ms`.

## 6. ROS / UR3 Real-Robot Runtime Test

### 6.1 Network and ROS Environment

Use direct RJ45 gigabit Ethernet instead of a USB virtual NIC. Raw 1280x720 RGB images at 30 Hz require about 633 Mbps, while a USB2 virtual NIC usually only reaches about 7 Hz.

Verified gigabit plan:

| Device           | IP                | Description                 |
| ---------------- | ----------------- | --------------------------- |
| UR3 / ROS Master | `172.16.0.200/24` | Machine running `roscore`   |
| Orin             | `172.16.0.100/24` | Container uses host network |

Before starting camera and robot nodes on the UR3 side:

```bash
export ROS_MASTER_URI=http://172.16.0.200:11311
export ROS_IP=172.16.0.200
```

On the Orin host or inside the container:

```bash
export ROS_MASTER_URI=http://172.16.0.200:11311
export ROS_IP=172.16.0.100
```

Start the container:

```bash
cd /home/limx/sober/FluxVLA
ROS_MASTER_URI=http://172.16.0.200:11311 \
ROS_IP=172.16.0.100 \
FLUXVLA_IMAGE=fluxvla:orin-ros-fa \
docker/run_docker.sh
```

Add a fallback host-name mapping inside the container:

```bash
echo "172.16.0.200 ur3" >> /etc/hosts
rostopic list
```

Validate camera bandwidth:

```bash
rostopic hz /front_camera/color/image_raw
rostopic hz /wrist_camera/color/image_raw
```

Pass criteria: front and wrist raw images should be close to 30 Hz on a gigabit link. If they are only 4-9 Hz, first check negotiated link speed and routing on both ends:

```bash
cat /sys/class/net/eth0/speed
ip route get 172.16.0.200
```

The Orin physical network interface might not be `eth0`; replace it according to `ip -br link`.

### 6.2 Real-Robot Inference Command

Run from `/workspace/FluxVLA`, because the config contains model directories as relative paths.

```bash
cd /workspace/FluxVLA

export ROS_MASTER_URI=http://172.16.0.200:11311
export ROS_IP=172.16.0.100

PYTHONUNBUFFERED=1 python3 -u scripts/inference_real_robot.py \
  --config configs/gr00t/gr00t_eagle_3b_ur3_full_finetune.py \
  --ckpt-path /mnt/nvme/gr00t-ur3-checkpoint/fab7d175_gr00t_eagle_3b_ur3_full_finetune_bs64/checkpoints/step-042600-epoch-06-loss=0.0238.pt
```

The checkpoint path must point to a concrete `checkpoints/*.pt` file or a converted `.model.safetensors` file. Do not point it to the checkpoint root directory. The code finds `config.json`, `dataset_statistics.json`, and the tokenizer by walking two levels up from the checkpoint file.

After the model finishes loading, it enters the interactive prompt:

```text
Enter task ID (or press Enter for default):
```

Do not input the task ID automatically. Before any real robot motion, a human must confirm the robot arm, gripper, cameras, workspace, and emergency stop state.

The current default code path usually only prints `actions`; the real action dispatch call is commented out. Only enable execution after confirming that the actions are reasonable.

### 6.3 Temporary ROS Master When `roscore` Is Unavailable

Recent debugging found that the current `fluxvla:orin-ros-fa` image can run ROS Python modules, but plain `roscore` is incomplete in some environments:

```text
ModuleNotFoundError: No module named 'defusedxml'
RLException: Invalid <param> tag: Cannot load command parameter [rosversion]: no such command [['rosversion', 'roslaunch']].
```

If you only need to validate the local `rospy` registration flow, start the master directly with `rosmaster`:

```bash
FLUXVLA_IMAGE=fluxvla:orin-ros-fa docker/run_docker.sh \
  bash -lc 'python3 -m pip install -q defusedxml >/dev/null 2>&1 || true; exec rosmaster --core -p 11311'
```

The long-term fix belongs in the ROS image layer: add `defusedxml`, and make sure the `rosversion` / `roslaunch` command environment is complete so plain `roscore` works directly.

## 7. Issues and Troubleshooting Notes

Recent debugging conclusions, checkpoint conversion suggestions, Q&A, the minimum acceptance checklist, and source documents have been split into a separate document:

```text
docs/orin_docker_runtime_issue_zh-CN.md
```
