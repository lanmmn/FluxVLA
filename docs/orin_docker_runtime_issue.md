# Jetson Orin Docker Runtime Issues and Troubleshooting Notes

> Source: split from section 7 and later content in `docs/orin_docker_runtime_testing_zh-CN.md`.

## 1. Recent Debugging Conclusions

During UR3 Orin inference debugging on 2026-06-22, the target command had already progressed to the real-robot interactive prompt stage:

```bash
FLUXVLA_IMAGE=fluxvla:orin-ros-fa docker/run_docker.sh \
  python3 scripts/inference_real_robot.py \
    --config configs/gr00t/gr00t_eagle_3b_ur3_full_finetune.py \
    --ckpt-path /mnt/nvme/gr00t-ur3-checkpoint/fab7d175_gr00t_eagle_3b_ur3_full_finetune_bs64/checkpoints/step-042600-epoch-06-loss=0.0238.pt
```

Key issues that have been handled or bypassed:

- `robosuite` should not be a top-level hard dependency for the UR3 inference path.
- The Orin dependency file pins `transformers==4.53.2`; package-level version checks need to accept the Orin version instead of only accepting `5.3.0`.
- LIBERO is only needed when creating a LIBERO environment. It should not be imported at the top level of common eval utils.
- `scripts/inference_real_robot.py` should avoid importing all of `fluxvla.models` at once, otherwise it triggers training/FSDP paths.
- Orin PyTorch may lack the full `torch._C._distributed_c10d`; FSDP/DDP training module imports should treat missing distributed support as optional.
- `/wrist_camera/color/camera_info` and `/front_camera/color/camera_info` may be absent or published late; missing camera-info should not block initialization.
- The current ROS image still needs complete runtime dependencies for `roscore`; use `rosmaster --core` temporarily during validation.
- When all `actions` are `nan`, recent debugging found that the root cause is the optimized language CUDA Graph path in `EagleInferenceBackbone`: with left-padding attention masks, it can produce all-`nan` hidden states. Fully masked rows are now handled in the optimized mask construction, so `EagleInferenceBackbone` can continue to be used after validation.

Reference startup times from this validation:

```text
[Startup] build_dataset: 0.6s
[*] [Startup] build_vla: ~43s
[*] [Startup] load_checkpoint_to_cpu: ~93s
[*] [Startup] load_state_dict: ~1.4s
[Startup] build_runner_from_cfg: 148.8s
[Startup] run_setup_ros: 4.6s
[Startup] total_before_interactive_loop: 153.4s
```

So the first startup looking "slow" does not necessarily mean it is stuck, especially when a 29 GB `.pt` checkpoint requires a long `torch.load(..., map_location='cpu')`.

## 2. Checkpoint Conversion Recommendation

The `.pt` file used in UR3 debugging is about 29 GB. Actual conversion record:

```text
torch.load: 492.0s
tensors: 899
skipped non-tensors: 0
save_file: 9.2s
output size: 11G
```

If inference will be restarted often, convert the training checkpoint to `.model.safetensors` to avoid repeatedly loading optimizer, scheduler, and other training-state content.

Use the temporary conversion script:

```bash
FLUXVLA_IMAGE=fluxvla:orin-ros-fa docker/run_docker.sh \
  python3 scripts/convert_checkpoint_to_safetensors.py \
    /mnt/nvme/gr00t-ur3-checkpoint/fab7d175_gr00t_eagle_3b_ur3_full_finetune_bs64/checkpoints/step-042600-epoch-06-loss=0.0238.pt
```

Default output:

```text
/mnt/nvme/gr00t-ur3-checkpoint/fab7d175_gr00t_eagle_3b_ur3_full_finetune_bs64/checkpoints/step-042600-epoch-06-loss=0.0238.model.safetensors
```

Do not run inference at the same time during conversion, otherwise the system may load two large checkpoints at once.

## 3. Q&A

### Q1: What if `torch.cuda.is_available()` is `False` inside the container?

Check the Docker runtime first. Do not start by changing FluxVLA code:

```bash
docker info | grep -i runtime
FLUXVLA_IMAGE=fluxvla:orin-ros-fa docker/run_docker.sh \
  python3 -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

Confirm that `docker/run_docker.sh` uses `--runtime=nvidia`, the host JetPack/CUDA stack is healthy, and the image is based on a Jetson L4T container.

### Q2: What if the `fluxvla:orin-ros-fa` image does not exist?

Build in the layered order, or first use `fluxvla:orin-fa` / `fluxvla:orin` for non-ROS smoke tests.

### Q3: What if the network is unstable during build?

Enable China mirrors:

```bash
FLUXVLA_USE_CN_MIRRORS=1 docker/build_docker.sh all
FLUXVLA_USE_CN_MIRRORS=1 docker/build_docker.sh ros
```

You can also explicitly set `FLUXVLA_UBUNTU_PORTS_MIRROR`, `FLUXVLA_PIP_INDEX_URL`, and `FLUXVLA_PIP_TRUSTED_HOST`.

### Q4: What if FlashAttention reports `no kernel image is available`?

Orin is SM87. `flash-attn==2.5.5` does not always include `sm_87` in the default build. Rebuild the wheel through the unified entry point:

```bash
docker/build_docker.sh wheel
docker/build_docker.sh fa
```

This path changes the arch block to `arch=compute_87,code=sm_87`, then builds the wheel into `/mnt/nvme/fluxvla-wheels`.

### Q5: What if `rostopic list` can see topics, but `echo` / `hz` receives no data?

A common cause is that the publisher is registered in ROS master as hostname `ur3`, but the Orin container cannot resolve it, or resolves it to an unreachable network.

Temporary fix inside the container:

```bash
echo "172.16.0.200 ur3" >> /etc/hosts
```

Long-term fix: before starting publisher nodes on the UR3 side, set `ROS_MASTER_URI=http://172.16.0.200:11311` and `ROS_IP=172.16.0.200`; on the Orin side, set `ROS_IP=172.16.0.100`.

### Q6: What if image topics only reach 4 Hz / 7 Hz instead of 30 Hz?

First confirm that data uses the RJ45 gigabit link instead of a USB virtual NIC or a 100 Mbps link:

```bash
ip route get 172.16.0.200
cat /sys/class/net/eth0/speed
```

Both ends should negotiate 1000M. If the Orin only gets 100M, first replace the cable with a Cat5e/Cat6 eight-core cable and avoid 100 Mbps switches. The fallback is to subscribe to compressed image topics, such as `/front_camera/color/image_raw/compressed`.

### Q7: What if `ModuleNotFoundError: No module named 'robotiq'` occurs?

`/gripper/position` uses the custom UR3-side ROS message `robotiq/StampedFloat32`. It cannot be simply replaced by `std_msgs/Float32`.

This package is optional. Regular open-source users do not have it by default and do not need to install it. It is only required for the UR3 + Robotiq real-robot path described here, where the gripper topic is published as `robotiq/StampedFloat32`. If you only run simulation, benchmarks, non-UR3 inference, or your own robot/gripper message types, skip this step or adapt the operator to your ROS topics and message types.

The correct approach is to dereference the pure Python message package generated by UR3 catkin and copy it to the Orin:

```bash
# On the UR3 side
tar -C /home/ur3/ur_ws/devel/lib/python3/dist-packages \
  -h --exclude='__pycache__' -czf /tmp/robotiq.tgz robotiq

# After copying to the Orin
mkdir -p ~/robotiq_pkg
rm -rf ~/robotiq_pkg/robotiq
tar -C ~/robotiq_pkg -xzf /tmp/robotiq.tgz
find ~/robotiq_pkg/robotiq -type l -print
```

`find` should no longer show symlinks. `docker/run_docker.sh` auto-detects `~/robotiq_pkg/robotiq` and mounts it into ROS site-packages. If it is stored elsewhere, set `ROBOTIQ_PY_PKG=/path/to/robotiq` when launching the container.

Validate:

```bash
FLUXVLA_IMAGE=fluxvla:orin-ros-fa docker/run_docker.sh \
  bash -lc 'source /opt/ros/noetic/setup.bash && python3 -c "from robotiq.msg import StampedFloat32; print(StampedFloat32._type)"'
```

### Q8: What if `ImportError: libboost_python310.so.1.74.0` occurs?

This means the Boost Python / Regex runtime libraries needed by `cv_bridge_boost.so` are missing from the final ROS+FA image. Add them in the ROS image layer:

```bash
apt-get update && apt-get install -y libboost-python1.74.0 libboost-regex1.74.0
```

Installing them temporarily inside the container is only suitable for debugging. The long-term fix should go into the Dockerfile.

### Q9: What if `roscore` reports missing `defusedxml` or `rosversion`?

For temporary validation, bypass `roscore` and start `rosmaster` directly:

```bash
FLUXVLA_IMAGE=fluxvla:orin-ros-fa docker/run_docker.sh \
  bash -lc 'python3 -m pip install -q defusedxml >/dev/null 2>&1 || true; exec rosmaster --core -p 11311'
```

The long-term fix is to repair the ROS image layer so `defusedxml`, `rosversion`, and `roslaunch` are all available.

### Q10: What if inference looks stuck after startup but there is no error?

There are three common cases:

1. It is loading a 29 GB `.pt` checkpoint. `torch.load(..., map_location='cpu')` can take several minutes.
2. The program has reached `input()`, but the prompt is buried by stderr logs or stdout buffering. Use `PYTHONUNBUFFERED=1 python3 -u`.
3. `get_frame()` is waiting for a ROS topic. If the screen repeatedly prints `2/3/5/6/g`, these correspond to missing wrist camera, front camera, joints, TCP pose, and gripper data.

The UR3 side needs to start at least the camera and robot control nodes, for example:

```bash
roslaunch ur_control ur_bringup.launch
```

### Q11: What if `Dataset statistics file not found` occurs?

`--ckpt-path` is wrong. It must point to a concrete `.pt` or `.model.safetensors` file under `checkpoints/`, not the model root directory.

Correct structure:

```text
fab7d175_gr00t_eagle_3b_ur3_full_finetune_bs64/
├── config.json
├── dataset_statistics.json
├── tokenizer/
└── checkpoints/
    └── step-042600-epoch-06-loss=0.0238.pt
```

### Q12: What if `URInferenceRunner is not in registry` occurs?

In recent troubleshooting, this was mostly caused by top-level import chain failures. Irrelevant paths such as FSDP/DDP, FlashAttention, LIBERO, or robosuite fail during import and prevent runner/transform/dataset registry registration from completing.

Find the first interrupted import:

```bash
cd /workspace/FluxVLA
python3 - <<'PY'
import traceback, importlib
for m in ['collators', 'datasets', 'engines', 'models', 'optimizers', 'tokenizers', 'transforms']:
    try:
        importlib.import_module(f'fluxvla.{m}')
        print('OK  ', m)
    except Exception:
        print('FAIL', m)
        traceback.print_exc()
        print('-' * 60)
PY
```

The repair direction is to make training, simulation, and FlashAttention dependencies lazy or optional so they do not block registry registration for the UR3 inference path.

### Q13: Does a `camera_info` topic timeout block inference?

After the recent local fix, it should not block inference. Missing `/wrist_camera/color/camera_info` and `/front_camera/color/camera_info` should only log warnings, and inference initialization should continue. In real deployments, camera drivers should still publish `camera_info` correctly, but the current UR3 inference path does not depend on `cam_info_dict`.

### Q14: What if host source edits do not take effect inside the container?

First confirm that you edited the source tree mounted to `/workspace/FluxVLA`. Use `_mount_probe.txt` for a two-way check. If the file matches but behavior does not change, the Python process was usually already running. Restart the inference process; Python modules do not hot-reload automatically.

### Q15: Should the old bare-metal installation document still be followed?

Not as the main path. Bare-metal conda/PyTorch/flash-attn installation documents should only be used as troubleshooting references. The current recommended path is Docker: keep the host lightweight and put complex dependencies inside images.

### Q16: What if all `actions` are `nan`?

Do not only look at `rostopic list`, because readable ROS topics do not guarantee finite model inputs or finite internal outputs. During recent UR3 Orin debugging, the raw ROS observations, `dataset_statistics.json`, checkpoint weights, and dataset outputs were all verified finite, but `raw_action` was still all `nan`.

Use the diagnostic switch to locate the first NaN:

```bash
export FLUXVLA_DEBUG_NAN=1
```

This run found:

```text
model_inputs: finite
llava_vla.last_hidden_state: finite=0/1228800
raw_action: finite=0/224
```

This means NaN first appears in the VLM backbone, not during action denormalization. Further comparison found that `EagleInferenceBackbone` produced all-`nan` hidden states. Switching back to the normal `EagleBackbone` produced:

```text
llava_vla.last_hidden_state: finite=1228800/1228800
raw_action: finite=224/224
postprocessed_actions: finite=224/224
```

The root cause is in the language CUDA Graph path of the optimized `Eagle2_5_VLInferenceForConditionalGeneration`: `ProcessPromptsWithImage` uses left padding by default. For some padding queries, the causal plus padding attention mask rows are all `-inf`; `scaled_dot_product_attention` softmax then produces `nan`, which contaminates the entire hidden state block. `ATTN_IMPLEMENTATION=eager` does not necessarily bypass this, because the inference file already uses a custom Triton/CUDA Graph path internally.

The verified fix is to handle fully masked rows inside the optimized `extract_language_feature()`:

```python
fully_masked_rows = ~combined.any(dim=-1, keepdim=True)
combined = combined | fully_masked_rows
```

After the fix, `EagleInferenceBackbone` produces finite outputs again:

```text
llava_vla.last_hidden_state: finite=1228800/1228800
raw_action: finite=224/224
postprocessed_actions: finite=224/224
```

UR3 real-robot inference can continue using the optimized backbone:

```python
inference_model = dict(
  # ...
  vlm_backbone=dict(
    type='EagleInferenceBackbone',
    vlm_path='fluxvla/models/third_party_models/eagle2_hg_model'),
)
```

If running on code without this patch, the temporary workaround is still to fall back to normal `EagleBackbone`. Long-term, the inference tokenizer/prompt can also be changed to right padding to avoid fully masked rows from left-padding queries at the source.

## 4. Minimum Acceptance Checklist

When completing one Orin Docker and runtime test pass, record at least these results:

- The target image exists in `docker images fluxvla`.
- `torch.cuda.is_available()` is `True` inside the container.
- `import triton` and `import fluxvla` succeed inside the container.
- If using an FA image, `import flash_attn` succeeds inside the container.
- If using a ROS image, `import rospy, cv_bridge` succeeds inside the container.
- If running UR3, `from robotiq.msg import StampedFloat32` succeeds inside the container.
- `rostopic list` can see UR3 topics.
- Front / wrist images are close to 30 Hz on the gigabit link.
- The GR00T dummy smoke test outputs `OK`.
- Real-robot inference at least reaches the `Enter task ID` interactive prompt. Do not automatically enter a task ID in unattended mode.

## 5. Source Documents

This document consolidates Orin-related content from the following local notes:

- `docs/orin_docker_refactor_2026-06.md`
- `docs/ur3_orin_inference_issue_2026-06-22.md`
- `/home/limx/sober/fluxvla-orin-docs/FLUXVLA_DOCKER_FINAL.md`
- `/home/limx/sober/fluxvla-orin-docs/orin-to-fluxvla-docker-phases.md`
- `/home/limx/sober/fluxvla-orin-docs/install-guide-orin.md`
- `/home/limx/sober/fluxvla-orin-docs/compatibility-analysis-v2.md`
- `/home/limx/sober/fluxvla-orin-docs/ros_noetic_orin_setup.md`
- `/home/limx/sober/fluxvla-orin-docs/orin_link/orin_link/README_network.md`
- `/home/limx/sober/fluxvla-orin-docs/orin_link/orin_link/README_gigabit_direct.md`
- `/home/limx/sober/fluxvla-orin-docs/orin_link/orin_link/README_inference.md`
- `docker/README_DOCKER_ORIN.md`
- `docker/DOCKER_ORIN_BUILD.md`
