
# Fluxvla eval at AGX Orin device on Oli platform

## Oli side：

Start camera:
```bash
ssh guest@10.192.1.3
cd camera
python hik_camera_publisher.py --side left --device-index 0
```

## Orin side：

Container start：
```bash
cd /home/limx/sober/FluxVLA

docker run --rm -it \
  --runtime=nvidia \
  --ipc=host \
  --network=host \
  --shm-size=16g \
  -e PYTHONPATH=/workspace/FluxVLA:/opt/limx/robot-tron2-r/install/bin/mrosrs/src \
  -e MROS_IP_LIST=10.192.1.x \
  -e WANDB_MODE=disabled \
  -e ATTN_IMPLEMENTATION=flash_attention_2 \
  -e TRANSFORMERS_ATTN_IMPLEMENTATION=flash_attention_2 \
  -v /home/limx/sober/FluxVLA:/workspace/FluxVLA \
  -v /mnt/nvme:/mnt/nvme \
  -v /opt/limx:/opt/limx:ro \
  -v /home/limx/mros-2.3.1-py3-none-any.whl:/tmp/mros-2.3.1-py3-none-any.whl:ro \
  -w /workspace/FluxVLA \
  fluxvla:orin-ros-fa \
  bash
```

Container setup and enter：

```bash
cd /workspace/FluxVLA
export MROS_IP_LIST=10.192.1.x
export WANDB_MODE=disabled
export ATTN_IMPLEMENTATION=flash_attention_2
export TRANSFORMERS_ATTN_IMPLEMENTATION=flash_attention_2
```

Install Mros in container:
```bash
source /opt/limx/robot-tron2-r/install/setup.bash
python3 -m pip install /tmp/mros-2.3.1-py3-none-any.whl
```

Test Mros in Orin:
```bash
mrostopic list
```

## Start Fluxvla Inference
```bash
python3 -u scripts/inference_real_robot.py \
  --config configs/gr00t/gr00t_hud04_rtc_no_done_rtc_kernel_inference.py \
  --ckpt-path <...>
```

