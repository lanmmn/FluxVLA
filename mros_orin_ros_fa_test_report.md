# mros on `fluxvla:orin-ros-fa` communication test report

## Summary

- Test time: 2026-07-05 14:55 +08:00
- Docker image: `fluxvla:orin-ros-fa`
- Image ID: `sha256:f711443c00a6a0618d12269ccc5f27a771ac8e63ba64793a2e3569d2525c09e6`
- Image platform: `linux/arm64`
- Container: `mros-whl-test-1783234485`
- Wheel: `/home/limx/mros-2.3.1-py3-none-any.whl`
- Python: `3.10.12`
- Runtime arch: `aarch64`
- Installed package: `mros==2.3.1`

Result: **PASS**

## Current run

### Installation/import

`mros-2.3.1-py3-none-any.whl` was installed into a temporary `fluxvla:orin-ros-fa` container.

```text
Python 3.10.12
aarch64
mros_import=ok
mros_file=/usr/local/lib/python3.10/dist-packages/mros/__init__.py
```

### Single-process pub/sub

- Topic: `/copilot_mros_single_test`
- Message type: `mros.std_msgs.msg.String`
- Publisher node: `mros_single_process_test`
- Payload: `hello-single-process`

```json
{
  "test": "single_process_pub_sub",
  "ok": true,
  "received": [
    {
      "data": "hello-single-process",
      "callerid": "mros_single_process_test"
    }
  ],
  "subscribers": 1
}
```

### Two-process pub/sub

- Topic: `/copilot_mros_two_process_test`
- Message type: `mros.std_msgs.msg.String`
- Subscriber node: `mros_subscriber_test`
- Publisher node: `mros_publisher_test`
- Payload: `hello-two-process`

```json
{
  "test": "two_process_pub_sub",
  "ok": true,
  "received_count": 21,
  "callerid": "mros_publisher_test"
}
```

## Current container

The test container is still running:

```bash
docker exec -it mros-whl-test-1783234485 bash
```

## Related session records

The session history contains prior related mros tests:

1. `2026-07-05T03:23:18Z`, session `48db5bfb-827c-47e3-ae39-4cb23110fbe7`, turn `1`
   - Used `/home/limx/mros-2.3.1-py3-none-any.whl` with `fluxvla:orin-ros-fa`.
   - Result: `mros` import passed; `std_msgs/String` single-process and two-process pub/sub both passed.
   - Previous container: `mros-temp-test-1783221822`.

2. `2026-07-05T03:34:01Z`, session `48db5bfb-827c-47e3-ae39-4cb23110fbe7`, turn `2`
   - Installed `mros-2.3.1` in a `fluxvla:orin-ros-fa` container and tested bag playback plus model inference.
   - Result: bag to mros to runner input passed; model checkpoint loaded with `missing=0, unexpected=0`; RTC inference produced `/teleop_cmd_WBT` and `/brainco1/hand/cmd` messages.
   - Previous container: `fluxvla-mros-infer-test-1783222514`.

3. `2026-07-03T17:42:46Z`, session `1de4be9a-7fc4-4910-9ab7-070441f5d593`, turn `0`
   - Tested `/home/limx/mros-2.2.2-py3-none-any.whl` in the Orin image.
   - Result: installation succeeded but runtime failed because bundled native libraries were `x86_64`, incompatible with the arm64/aarch64 image.

## 2026-07-05 humanoid GR00T/FluxVLA weight test

### Environment

- Test time: 2026-07-05 15:36 +08:00
- Docker image: `fluxvla:orin-ros-fa`
- Image ID: `sha256:f711443c00a6a0618d12269ccc5f27a771ac8e63ba64793a2e3569d2525c09e6`
- Container: `fluxvla-mros-humanoid-test`
- Runtime: `--runtime=nvidia --network host --ipc host --privileged`
- Python: `3.10.12`
- Runtime arch: `aarch64`
- CUDA device: `Orin`
- PyTorch: `2.5.0a0+872d972e41.nv24.08`
- Wheel: `/home/limx/mros-2.3.1-py3-none-any.whl`
- Installed package: `mros==2.3.1`

The requested path `/mnt/nvme/gr00t-oli` did not exist on the host/container. The matching humanoid checkpoint was found under `/mnt/nvme/gr00t-oli-checkpoint`.

### Container startup

```bash
docker run -d --name fluxvla-mros-humanoid-test \
  --network host --ipc host --privileged --runtime=nvidia \
  -v /home/limx/sober/FluxVLA:/workspace/FluxVLA \
  -v /home/limx/mros-2.3.1-py3-none-any.whl:/tmp/mros-2.3.1-py3-none-any.whl:ro \
  -v /mnt/nvme:/mnt/nvme \
  -w /workspace/FluxVLA \
  fluxvla:orin-ros-fa sleep infinity
```

### mros installation/import

```text
Python 3.10.12
aarch64
mros_import=ok
mros_file=/usr/local/lib/python3.10/dist-packages/mros/__init__.py
```

### Weight test

- Config: `configs/gr00t/gr00t_hud04_rtc_no_done_rtc_kernel_inference.py`
- Checkpoint: `/mnt/nvme/gr00t-oli-checkpoint/gr00t_rtc_no_done_basket_delta_20260624_204819/checkpoints/step-044920-epoch-04-loss=0.0270.safetensors`
- Backbone: `EagleInferenceBackbone`
- Head: `FlowMatchingInferenceHead`
- Dummy input: batch size `1`, views `2`, image size `224`, language length `580`
- Warmup: `2`
- Predict runs: `10`

```bash
python3 /tmp/test-gr00t-100times.py \
  --config configs/gr00t/gr00t_hud04_rtc_no_done_rtc_kernel_inference.py \
  --ckpt /mnt/nvme/gr00t-oli-checkpoint/gr00t_rtc_no_done_basket_delta_20260624_204819/checkpoints/step-044920-epoch-04-loss=0.0270.safetensors \
  --warmup 2 --predict-runs 10 --num-views 2 --lang-len 580
```

### Result

Result: **PASS**

```text
load_state_dict: missing=0, unexpected=0
CUDA: Orin
Head: FlowMatchingInferenceHead
Backbone: EagleInferenceBackbone
image_token_id=151669, image_token_count=512
predict_action output shape: (1, 32, 42)
latency_ms: min=123.008 max=124.287 mean=123.883 median=123.958 stdev=0.375
total_wall_predict=1238.828 ms (10 runs, excl. warmup)
OK
```

The running container can be entered with:

```bash
docker exec -it fluxvla-mros-humanoid-test bash
```
