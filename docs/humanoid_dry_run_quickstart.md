# 人形推理/发布快速启动

当前没有单独的 `dry_run.py` 程序。当前调试状态用于 1-step
`/teleop_cmd_WBT` 发布测试：

1. `Teleop02WbtRTCInferenceRunner` 会打印 `actions:`，并调用
   `send_action(...)` / `send_action_absolute(...)`。
2. `Teleop02WbtOperator` 会 publish `/teleop_cmd_WBT`。
3. `/brainco1/hand/cmd` 的 publish 仍保持注释，只打印 `finger_cmd:`。
4. HUD04 RTC kernel inference 的去噪步数已对齐为 4：
   `inference_model.vla_head.num_inference_timesteps=4`。

## 1. 检查 MROS topic

宿主机：

```bash
source /opt/limx/robot-tron2-r/install/setup.bash
mrostopic list | grep -E '(/head/color/image_raw/compressed|/left_wrist_camera/color/image_raw/compressed|/joint/state|/brainco1/hand/state|/teleop_cmd_WBT)'
```

## 2. 使用已有容器运行 1-step 发布测试

本次跑通的容器：

```text
fluxvla-mros-humanoid-test
```

进入已有容器手动执行：

```bash
docker exec -it fluxvla-mros-humanoid-test bash
```

进入容器后先设置环境：

```bash
cd /workspace/FluxVLA
export MROS_IP_LIST=10.192.1.x
export WANDB_MODE=disabled
export ATTN_IMPLEMENTATION=flash_attention_2
export TRANSFORMERS_ATTN_IMPLEMENTATION=flash_attention_2
```

可选：检查容器内依赖是否齐全：

```bash
python3 - <<'PY'
for m in ("mmengine", "torch", "mros", "mros.controller_msgs.msg"):
    __import__(m)
    print(m, "OK")
PY
```

如果已经进入容器，直接运行 1-step 发布测试：

```bash
PYTHONUNBUFFERED=1 timeout --signal=INT 900s python3 -u - <<'PY'
from mmengine import Config
from fluxvla.engines import build_runner_from_cfg
import fluxvla.engines.utils.torch_utils as torch_utils

if not hasattr(torch_utils, "configure_inference_attention_defaults"):
    torch_utils.configure_inference_attention_defaults = lambda: None
torch_utils.configure_inference_attention_defaults()

cfg = Config.fromfile("configs/gr00t/gr00t_hud04_rtc_no_done_rtc_kernel_inference.py")
cfg.inference.cfg = cfg
cfg.inference.ckpt_path = "/mnt/nvme/gr00t-oli-checkpoint/gr00t_rtc_no_done_basket_delta_20260624_204819/checkpoints/step-044920-epoch-04-loss=0.0270.safetensors"

# 只发布 1 条 action，避免持续循环。
cfg.inference.max_publish_step = 1
cfg.inference.execute_horizon = 1
cfg.inference.target_hz = 1

runner = build_runner_from_cfg(cfg.inference)
runner.run_setup()
runner.run()
PY
```

如果还没进入容器，也可以从宿主机一条命令启动 1-step 发布测试：

```bash
docker exec \
  -e MROS_IP_LIST=10.192.1.x \
  -e PYTHONUNBUFFERED=1 \
  -e WANDB_MODE=disabled \
  -e ATTN_IMPLEMENTATION=flash_attention_2 \
  -e TRANSFORMERS_ATTN_IMPLEMENTATION=flash_attention_2 \
  fluxvla-mros-humanoid-test \
  bash -lc 'cd /workspace/FluxVLA && timeout --signal=INT 900s python3 -u - <<PY
from mmengine import Config
from fluxvla.engines import build_runner_from_cfg
import fluxvla.engines.utils.torch_utils as torch_utils

if not hasattr(torch_utils, "configure_inference_attention_defaults"):
    torch_utils.configure_inference_attention_defaults = lambda: None
torch_utils.configure_inference_attention_defaults()

cfg = Config.fromfile("configs/gr00t/gr00t_hud04_rtc_no_done_rtc_kernel_inference.py")
cfg.inference.cfg = cfg
cfg.inference.ckpt_path = "/mnt/nvme/gr00t-oli-checkpoint/gr00t_rtc_no_done_basket_delta_20260624_204819/checkpoints/step-044920-epoch-04-loss=0.0270.safetensors"

cfg.inference.max_publish_step = 1
cfg.inference.execute_horizon = 1
cfg.inference.target_hz = 1

runner = build_runner_from_cfg(cfg.inference)
runner.run_setup()
runner.run()
PY'
```

## 3. 直接运行 `inference_real_robot.py`

如果要直接执行入口脚本：

```bash
PYTHONUNBUFFERED=1 python3 -u scripts/inference_real_robot.py \
  --config configs/gr00t/gr00t_hud04_rtc_no_done_rtc_kernel_inference.py \
  --ckpt-path /mnt/nvme/gr00t-oli-checkpoint/gr00t_rtc_no_done_basket_delta_20260624_204819/checkpoints/step-044920-epoch-04-loss=0.0270.safetensors \
  --cfg-options \
  inference.max_publish_step=1 \
  inference.execute_horizon=1 \
  inference.target_hz=1
```

注意：

- 必须覆盖 `max_publish_step=1`；不覆盖时 no_done runner 会持续运行。
- `configure_inference_attention_defaults` 已在 `torch_utils.py` 中补齐，直接跑脚本不再需要 Python wrapper 临时 monkey patch。

看到下面日志即可说明 1-step 发布测试成功：

```text
actions: [...]
teleop_wbt_msg: ...
[ACTOR] Sent chunk 1 at action_count=0
```

## 4. 正常连续推理

如果要按配置原始 setting 正常连续推理，不要加 `--cfg-options`：

```bash
PYTHONUNBUFFERED=1 python3 -u scripts/inference_real_robot.py \
  --config configs/gr00t/gr00t_hud04_rtc_no_done_rtc_kernel_inference.py \
  --ckpt-path /mnt/nvme/gr00t-oli-checkpoint/gr00t_rtc_no_done_basket_delta_20260624_204819/checkpoints/step-044920-epoch-04-loss=0.0270.safetensors
```

当前原始 setting 包括：

- `execute_horizon=16`
- `target_hz=50`
- `async_execution=True`
- `rtc_config.enabled=True`
- `rtc_config.method='prefix'`
- `rtc_config.prefix_len=4`
- `inference_model.vla_head.num_inference_timesteps=4`

正常连续推理会持续 publish `/teleop_cmd_WBT`，停止时用 `Ctrl-C`。

## 5. 注意

- 不设置 `max_publish_step=1` 时，当前 no_done runner 会持续运行。
- 当前配置 `interactive=False`、`use_done_state_machine=False`，不会要求输入 task id，会自动使用配置里的 `task_id='0'` prompt。
- 如果新起容器缺 `mros`，可安装宿主机 wheel：`/home/limx/mros-2.3.1-py3-none-any.whl`。
