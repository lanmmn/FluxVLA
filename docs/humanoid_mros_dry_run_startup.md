# 人形 MROS dry-run 推理调试启动记录

本文记录 2026-07-06 人形机器人 GR00T no_done 推理调试时临时补充的环境配置和启动方式。目标是以后按本文快速启动同类调试：读取 MROS 观测、跑模型推理、打印 `actions:`，但不执行动作发布。

## 1. 安全前提

启动前先确认当前代码仍是 dry-run 状态：

- `fluxvla/engines/runners/teleop02_wbt_rtc_inference_runner.py`
  - actor 线程中保留 `print('actions:', action, flush=True)`
  - `self.ros_operator.send_action_absolute(...)` 保持注释
  - `self.ros_operator.send_action(action)` 保持注释
- `fluxvla/engines/operators/teleop02_wbt_operator.py`
  - `self.teleop_wbt_publisher.publish(teleop_msg)` 保持注释
  - `self.finger_publisher.publish(finger_msg)` 保持注释

可用下面命令快速检查：

```bash
cd /workspace/FluxVLA
python3 - <<'PY'
from pathlib import Path

checks = {
    'fluxvla/engines/runners/teleop02_wbt_rtc_inference_runner.py': [
        "print('actions:', action, flush=True)",
        '#     self.ros_operator.send_action_absolute(',
        '#     self.ros_operator.send_action(action)',
    ],
    'fluxvla/engines/operators/teleop02_wbt_operator.py': [
        '# self.teleop_wbt_publisher.publish(teleop_msg)',
        '# self.finger_publisher.publish(finger_msg)',
    ],
}

for file, needles in checks.items():
    text = Path(file).read_text()
    missing = [needle for needle in needles if needle not in text]
    if missing:
        raise SystemExit(f'{file} missing safety checks: {missing}')
    print(f'{file}: dry-run action output confirmed')
PY
```

## 2. 宿主机 MROS 环境

宿主机上先 source 人形 MROS 环境：

```bash
source /opt/limx/robot-tron2-r/install/setup.bash
```

检查关键 topic：

```bash
mrostopic list | grep -E '(/head/color/image_raw/compressed|/left_wrist_camera/color/image_raw/compressed|/joint/state|/brainco1/hand/state|/brainco1/hand/cmd|/teleop_cmd_WBT)'
```

本次确认过的关键 topic：

- `/head/color/image_raw/compressed`
- `/left_wrist_camera/color/image_raw/compressed`
- `/joint/state`
- `/brainco1/hand/state`
- `/brainco1/hand/cmd`
- `/teleop_cmd_WBT`

## 3. Docker 镜像

使用镜像：

```text
fluxvla:orin-ros-fa
```

本次最终跑通是在已有容器中：

```text
fluxvla-mros-humanoid-test
```

该容器里同时具备：

- `mmengine`
- `torch`
- `mros`
- `mros.controller_msgs`

可检查：

```bash
docker exec fluxvla-mros-humanoid-test bash -lc 'python3 - <<PY
for m in ("mmengine", "torch", "mros", "mros.controller_msgs.msg"):
    try:
        __import__(m)
        print(m, "OK")
    except Exception as e:
        print(m, "ERR", type(e).__name__, str(e))
PY'
```

## 4. 如果新起临时容器

宿主机 `~/` 下有可用 MROS wheel：

```text
/home/limx/mros-2.3.1-py3-none-any.whl
/home/limx/mros-2.2.2-py3-none-any.whl
```

新起容器时建议挂载源码、NVMe、MROS 安装目录和 wheel：

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

进入容器后：

```bash
source /opt/limx/robot-tron2-r/install/setup.bash
python3 -m pip install /tmp/mros-2.3.1-py3-none-any.whl
```

如果只缺 `mros` 顶层包，也可以先尝试仅设置：

```bash
export PYTHONPATH=/workspace/FluxVLA:/opt/limx/robot-tron2-r/install/bin/mrosrs/src:$PYTHONPATH
```

但本次临时新容器遇到过 `mros.controller_msgs` 缺失，因此推荐直接安装 wheel。

## 5. 本次 no_done dry-run 推理命令

权重：

```text
/mnt/nvme/gr00t-oli-checkpoint/gr00t_rtc_no_done_basket_delta_20260624_204819/checkpoints/step-044920-epoch-04-loss=0.0270.safetensors
```

配置：

```text
configs/gr00t/gr00t_hud04_rtc_no_done_rtc_kernel_inference.py
```

在已有容器 `fluxvla-mros-humanoid-test` 中运行：

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

# 当前挂载代码里 inference_real_robot.py 会导入该函数；
# 若 torch_utils 没有该函数，运行时临时补 no-op，不改源码。
if not hasattr(torch_utils, "configure_inference_attention_defaults"):
    torch_utils.configure_inference_attention_defaults = lambda: None
torch_utils.configure_inference_attention_defaults()

cfg = Config.fromfile("configs/gr00t/gr00t_hud04_rtc_no_done_rtc_kernel_inference.py")
cfg.inference.cfg = cfg
cfg.inference.ckpt_path = "/mnt/nvme/gr00t-oli-checkpoint/gr00t_rtc_no_done_basket_delta_20260624_204819/checkpoints/step-044920-epoch-04-loss=0.0270.safetensors"

# dry-run 调试只打一条 action，避免长时间循环。
cfg.inference.max_publish_step = 1
cfg.inference.execute_horizon = 1
cfg.inference.target_hz = 1

runner = build_runner_from_cfg(cfg.inference)
runner.run_setup()
runner.run()
PY'
```

期望日志中出现：

```text
[warm-up] Dummy model warm-up completed ...
Inference time: ...
[GET_ACTIONS] Chunk inference took ... ms (total ... ms, postprocess ... ms, ...)
actions: [...]
[ACTOR] Dry-run chunk 1 at action_count=0
```

看到 `actions:` 后可停止进程。

## 6. 本次临时环境变量说明

| 环境变量 | 用途 |
| --- | --- |
| `MROS_IP_LIST=10.192.1.x` | MROS 网络发现/通信配置 |
| `PYTHONUNBUFFERED=1` | 立即打印日志，方便观察 `actions:` |
| `WANDB_MODE=disabled` | 禁用 wandb |
| `ATTN_IMPLEMENTATION=flash_attention_2` | 指定 attention 实现 |
| `TRANSFORMERS_ATTN_IMPLEMENTATION=flash_attention_2` | 指定 transformers attention 实现 |
| `PYTHONPATH=/workspace/FluxVLA:/opt/limx/robot-tron2-r/install/bin/mrosrs/src` | 新容器内补源码和部分 MROS Python 路径 |

这些都是命令级临时设置，不会持久修改系统环境。

## 7. 本次耗时参考

从 `/tmp/copilot-tool-output-1783310640544-qzhfvd.txt` 中对应一次 action 的完整流程：

| 阶段 | 时间 |
| --- | ---: |
| `get_frame()` | 16.279 ms |
| `get_ros_observation_total` | 16.389 ms |
| `jpeg_compression` | 21.741 ms |
| `update_observation_window` | 38.274 ms |
| `dataset_transform` | 8.152 ms |
| `Inference time` | 39 ms |
| `Chunk inference took` | 165.2 ms |
| `postprocess` | 0.7 ms |
| `total` | 212.5 ms |

注意：`Inference time` 没有在计时前后包 `torch.cuda.synchronize()`；更可信的模型推理耗时看 `Chunk inference took`。
