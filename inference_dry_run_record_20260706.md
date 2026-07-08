# GR00T no_done 推理 dry-run 记录

时间：2026-07-06 12:08

## 本次实际修改

本次没有修改仓库源码，也没有提交任何代码改动。

实际做的是一次只打印 action、不执行控制命令的推理 dry-run：

- 使用配置：`configs/gr00t/gr00t_hud04_rtc_no_done_rtc_kernel_inference.py`
- 使用权重：`/mnt/nvme/gr00t-oli-checkpoint/gr00t_rtc_no_done_basket_delta_20260624_204819/checkpoints/step-044920-epoch-04-loss=0.0270.safetensors`
- 使用镜像/容器：`fluxvla:orin-ros-fa`
- 使用 mros 环境：`source /opt/limx/robot-tron2-r/install/setup.bash`
- 运行时临时覆盖推理参数：
  - `inference.max_publish_step = 1`
  - `inference.execute_horizon = 1`
  - `inference.target_hz = 1`

## 安全确认

推理前确认 action 输出端处于 dry-run 状态：

- `fluxvla/engines/runners/teleop02_wbt_rtc_inference_runner.py`
  - actor 线程会 `print('actions:', action, flush=True)`
  - `self.ros_operator.send_action_absolute(...)` 已注释
  - `self.ros_operator.send_action(action)` 已注释
- `fluxvla/engines/operators/teleop02_wbt_operator.py`
  - `self.teleop_wbt_publisher.publish(teleop_msg)` 已注释
  - `self.finger_publisher.publish(finger_msg)` 已注释

因此本次推理只打印 action，没有执行机器人动作发布。

## topic 检查

通过 `mrostopic list` 确认了所需 topic 存在，包括：

- `/head/color/image_raw/compressed`
- `/left_wrist_camera/color/image_raw/compressed`
- `/joint/state`
- `/brainco1/hand/state`
- `/brainco1/hand/cmd`
- `/teleop_cmd_WBT`

## 运行结果

推理成功输出了 `actions:`，日志中也出现了 `Dry-run chunk`，例如：

```text
actions: [-1.19348622e-01  3.50629803e-02 ... 7.61718750e-01  7.07031250e-01]
[ACTOR] Dry-run chunk 1 at action_count=0
```

完整输出文件：

```text
/tmp/copilot-tool-output-1783310640544-qzhfvd.txt
```

推理进程已停止，没有遗留 dry-run 进程。
