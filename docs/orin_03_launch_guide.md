# Orin FluxVLA：启动手册

更新时间：2026-07-27

本文只包含现场检查和启动命令。基础环境见
`orin_01_install_and_configuration.md`，工程记录见 `orin_02_work_record.md`。

## 1. 启动前检查

以下命令会向机器人发布动作。必须同时满足：

- 机器人周围安全，有人看守；
- 急停可用；
- Orin 已接入机器人网络；
- 当前模型需要的相机均在线；
- 同一时间只运行一个推理进程。

在 Orin 宿主机检查：

```bash
cat /sys/class/net/eno1/carrier
ip -brief addr
ping -c 1 10.192.1.185
```

`carrier` 正常应为 `1`。若现场机器人接口或地址不同，以现场网络为准。

双相机模型需要：

```text
/head/color/image_raw/compressed
/left_wrist_camera/color/image_raw/compressed
/joint/state
/brainco1/hand/state
```

Head-only 模型不需要左腕相机。

## 2. 进入容器

在操作电脑执行：

```bash
ssh limx@192.168.55.1
sudo docker start fluxvla_eval
sudo docker exec -it fluxvla_eval bash
```

进入容器后：

```bash
cd /workspace/FluxVLA
source /opt/limx/robot-tron2-r/install/setup.bash
export MROS_IP_LIST=10.192.1.185
```

若现场地址变化，修改 `MROS_IP_LIST`。

## 3. June done-dim：8 个 prompt

共同设置：

```text
相机             head + left_wrist
有效 action dim  43
done dim          42
RTC prefix        4
prompt 顺序       3 -> 0 -> 1 -> 2 -> 5 -> 4 -> 6 -> 7
final done        第 8 个 prompt 后自动停止
```

### Teacher 4-step

```bash
./scripts/inference_gr00t_rtc_wbt_done_teacher.sh --verbose
```

使用：

```text
config
configs/gr00t/gr00t_hud04_rtc_done_rtc_kernel_inference.py

checkpoint
/data/ckpts/gr00t_rtc_wbt_june_task7_0630_latesttrash_8gpu_20260630_134743_epoch30/checkpoints/step-490560-epoch-30-loss=0.0123.safetensors
```

### 同一 Teacher 直接 2-step

```bash
./scripts/inference_gr00t_rtc_wbt_done_teacher_2step.sh --verbose
```

使用与 4-step 完全相同的 teacher checkpoint，只通过 config 将
`num_inference_timesteps` 改为 `2`：

```text
config
configs/gr00t/gr00t_hud04_rtc_done_rtc_kernel_inference_2step.py

checkpoint
/data/ckpts/gr00t_rtc_wbt_june_task7_0630_latesttrash_8gpu_20260630_134743_epoch30/checkpoints/step-490560-epoch-30-loss=0.0123.safetensors
```

### Residual 蒸馏 2-step

```bash
./scripts/inference_gr00t_rtc_wbt_done_residual.sh --verbose
```

使用：

```text
config
configs/gr00t/gr00t_hud04_rtc_done_rtc_kernel_inference_residual.py

checkpoint
/data/ckpts/gr00t_rtc_wbt_june_done_4to2_residual_stage1_20260727_1205/checkpoints/step-000350-epoch-00-loss=0.0038.safetensors
```

三组对照顺序：

```text
第 1 次   teacher checkpoint    4-step   无 residual
第 2 次   teacher checkpoint    2-step   无 residual
第 3 次   distilled checkpoint  2-step   有 residual
```

不需要详细日志时去掉 `--verbose`。

正常启动日志：

```text
model loaded; waiting for frames
[warm-up] First image received. Starting model warm-up...
[warm-up] Dummy model warm-up completed ... action discarded
[task] 1/8 id=3
```

第 8 个 prompt 检测到 final done 后自动停止。提前停止按 `Ctrl-C`。

## 4. Basket：三组对照

Basket 使用 `no_done` 双相机链路。三组必须使用相同任务、物体摆放和机器人初始
姿态。每组只跑一次完整 trial；结束后按 `Ctrl-C`，恢复机器人和物体，再开始下一组。

三组顺序：

```text
第 1 次   teacher checkpoint    4-step   无 residual
第 2 次   teacher checkpoint    2-step   无 residual
第 3 次   distilled checkpoint  2-step   有 residual
```

### 第 1 次：原生 teacher 4-step

```bash
python3 -u scripts/inference_real_robot.py \
  --config configs/gr00t/gr00t_hud04_rtc_no_done_rtc_kernel_inference_4step.py \
  --ckpt-path /data/ckpts/tiga_basket_delta_20260626_101236/checkpoints/step-070578-epoch-06-loss=0.0170.safetensors
```

### 第 2 次：同一 teacher 直接 2-step

```bash
python3 -u scripts/inference_real_robot.py \
  --config configs/gr00t/gr00t_hud04_rtc_no_done_rtc_kernel_inference.py \
  --ckpt-path /data/ckpts/tiga_basket_delta_20260626_101236/checkpoints/step-070578-epoch-06-loss=0.0170.safetensors
```

第 1、2 次 checkpoint 相同，只通过 config 把
`num_inference_timesteps` 从 `4` 切换为 `2`。

### 第 3 次：4→2 residual 蒸馏模型

```bash
python3 -u scripts/inference_real_robot.py \
  --config configs/gr00t/gr00t_hud04_rtc_no_done_rtc_kernel_inference_residual.py \
  --ckpt-path /data/ckpts/tiga_basket_flow_4to2_residual_stage1/checkpoints/step-000400-epoch-00-loss=0.0039.safetensors
```

第 3 次仍为 2-step，但使用 distilled checkpoint 和
`ResidualFlowMatchingInferenceHead`。第一帧 CUDA graph capture 可能短暂停顿。

## 5. Head-only 模型

该模型只订阅头部相机：

```bash
export CONFIG=/workspace/FluxVLA/configs/gr00t/gr00t_hud04_rtc_0609_0630_headonly_6aeaeea.py
export CKPT_PATH=/data/ckpts/gr00t_rtc_wbt_0609_0630_headonly_6aeaeea_8gpu_20260717_121657_epoch30/checkpoints/step-574140-epoch-30-loss=0.0125.safetensors

./scripts/inference_gr00t_rtc_wbt_done.sh --verbose
```

正常日志应包含：

```text
Model loaded
left_wrist=disabled
First image received
Dummy model warm-up completed
```

## 6. 停止与日志

- 当前台进程需要提前结束时按 `Ctrl-C`；
- 离开容器 shell 使用 `exit`，不会停止常驻容器；
- 不要同时运行 teacher 和 residual；
- 模型首次加载通常需要约 40–60 秒；
- warm-up dummy action 会被丢弃；
- 上述启动命令本身不会自动启动 rosbag 录包，如需录包必须另开进程。

## 7. 快速排错

一直等待图像：

```text
检查 MROS_IP_LIST
检查 Orin 机器人网口
检查相机发布进程
检查对应 MROS topic
```

Head-only 仍等待左腕图像：

```text
检查 use_left_wrist_camera=False
检查 camera_names=['head']
```

容器未运行：

```bash
docker start fluxvla_eval
docker ps --filter name=fluxvla_eval
```

MROS import 失败：

```bash
python3 -m pip install /tmp/mros-2.3.1-py3-none-any.whl
```

CUDA 扩展缺失时，按装机与配置文档重新执行 `setup.py build_ext --inplace`。
