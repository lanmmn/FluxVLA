# Orin FluxVLA：工作记录

更新时间：2026-07-29

本文记录已经完成的工程工作、模型部署和验证结论。基础搭建见
`orin_01_install_and_configuration.md`，现场命令见 `orin_03_launch_guide.md`。

## 1. 基础环境

### 2026-07-10 至 2026-07-11

- 确认 Jetson AGX Orin 的 L4T/Ubuntu 底座无需重刷；
- 格式化并挂载 1 TB NVMe 到 `/data`，同时 bind 到 `/mnt/nvme`；
- 安装 Docker、buildx 与 NVIDIA container runtime；
- 将 Docker/containerd 数据迁到 NVMe，解决 eMMC 满盘；
- 修复 dlimp、flash-attn wheel 和 ros-fa 构建链问题；
- 完成 `fluxvla:orin-ros-fa` 镜像构建与 GPU/flash-attn/ROS 验证。

### 2026-07-14 至 2026-07-15

- 部署 aarch64 robot-tron2-r 和 MROS；
- 编译 `gemma_rotary_embedding_ext`、`rotary_pos_embedding_ext`、
  `matmul_bias_ext`；
- 创建常驻容器 `fluxvla_eval`；
- 模型加载、CUDA、flash-attn、MROS 和相机订阅链路通过；
- 定位 MODE_30W 为推理偏慢原因，切换到 MAXN 并配置开机锁频。

## 2. Basket teacher 与 4→2 residual

### Checkpoint

Teacher：

```text
/data/ckpts/tiga_basket_delta_20260626_101236/checkpoints/step-070578-epoch-06-loss=0.0170.safetensors
```

4→2 residual：

```text
/data/ckpts/tiga_basket_flow_4to2_residual_stage1/checkpoints/step-000400-epoch-00-loss=0.0039.safetensors
```

### 端侧适配

新增 residual kernel head：

```text
fluxvla/models/heads/residual_flow_matching_inference_head.py
```

Residual MLP：

```text
85 -> 256 -> 40
residual_max_abs = 0.1
```

Residual 在每个 Euler step 的 continuous velocity 上生效。必须集成到 kernel
head 的 `record_run` 路径，否则端侧 CUDA graph 会绕过训练侧的 `denoise_step`。

配置：

```text
teacher 4-step
configs/gr00t/gr00t_hud04_rtc_no_done_rtc_kernel_inference_4step.py

teacher 直接 2-step
configs/gr00t/gr00t_hud04_rtc_no_done_rtc_kernel_inference.py

residual 2-step
configs/gr00t/gr00t_hud04_rtc_no_done_rtc_kernel_inference_residual.py
```

已验证：

```text
teacher 4-step config     FlowMatchingInferenceHead / 4 steps
teacher 2-step config     FlowMatchingInferenceHead / 2 steps
residual 2-step config    ResidualFlowMatchingInferenceHead / 2 steps
residual strict load      通过
CUDA graph capture        通过
输出 shape                (1, 32, 42)
输出 finite               通过
```

## 3. Head-only 模型适配

Checkpoint：

```text
/data/ckpts/gr00t_rtc_wbt_0609_0630_headonly_6aeaeea_8gpu_20260717_121657_epoch30/checkpoints/step-574140-epoch-30-loss=0.0125.safetensors
```

SHA-256：

```text
f6f445d20c67e3046da03e24016a4700abf9a9b01fa607998f18e20170cc936d
```

新增配置：

```text
configs/gr00t/gr00t_hud04_rtc_0609_0630_headonly_6aeaeea.py
```

适配内容：

- 图像输入由 `head + left_wrist` 改为仅 `head`；
- `NormalizeImages` 的 mean/std 修正为一组；
- operator 支持 `use_left_wrist_camera=False`；
- runner 的 warm-up、observation、JPEG 和 debug image 支持单相机；
- `FlowMatchingHead` 兼容训练配置中的占位参数。

修改前备份：

```text
/data/FluxVLA-headonly-backup-20260724-121508
```

验证结果：

```text
checkpoint strict load       通过
完整 runner 初始化           通过
左腕相机                     disabled
头部图像接收                 通过
单图像预处理                 通过
CUDA/bfloat16                 通过
warm-up                      通过
predict_action               约 770 ms
dummy action                 已丢弃，未发给机器人
```

## 4. June done-dim 4→2 residual

### 数据与训练

Teacher：

```text
/data/ckpts/gr00t_rtc_wbt_june_task7_0630_latesttrash_8gpu_20260630_134743_epoch30/checkpoints/step-490560-epoch-30-loss=0.0123.safetensors
```

训练时选择 `wbt_done_dim_0610` 至 `wbt_done_dim_0630` 的逐日标准目录：

- 排除 3frame、其他后缀格式和聚合目录 `wbt_done_dim_0609_0630`；
- 抓垃圾任务只保留 0628；
- 抓钱包任务只保留 0629；
- 同日期的其他任务保留；
- 最终 959,943 行。

Teacher action 有效维度为 43：

```text
continuous   0:40
hands        40:42
done         42
```

Residual：

```text
输入          87 = 2 * 43 + 1
隐藏层        256
输出          40
训练          8 GPU FSDP，500 steps
```

训练目录：

```text
/mnt/data/cpfs/bob/fluxvla/outputs/gr00t_rtc_wbt_june_done_4to2_residual_stage1_20260727_1205
```

最初只正式训练了第 3 种零初始化 residual。2026-07-27 18:03 起，按相同
teacher、相同 959,943 行筛选数据补跑另外两种蒸馏方法：

```text
方法一 stage 1   全 action-head teacher-forced progressive，10000 steps
方法一 stage 2   从 stage 1 接续的 on-policy correction，3000 steps
方法二           连续维优先的全 action-head 重训，3000 steps
方法三           零初始化 residual，500 steps
```

新增三种训练配置均已通过 8 卡、2-step 冒烟，包括 43 维 action、非零梯度和
checkpoint 保存。正式队列于 2026-07-27 20:50（北京时间）完成训练，于 20:58
完成统一 endpoint 评测；所有训练和评测退出码均为 0，并生成
`ALL_TRAINING_COMPLETE` 与 `ALL_EVAL_COMPLETE`。

正式队列状态目录：

```text
/mnt/data/cpfs/bob/fluxvla/outputs/gr00t_rtc_wbt_june_done_4to2_other_methods_20260727_1803
```

最终 checkpoint：

```text
方法一 stage 1
/mnt/data/cpfs/bob/fluxvla/outputs/gr00t_rtc_wbt_june_done_4to2_progressive_stage1_20260727_1803/checkpoints/step-010000-epoch-00-loss=0.0265.safetensors

方法一 stage 2
/mnt/data/cpfs/bob/fluxvla/outputs/gr00t_rtc_wbt_june_done_4to2_progressive_stage2_20260727_1803/checkpoints/step-003000-epoch-00-loss=0.0296.safetensors

方法二
/mnt/data/cpfs/bob/fluxvla/outputs/gr00t_rtc_wbt_june_done_4to2_continuous_stage1_20260727_1803/checkpoints/step-003000-epoch-00-loss=0.0121.safetensors

方法三
/mnt/data/cpfs/bob/fluxvla/outputs/gr00t_rtc_wbt_june_done_4to2_residual_stage1_20260727_1205/checkpoints/step-000350-epoch-00-loss=0.0038.safetensors
```

### 三种方法的统一离线评测

评测条件相同：256 个样本、seed 7、归一化 action space；reference 为原始
4-step teacher，baseline 为同一 teacher 直接使用 2-step。以下数值为 normalized
MSE，越小越好；括号内为相对 baseline 的改善率：

```text
改善率 = (baseline MSE - distilled MSE) / baseline MSE
```

实际端侧使用的 RTC prefix 4：

| 模型 | 全 43 维 | 连续 40 维 | hands 2 维 | done 1 维 |
|---|---:|---:|---:|---:|
| teacher 直接 2-step baseline | 0.000258743 | 0.000228796 | 0.000844900 | 0.000284302 |
| 方法一 progressive stage 2 | 0.001088632 (-320.74%) | 0.000967416 (-322.83%) | 0.003494685 (-313.62%) | 0.001125136 (-295.75%) |
| 方法二 continuous | 0.000891234 (-244.45%) | 0.000837906 (-266.22%) | 0.002143994 (-153.76%) | 0.000518832 (-82.49%) |
| 方法三 residual step 350 | 0.000205943 (+20.41%) | 0.000171980 (+24.83%) | 0.000844544 (+0.04%) | 0.000287232 (-1.03%) |

无 RTC prefix 的 prefix 0 对照：

| 模型 | 全 43 维 | 连续 40 维 | hands 2 维 | done 1 维 |
|---|---:|---:|---:|---:|
| teacher 直接 2-step baseline | 0.000288004 | 0.000256189 | 0.000883208 | 0.000370217 |
| 方法一 progressive stage 2 | 0.001492466 (-418.21%) | 0.001401837 (-447.19%) | 0.002964679 (-235.67%) | 0.002173177 (-487.00%) |
| 方法二 continuous | 0.001129819 (-292.29%) | 0.001097256 (-328.30%) | 0.001684448 (-90.72%) | 0.001323097 (-257.38%) |
| 方法三 residual step 350 | 0.000237003 (+17.71%) | 0.000201158 (+21.48%) | 0.000889231 (-0.68%) | 0.000366376 (+1.04%) |

方法一和方法二虽已完成训练，但 endpoint normalized MSE 均显著差于直接
2-step baseline。方法三是本轮唯一改善全维与连续维 MSE 的方案。这里的
teacher-matching 结果不是机器人成功率，仍需真机对照验证。

### 模型选择

端侧选用 step 350：

```text
/data/ckpts/gr00t_rtc_wbt_june_done_4to2_residual_stage1_20260727_1205/checkpoints/step-000350-epoch-00-loss=0.0038.safetensors
```

相同 256 样本、seed 7、RTC prefix 4 的 endpoint teacher-matching：

```text
                           step 350    step 500
全 43 维 MSE 改善           20.41%      20.60%
连续 40 维 MSE 改善         24.83%      25.17%
done MSE 相对变化           -1.03%      -2.47%
```

step 350 的整体指标与 step 500 接近，但 done 退化更小、hands 更稳，因此用于
依赖 done 自动切换的 8-prompt 端侧运行。以上是归一化 endpoint teacher-matching，
不是机器人成功率。

SHA-256：

```text
teacher
3c14ce2f40d45d89f8c84fca88775dcd8c52a722e14ca0433560394c8f79dc4f

residual step 350
e76a019bcf3ba56a454702671ce01c8ca76dad30daf3d84fb3de367dabf533a5
```

### USB 部署与端侧验证

为减少 USB 传输，只传输 4 个 residual tensor，在 Orin 上与已有 teacher 合成完整
903-tensor checkpoint；合成文件与 A800 原文件 SHA-256 完全一致。

端侧配置与入口：

```text
teacher 4-step
configs/gr00t/gr00t_hud04_rtc_done_rtc_kernel_inference.py
scripts/inference_gr00t_rtc_wbt_done_teacher.sh

teacher 直接 2-step
configs/gr00t/gr00t_hud04_rtc_done_rtc_kernel_inference_2step.py
scripts/inference_gr00t_rtc_wbt_done_teacher_2step.sh

residual 2-step
configs/gr00t/gr00t_hud04_rtc_done_rtc_kernel_inference_residual.py
scripts/inference_gr00t_rtc_wbt_done_residual.sh
```

无动作预检结果：

```text
teacher/residual strict load       通过
完整 903 tensor strict load        通过
CUDA graph capture                 通过
输出 shape                         (1, 32, 43)
输出 finite                        通过
GPU bf16                           通过
MROS 初始化                        通过
双相机订阅创建                     通过
8-prompt 顺序                      通过
```

### 手部状态缺失时的临时命令回填

2026-07-29 现场检查发现机器人未连接手部，MROS 发现表中存在
`/brainco1/hand/state`，但该 topic 不出消息。June done-dim checkpoint 的手部
state/action 均为左右手 open/closed 两维，因此为 8-prompt 增加临时 fallback：

- `Teleop02WbtOperator` 新增 `use_finger_state`，默认仍为 `True`；
- `use_finger_state=False` 时不订阅、也不等待物理 hand-state；
- 首次推理输入的手状态为 `[0, 0]`，即左右手均张开；
- 后续输入使用最近一次实际发送的模型 action `[40:42]`，经 `0.5` 阈值离散化后
  回填为下一轮手状态；
- 手部 command 仍按原链路发布，回填值表示模型的最近命令，不代表机械手真实反馈；
- 该 fallback 只允许用于 `hand_state_dim=2`，防止 12 维原始手状态配置误用。

启用范围仅限 June 8-prompt 三组对照：

```text
teacher 4-step    use_finger_state=False
teacher 2-step    继承 teacher 4-step 配置
residual 2-step   use_finger_state=False
```

验证结果：

```text
三组 config 继承检查              通过
左右手 action -> state 回填        通过
双相机 + joint 在线只读取帧        通过
输出 state                         33 维，初始 hand=[0, 0]
物理 hand-state 等待               已跳过
机器人动作                         未发布
```

这是物理手部反馈缺失时的临时部署模式。如果手部执行链路也不可用，涉及抓取的
prompt 无法真正完成，done 自动切换也可能失效。手部反馈恢复后，应删除配置中的
override 或将 `use_finger_state` 恢复为 `True`。

### 双相机 runtime warm-up 配置修复

2026-07-29 首次按启动手册运行 residual 2-step 时，完整模型加载成功，但双相机
warm-up 暴露出 June done-dim 公共 inference 配置中的三处不一致：

```text
图像数量              2
NormalizeImages 参数  3 组
prompt padding        900
Eagle CUDA buffer     900
action-head buffer    580
```

仅修改 `gr00t_hud04_rtc_done_full_finetune.py` 的 runtime inference 部分：

- `NormalizeImages` 的 mean/std 从 3 组改为与 `head + left_wrist` 对应的 2 组；
- `ProcessPromptsWithImage.max_len` 从 900 改为 580；
- `inference_model.vlm_backbone.vlm_config.max_input_seq_len` 从 900 改为 580；
- 训练模型配置和训练数据 transforms 保持不变。

Teacher 4-step、Teacher 2-step 和 residual 2-step 均继承该修复。验证结果：

```text
三组 config 图像归一化参数          means=2, stds=2
三组序列长度                         dataset=580, Eagle=580, head=580
双相机 dataset-only 输出             (1, 6, 224, 224)
residual 完整 checkpoint 加载         通过
真实双相机 + joint warm-up            通过
dataset transform                     约 23.8 ms
dummy predict_action                  约 3.71 s
dummy action                          已丢弃
机器人动作                            未发布
```

`setup.bash` 输出的若干缺失 package `local_setup.bash` 为现有 release 的非致命
警告，不是本次 warm-up 失败原因。

## 5. 当前边界

- 工程预检不会替代真机成功率评估；
- teacher 与 residual 不能同时向同一机器人发布动作；
- June residual 的定量改善是离线 teacher-matching，不是真机成功率；
- 正式运行前必须重新确认机器人网络、两路相机、环境安全和急停；
- FluxVLA 工作树包含端侧适配，更新仓库前应先备份或提交这些改动。
