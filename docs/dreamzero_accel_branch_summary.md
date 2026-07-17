# DreamZero Accel 分支改进概览

本文总结 `refs/remotes/yiming/dreamzero-accel` 相对 `master` 对 DreamZero
推理所做的主要改进。

## 分支关系

- 加速分支顶端：`75a8476`
- 与 `master` 的共同祖先：`5f6f195`
- 加速分支在共同祖先之后有 4 个独有提交
- 当前 `master` 在共同祖先之后另有 14 个提交

因此，本文按加速分支的 4 个独有提交进行总结，而不直接把两个分支尖端的
全部差异都视为 DreamZero 改动。直接比较分支尖端会混入 `master` 后续增加的
训练、评估和机器人平台相关功能。

## 独有提交

| 提交 | 主要内容 |
| --- | --- |
| `3ae4728` | DreamZero CFG 双卡并行及推理性能分析 |
| `a454f7a` | DiT 静态跳步与动态缓存调度 |
| `eb93b5a` | TensorRT 推理路径、构建工具及测试 |
| `75a8476` | Wan 编码器 `torch.compile` 加速及 CFG 通信优化 |

## 1. 推理性能分析

该分支为 LIBERO 评估和 DreamZero 模型内部增加了可选的性能采集能力。

LIBERO 评估层面能够记录：

- 输入预处理耗时
- 模型预测耗时
- action chunk 对应的环境执行耗时
- 单个 episode 总耗时
- CUDA allocated/reserved 显存及峰值显存

DreamZero 模型内部进一步记录：

- Wan 编码阶段耗时
- KV Cache 初始化和更新耗时
- DiT denoise forward 耗时
- scheduler step 耗时
- TensorRT 与 PyTorch forward 次数
- 实际执行和跳过的 DiT step 数量

性能数据以逐 rank JSON 文件输出，并新增
`tools/summarize_eval_profile.py`，用于统计均值、P50、P90、P95 和最大值。
同时支持通过 `FLUXVLA_NVTX_PROFILE` 添加 NVTX range，便于使用 Nsight
Systems 或 Nsight Compute 分析 GPU 热点。

## 2. 双卡 CFG 并行

DreamZero 使用 CFG 时，conditional 和 negative/unconditional 分支原本在同一
GPU 上顺序执行。该分支增加 `cfg_parallel` 模式，将两个分支拆到两张 GPU：

- `rank 0`：运行 LIBERO 环境并计算 conditional 分支
- `rank 1`：作为 worker 计算 negative/unconditional 分支
- 两个 rank 通过 CUDA tensor broadcast 接收输入
- DiT 输出通过 P2P send/recv 交换，再由 `rank 0` 完成 CFG 合并

该模式保留 16 个 scheduler denoise step 和原有 CFG 计算语义。目前存在以下
限制：

- 只支持 DreamZero
- 必须恰好使用 2 个 rank
- `rank 1` 不运行独立的 LIBERO episode

提交中的小规模验证结果显示，DreamZero Head 采样耗时约从 `3424 ms` 降至
`1738 ms`，约缩短一半。

后续提交还将 Python object broadcast 调整为序列化后的 CUDA `uint8` tensor
broadcast，使通信方式更接近官方 DreamZero serving 路径。

## 3. DiT 静态跳步与动态缓存调度

DreamZero 的主要推理成本来自每个 denoise step 中的 DiT forward。该分支在
保持 scheduler 仍执行 16 步的前提下，允许部分 step 跳过 DiT 计算，并复用最近
一次 DiT 输出。

### 静态模式

通过 `dit_compute_steps` 选择预定义的 DiT step mask。目前针对 16-step 推理
提供以下档位：

- 5 次 DiT forward
- 6 次 DiT forward
- 7 次 DiT forward
- 8 次 DiT forward

未被选中的 denoise step 仍会执行 scheduler，只是不重新运行 DiT。

### 动态模式

通过 `dynamic_cache_schedule` 启用动态调度。模型比较最近两次 action prediction
的 cosine similarity：

- similarity 高于 `0.95` 时，进入较长的跳步倒计时
- similarity 高于 `0.93` 时，进入较短的跳步倒计时
- similarity 较低时，继续执行新的 DiT forward

这使简单或稳定的 denoise 阶段能够自动减少计算，而变化较大的阶段保留更多
DiT forward。

提交记录中的 LIBERO-10 结果如下：

| 模式 | 成功率 | 总耗时 | 备注 |
| --- | ---: | ---: | --- |
| CFG 双卡并行 | 92/100 | 1 小时 42 分 16 秒 | 保留完整 DiT 计算 |
| 静态 8-step | 100/100 | 58 分 03 秒 | 16 个 scheduler step，8 次 DiT forward |
| 动态调度 | 97/100 | 42 分 08 秒 | 平均 4.06 次 DiT forward |

这些结果来自对应提交中的实验记录，性能和成功率仍会受到 checkpoint、GPU、
任务顺序及运行环境影响。

## 4. TensorRT 推理路径

该分支为 cached DreamZero DiT 增加了 TensorRT 推理路径，包括：

- DreamZero DiT ONNX 导出
- TensorRT engine 构建命令生成
- 动态 KV Cache 长度 profile
- TensorRT runtime binding 和 tensor shape 检查
- PyTorch/TensorRT backend 统计
- engine builder 和 runtime 单元测试
- TensorRT、Flash Attention 和 Transformer Engine backend 诊断日志

新增的主要文件包括：

- `tools/build_dreamzero_trt_engine.py`
- `fluxvla/models/third_party_models/dreamzero/tensorrt_utils.py`
- `test/test_models/test_dreamzero_trt_engine_builder.py`
- `test/test_models/test_dreamzero_trt_runtime.py`

TensorRT 只负责 `update_cache=False` 的重复 denoise forward。需要写入或更新
KV Cache 时仍然使用 PyTorch，作为兼容和正确性保障。

该分支还修复了 TensorRT forward 中写死的 `frame_seqlen=880` 和固定两帧
sequence length，改为使用模型配置及输入 tensor 的实际形状。这对于当前
LIBERO 配置中的 `frame_seqlen=128` 尤其重要。

提交记录中的 2×A800 LIBERO-10 结果如下：

| 模式 | 成功率 | 总耗时 | 平均 action chunk 预测耗时 |
| --- | ---: | ---: | ---: |
| Baseline | 95/100 | 3 小时 09 分 03 秒 | 约 3849.30 ms |
| 动态调度 + TensorRT | 96/100 | 36 分 14 秒 | 约 613.05 ms |

## 5. Wan 编码器 `torch.compile`

该分支为 `WanBackbone` 增加可选的 `torch.compile` 支持，覆盖：

- T5 text encoder forward
- CLIP visual encoder forward
- VAE encode

支持配置：

- `use_torch_compile_encoders`
- `torch_compile_mode`
- `torch_compile_fullgraph`
- `torch_compile_dynamic`

默认 compile mode 为 `reduce-overhead`。代码还避免在特定 TensorRT 导出环境中
同时编译 encoder，以减少 ONNX/TensorRT 构建路径的兼容问题。

最终提交记录中的“动态调度 + TensorRT + encoder compile”结果为：

- LIBERO-10 成功率：`95/100`
- 平均 predict latency：`506.50 ms`
- 测试硬件：2×A800

## 如何启用

这些能力大多是可选项，切换到该分支后不会自动全部启用。主要配置入口如下：

| 配置位置 | 参数 | 作用 |
| --- | --- | --- |
| `eval` | `profile_inference` | 开启评估性能采集 |
| `eval` | `profile_warmup_predictions` | 排除前若干次 warmup 预测 |
| `eval` | `profile_max_predictions` | 限制采集的预测次数 |
| `eval` | `profile_output_path` | 指定 profile JSON 输出路径 |
| `eval` | `cfg_parallel` | 开启双卡 CFG 并行 |
| `vla_head` | `dit_compute_steps` | 选择静态 DiT 计算次数 |
| `vla_head` | `dynamic_cache_schedule` | 开启动态 DiT 跳步 |
| `vla_head` | `trt_engine_path` | 指定 TensorRT engine，也可使用 `LOAD_TRT_ENGINE` |
| `vlm_backbone` | `use_torch_compile_encoders` | 编译 Wan 编码器 |

TensorRT 模式还需要预先导出 ONNX 并构建对应 shape 的 engine，不能只通过一个
布尔配置直接启用。

## 与 master 合并时的注意事项

该加速分支尚未合入当前 `master`。由于两个分支都在共同祖先之后继续开发，
直接 merge 或 cherry-pick 时需要重点检查：

- `fluxvla/engines/runners/libero_eval_runner.py`
- `fluxvla/models/heads/dreamzero_head.py`
- `fluxvla/models/vlas/dreamzero_vla.py`
- `fluxvla/models/backbones/vlms/wan_backbone.py`

其中 `libero_eval_runner.py` 和 `dreamzero_head.py` 改动最大，预计是主要冲突区域。
建议按“profile → CFG 并行 → DiT 跳步 → TensorRT → encoder compile”的顺序
逐项移植和验证，而不是一次性合并全部变更。
