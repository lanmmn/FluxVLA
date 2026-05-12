# Training-time RTC: PI0.5 vs GR00T

## 结论

`PI0.5` 和 `GR00T` 都支持 training-time RTC，但两者接入的位置、时间变量语义、前向结构、以及可扩展能力并不一样。

最核心的区别是：

- `GR00T` 的 training-time RTC 只是在 `FlowMatchingHead` 内把部分 action token 位置改成 clean-time，并把这些位置从 loss 里 mask 掉。
- `PI0.5` 的 training-time RTC 除了支持同样的 sampled-delay 训练，还额外支持了 VLASH 风格的 `shared_observation=True` 多分支联合训练，即同一个 observation 一次前向同时覆盖多个 delay 分支。

## 1. 配置入口不同

### GR00T

配置挂在 `model.vla_head.rtc_training_config`。

对应文档位置：

- `docs/rtc.md`

对应实现位置：

- `fluxvla/models/heads/flow_matching_head.py`

### PI0.5

配置直接挂在 `model.rtc_training_config`。

对应文档位置：

- `docs/rtc.md`

对应实现位置：

- `fluxvla/models/vlas/pi0_flowmatching.py`
- `fluxvla/models/vlas/pi05_flowmatching.py`

原因是 `PI05FlowMatching` 继承自 `PI0FlowMatching`，RTC 训练逻辑主要实现在共享基类里。

## 2. clean-time 语义不同

RTC 的本质是：

- 前缀 delay 区域不再是 noisy action
- 而是直接视为已知 clean action
- 同时这些位置不参与 loss

但两条模型路径对 time 的定义不同。

### GR00T

在 `FlowMatchingHead` 中：

- `clean_time = 1.0`
- `noisy_trajectory = (1 - t) * noise + t * actions`

因此：

- `t = 1` 表示完全干净
- delay prefix 位置会被设置成 `t=1`

对应代码：

- `fluxvla/models/heads/flow_matching_head.py`
- `fluxvla/engines/utils/rtc_training.py`

### PI0.5

在 `PI0FlowMatching / PI05FlowMatching` 中：

- `clean_time = 0.0`
- `x_t = t * noise + (1 - t) * actions`

因此：

- `t = 0` 表示完全干净
- delay prefix 位置会被设置成 `t=0`

对应代码：

- `fluxvla/models/vlas/pi0_flowmatching.py`
- `fluxvla/engines/utils/rtc_training.py`

这也是 `apply_rtc_time_conditioning(...)` 需要传不同 `clean_time` 的根本原因。

## 3. 前向结构不同

### GR00T: RTC 只改 action 分支时间条件

GR00T 的 RTC 训练发生在 `FlowMatchingHead.forward(...)` 中。

大致流程：

1. 从 backbone 得到 `input_features`
2. 构造 `state_features`
3. 采样 `noise` 和 `t_scalar`
4. 如果开启 RTC：
   - 采样 `delays`
   - 得到 per-position 的 `t`
   - 更新 `action_masks`
5. 继续走 DiT action head
6. 用更新后的 `action_masks` 计算 loss

这个版本的特点是：

- prefix observation 不需要额外拆流
- RTC 只作用在 action denoising 的局部时间条件
- global time `t_global` 仍然保持标量时间，用于 DiT AdaLN
- per-position time 只给 action encoder 使用

### PI0.5: RTC 改的是 suffix token 生成路径

PI0.5 的 RTC 训练发生在 `PI0FlowMatching.forward(...)` 中。

大致流程：

1. `embed_prefix(...)` 编码 image + language
2. `embed_suffix(...)` 编码 state + noisy actions
3. prefix/suffix 拼接后进入双 Gemma transformer
4. suffix 输出投影成 velocity
5. 用 action mask 计算 loss

RTC 影响的是：

- suffix token 对应的 noisy action 构造
- suffix token 的 time embedding
- suffix 的 loss mask

这意味着 PI0.5 的 RTC 更深地嵌在 prefix/suffix 双流结构中，而不是像 GR00T 那样主要局限在 action head 内部。

## 4. sampled-delay 训练两边都支持

两者都支持最基础的 training-time RTC：

- 每个样本随机采一个 `delay`
- delay prefix 位置设成 clean-time
- delay prefix 位置不参与 loss

对应公共工具：

- `fluxvla/engines/utils/rtc_training.py`
  - `sample_training_delay(...)`
  - `apply_rtc_time_conditioning(...)`

这部分是两者共享的 RTC 核心机制。

## 5. shared-observation 多分支训练只在 PI0.5 上加了

这是当前最大的实现差异。

### GR00T

目前 `GR00T` 的 `FlowMatchingHead.forward(...)` 仍然是：

- 一个样本
- 一个 delay
- 一次 forward

没有增加：

- `shared_observation`
- 多 delay branch 联合 loss
- cross-offset attention mask

### PI0.5

`PI0.5` 当前新增了：

- `shared_observation=True`
- `shared_observation_loss_weighting`
- 多 offset branch 一次前向联合训练
- shared prefix + blocked cross-offset suffix attention

对应代码位置：

- `fluxvla/models/vlas/pi0_flowmatching.py`
- `fluxvla/engines/utils/model_utils.py`
- `fluxvla/engines/utils/rtc_training.py`

新增能力包括：

1. 同一个 observation 同时展开多个 delay 分支
2. prefix 只保留一份
3. suffix 按多 offset 拼接成一个大 suffix 序列
4. 通过 shared-observation attention mask 阻断不同 offset 之间的 suffix attention
5. loss 可选：
   - `distribution`
   - `uniform`

换句话说：

- `GR00T` 现在是 sampled-delay RTC
- `PI0.5` 现在是 sampled-delay RTC + VLASH-style shared-observation RTC

## 6. loss 聚合方式不同

### GR00T

GR00T 的 loss 仍然是单分支 loss：

- `loss = mse(pred_actions, velocity) * action_masks`
- 再按有效 action 数归一化

没有 branch-level aggregation。

### PI0.5

PI0.5 在 `shared_observation=True` 时，多了 branch-level aggregation：

- 先对每个 delay 分支分别得到逐位置 loss
- 再按 branch weight 聚合

当前支持两种 branch 权重：

- `shared_observation_loss_weighting='distribution'`
  - 贴近原 sampled RTC 目标
- `shared_observation_loss_weighting='uniform'`
  - 更接近 VLASH 的“每个 offset 等权”

## 7. attention 结构差异决定了扩展难度

### GR00T

GR00T 的 RTC 训练主要发生在 `FlowMatchingHead` 内部的 action encoder / DiT 路径。

优点：

- 实现简单
- 风险小

缺点：

- 想扩成 VLASH-style shared-observation multi-branch，需要重构 head 内部对多分支 action token 的组织方式

### PI0.5

PI0.5 本身就有明确的：

- prefix tokens
- suffix tokens
- 双流 attention 结构

因此更适合做：

- shared prefix
- multi-offset suffix
- cross-offset suffix attention blocking

所以这次 shared-observation RTC 先落在 PI0.5，是结构上更自然的选择。

## 8. 配置上的实际区别

### GR00T 示例

```python
model = dict(
    vla_head=dict(
        rtc_training_config=dict(
            enabled=True,
            max_delay=7,
            distribution='exponential',
            temperature=1.0,
        )))
```

### PI0.5 普通 RTC 示例

```python
model = dict(
    type='PI05FlowMatching',
    rtc_training_config=dict(
        enabled=True,
        max_delay=7,
        distribution='exponential',
        temperature=1.0,
    ))
```

### PI0.5 VLASH-style RTC 示例

```python
model = dict(
    type='PI05FlowMatching',
    rtc_training_config=dict(
        enabled=True,
        max_delay=7,
        shared_observation=True,
        distribution='exponential',
        shared_observation_loss_weighting='distribution',
        temperature=1.0,
    ))
```

## 9. 一句话总结

如果只看“training-time RTC”这个最基本机制：

- `GR00T` 和 `PI0.5` 都是“sample delay -> clean prefix -> mask loss”

如果看“这次项目里真正加出来的能力”：

- `GR00T` 还是单 delay sampled RTC
- `PI0.5` 已经扩展成了 VLASH-style shared-observation multi-branch RTC

所以两者最本质的区别不是“有没有 RTC”，而是：

- `GR00T` 的 RTC 是轻量级 head 内时间条件修改
- `PI0.5` 的 RTC 已经进入 prefix/suffix token 级别，并支持共享 observation 的多分支联合训练
