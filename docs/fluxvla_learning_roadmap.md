# FluxVLA 代码技术学习路线

从「实现」与「设计」两个角度，按**从地基到上层**的顺序组织。每层标注：**对应 fluxvla 代码 / 需要的知识 / 资料 / 书籍**。这是依赖关系路线，不是时间表。

---

## L0. 工程基座：PyTorch + 配置系统

**对应代码**：`engines/utils/builder.py`、所有 `configs/`、`engines/runners/base_train_runner.py`、`collators/`

**需要的知识**
- PyTorch 核心：`nn.Module` 生命周期、`autograd`（计算图 / `grad_fn` / `autograd.Function`）、dtype/device、`state_dict`、`register_buffer`
- 混合精度：`torch.autocast`、`GradScaler`、bf16/fp16 区别
- 注册器 + 配置驱动：fluxvla 用 **mmengine 风格**的 `Registry` + `build_from_cfg`，理解 `dict(type=...)` 如何映射到类
- `torch.compile` 基础（Inductor 如何生成 Triton）

**资料 / 书籍**
- PyTorch 官方 tutorials（autograd、custom `autograd.Function`、AMP）
- mmengine 文档（Registry / Config / Runner）
- 《Deep Learning with PyTorch》(Stevens)；在线《Dive into Deep Learning》d2l.ai

---

## L1. Transformer & VLM 架构（模型设计核心）

**对应代码**：`models/backbones/llms/`（gemma、qwen、sarm）、`models/backbones/visions/`（siglip）、`models/backbones/vlms/`（paligemma、eagle、florence2、smolvlm）、`models/blocks/cross_attention_dit.py`

**需要的知识**
- Transformer 全套：self-attention、MHA/GQA、FFN、残差、LayerNorm vs **RMSNorm**
- 位置编码：尤其 **RoPE**（kernel 里的 `matmul_rope_qkv`）
- 视觉编码器：**ViT → SigLIP / DINOv2**，patch embedding、CLS token
- **VLM 融合机制**：视觉 token 如何投影进 LLM 空间（`projectors/`），PaliGemma / Qwen-VL / Eagle 的 prefix 拼接
- KV cache、causal mask、attention mask 工程细节

**资料 / 书籍**
- 《Attention Is All You Need》、《An Image is Worth 16x16 Words》(ViT)
- RoFormer (RoPE)、RMSNorm 论文、SigLIP、DINOv2、PaliGemma 技术报告
- Karpathy **nanoGPT / minGPT** 与《Let's build GPT》视频
- The Illustrated Transformer（Jay Alammar）

---

## L2. VLA 与动作生成（领域核心）

**对应代码**：`models/vlas/`（pi0/pi05、dreamzero、open_vla、llava、x_vla、smolvla）、`models/heads/flow_matching_head.py`、`engines/utils/rtc_*.py`、`engines/utils/trajectory_utils.py`

**需要的知识**
- VLA 范式：RT-2 / **OpenVLA**（离散 token 动作）→ 连续动作
- 动作头两条路线：
  - **扩散 / Flow Matching**：DDPM → **Flow Matching (Lipman)** → Rectified Flow（pi0/gr00t 动作头基础）
  - 离散 token + **FAST tokenizer**（`transforms/fast_tokenizer.py`）
- **Action chunking**：一次预测一段动作
- **RTC（Real-Time Chunking）**：把推理当 inpainting，prefix 锁定
- 具体模型：π0 / π0.5、GR00T N1.5、Diffusion Policy
- 视频动力学头（dreamzero/Wan）：`models/heads/dreamzero_head.py`

**资料 / 书籍**
- 论文：OpenVLA、Diffusion Policy、π0 / π0.5、GR00T N1 技术报告、Pi 官方 RTC blog（real_time_chunking）
- Flow Matching：Lipman《Flow Matching for Generative Modeling》、《Rectified Flow》
- 扩散入门：Lilian Weng《What are Diffusion Models?》、Calvin Luo《Understanding Diffusion Models》

---

## L3. GPU 编程与 Kernel（实现 / 性能主场）

**对应代码**：`ops/triton/`（norm、matmul、attention、rope）、`ops/cuda/matmul_bias/`、`ops/atomic_ops.py`、`models/vlas/pi05_flowmatching_inference.py`（CUDA Graph）

**需要的知识**
- **CUDA 基础**：内存层级（global/shared/register）、warp / 线程块、occupancy、coalesced access、bank conflict、Tensor Core (MMA)
- **Triton**：tile 编程模型、`tl.load/store/dot`、`@triton.autotune`、JIT、block 指针
- **Kernel fusion** 原理、**FlashAttention** 算法
- **CUDA Graph**：静态捕获、buffer 预分配
- **量化**：FP8(E4M3/E5M2)、INT8、NVFP4；per-tensor / per-channel scale；校准；**AWQ / SmoothQuant**
- cuBLASLt / CUTLASS 的角色（计算绑定交给库）

**资料 / 书籍**
- 书：**《Programming Massively Parallel Processors》(Kirk & Hwu)**；NVIDIA《CUDA C++ Programming Guide》
- **Triton 官方 tutorials**（matmul、fused-softmax、layernorm、fp8）
- **Liger Kernel** 源码（训练 Triton kernel 范本）、FlashAttention 论文
- 量化：LLM.int8()、SmoothQuant、AWQ 论文；NVIDIA FP8 formats 白皮书；TransformerEngine 文档
- **GPU MODE**（原 CUDA MODE）讲座系列

---

## L4. 分布式训练与显存

**对应代码**：`engines/runners/fsdp_train_runner.py`、`ddp_train_runner.py`、config 里 `sharding_strategy` / `enable_gradient_checkpointing`

**需要的知识**
- 并行范式：DP → **FSDP / ZeRO**（参数 / 梯度 / 优化器分片）→ TP/PP/SP
- 显存优化：activation / gradient checkpointing、offload、混合精度训练
- 通信：all-reduce / all-gather / reduce-scatter 与重叠
- 训练侧低精度：**FP8 训练**（TransformerEngine）

**资料 / 书籍**
- PyTorch FSDP 官方文档与教程
- 论文：ZeRO/DeepSpeed、Megatron-LM（TP/PP）
- **TorchTitan** 源码（FSDP2+TP+PP+float8，已集成 Liger）
- HuggingFace《The Ultra-Scale Playbook》

---

## L5. 数据管线与 Transform

**对应代码**：`datasets/parquet_dataset*.py`、`transforms/`（inputs、images、prompts、normalize、rlds_transform、fast_tokenizer）、`collators/`

**需要的知识**
- 机器人数据格式：**LeRobot dataset**、**RLDS**（Open X-Embodiment）、Parquet
- VLA 数据处理：图像 resize/normalize、proprio/action 归一化（mean_std）、prompt tokenize、action chunk 切窗
- `Dataset` / `DataLoader` / collator、`statistic`（norm stats）机制

**资料 / 书籍**
- LeRobot 仓库与文档、Open X-Embodiment / RLDS 规范
- HuggingFace `datasets`、Apache Arrow / Parquet 基础
- fluxvla 自带 `docs/` 数据相关文档

---

## L6. 机器人部署与实时执行

**对应代码**：`engines/runners/aloha_inference_runner.py`、`*_rtc_inference_runner.py`、`engines/operators/`（aloha、tron2）、`engines/runners/serving/`（serve、zmq_server、serializers、proto）

**需要的知识**
- **ROS**（topic / publisher / subscriber、message sync、`rospy.Rate`）
- 机器人控制基础：关节空间、夹爪、轨迹、控制频率、同步 vs **异步执行**（RTC + resample）
- 多频率分层控制概念（System 0/1/2，对应 Figure / Sharpa）
- **推理服务**：ZMQ REQ/REP、序列化（msgpack / protobuf）、延迟 profiling
- 推理引擎对比：TensorRT（编译 / tactic search）、vLLM（吞吐）、FlashRT（边缘实时）

**资料 / 书籍**
- ROS Wiki / ROS2 文档；《Programming Robots with ROS》
- 《Modern Robotics》(Lynch & Park)（运动学 / 轨迹，按需）
- ZeroMQ Guide；protobuf 官方文档
- Pi RTC blog、Dexmal realtime-vla、FlashRT README

---

## 依赖关系与用法

- **L0 → L1 → L2**：理解 fluxvla **设计**的主干（看懂模型在干嘛）
- **L3**：**实现 / 性能**主场，重点补 CUDA 内存模型 + 量化 + Liger 训练 kernel
- **L4 / L5 / L6**：按目标取用
  - 做训练加速：L3 + L4 + Liger / TorchTitan / TransformerEngine
  - 做实时部署：L3 + L6

### 三个最高杠杆的资料
1. **Liger Kernel 源码** —— 训练 Triton kernel 范本（对应「推理 kernel 补反向」）
2. **《Programming Massively Parallel Processors》** —— 打牢 CUDA 内存 / Tensor Core 地基
3. **Flow Matching (Lipman) + π0 论文** —— 看懂 fluxvla 动作头的数学本质
