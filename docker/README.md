# FluxVLA Docker Deployment

Four Docker images for different deployment scenarios:

| Image | Target | Use Case |
|---|---|---|
| `fluxvla-a100` | A100 / x86 GPU | Full environment: training + evaluation + inference |
| `fluxvla-server` | x86 GPU server | ZMQ inference service only (lightweight) |
| `fluxvla-orin` | AGX Orin (aarch64) | Full local inference on Jetson |
| `fluxvla-orin-client` | AGX Orin (aarch64) | Lightweight ZMQ client (no model weights needed) |

## Architecture

```
┌──────────────────────────────┐
│  A100 GPU Server (full)      │
│                              │
│  fluxvla-a100                │
│  ├─ Training (torchrun)      │
│  ├─ Evaluation (LIBERO)      │
│  ├─ ZMQ PolicyServer         │
│  └─ All dependencies         │
└──────────────────────────────┘

┌─────────────────────────┐      ZMQ TCP      ┌───────────────────────┐
│  x86 GPU Server (slim)  │◄──── (obs/action)──│  AGX Orin (Robot)     │
│                         │      :5555         │                       │
│  fluxvla-server         │                    │  fluxvla-orin-client  │
│  ├─ ZMQ PolicyServer    │                    │  ├─ RemoteInference   │
│  ├─ VLA Model (GPU)     │                    │  └─ Robot Control     │
│  └─ Serialize/Deserial  │                    └───────────────────────┘
└─────────────────────────┘
                                               ┌───────────────────────┐
                                               │  AGX Orin (Standalone) │
                                               │                       │
                                               │  fluxvla-orin         │
                                               │  ├─ VLA Model (Orin)  │
                                               │  ├─ CUDA Extensions   │
                                               │  └─ Robot Control     │
                                               └───────────────────────┘
```

## Quick Start

### Build

```bash
# A100 full environment (train + eval + inference)
bash docker/scripts/build.sh a100

# x86 inference server only (lightweight)
bash docker/scripts/build.sh server

# Orin full inference (run on Orin)
bash docker/scripts/build.sh orin

# Orin lightweight client (run on Orin)
bash docker/scripts/build.sh orin-client
```

### A100 Full Environment

**Training:**

```bash
docker run --gpus all --shm-size=64g \
  -v ./checkpoints:/app/checkpoints \
  -v ./datasets:/app/datasets \
  -v ./work_dirs:/app/work_dirs \
  fluxvla-a100:latest \
  torchrun --standalone --nnodes 1 --nproc-per-node 8 \
    scripts/train.py \
    --config configs/pi05/pi05_paligemma_libero_10_full_finetune.py \
    --work-dir ./work_dirs/pi05 \
    --cfg-options train_dataloader.per_device_batch_size=2
```

**Evaluation:**

```bash
docker run --gpus all --shm-size=64g \
  -v ./checkpoints:/app/checkpoints \
  -v ./work_dirs:/app/work_dirs \
  fluxvla-a100:latest \
  torchrun --standalone --nnodes 1 --nproc-per-node 2 \
    scripts/eval.py \
    --config configs/pi05/pi05_paligemma_libero_10_full_finetune.py \
    --ckpt-path checkpoints/model.safetensors
```

**ZMQ Inference Server:**

```bash
docker run --gpus all -p 5555:5555 \
  -v ./checkpoints:/app/checkpoints \
  fluxvla-a100:latest \
  python -m fluxvla.engines.runners.serving.serve \
    --config configs/pi05/pi05_paligemma_aloha_remote_inference.py \
    --ckpt-path checkpoints/model.safetensors
```

**Interactive Shell:**

```bash
docker run --gpus all -it --shm-size=64g \
  -v ./checkpoints:/app/checkpoints \
  -v ./datasets:/app/datasets \
  -v ./work_dirs:/app/work_dirs \
  fluxvla-a100:latest bash
```

### Lightweight Inference Server (fluxvla-server)

```bash
docker run --gpus all -p 5555:5555 \
  -v ./checkpoints:/app/checkpoints \
  -v ./work_dirs:/app/work_dirs \
  fluxvla-server:latest \
  --config configs/pi05/pi05_paligemma_aloha_remote_inference.py \
  --ckpt-path checkpoints/pi05_libero/model.safetensors \
  --host 0.0.0.0 --port 5555 --dtype bf16
```

Or with Docker Compose:

```bash
cd docker/
docker compose up server
```

### Run on Orin (Remote Client Mode)

```bash
docker run --runtime nvidia --network host \
  fluxvla-orin-client:latest \
  --config configs/pi05/pi05_paligemma_aloha_remote_inference.py
```

### Run on Orin (Local Inference)

```bash
docker run --runtime nvidia --network host \
  -v ./checkpoints:/app/checkpoints \
  -v ./work_dirs:/app/work_dirs \
  fluxvla-orin:latest \
  --config configs/pi05/pi05_paligemma_aloha_remote_inference.py \
  --ckpt-path checkpoints/pi05_libero/model.safetensors
```

## JetPack Version Support

### JetPack 5.x (default)

```bash
JETPACK_TAG=r35.4.1 bash docker/scripts/build.sh orin
```

- Base image: `nvcr.io/nvidia/l4t-jetpack:r35.4.1`
- CUDA 11.4, cuDNN 8.6
- PyTorch 2.1 (NVIDIA Jetson wheel)

### JetPack 6.x

```bash
JETPACK_TAG=r36.4.0 \
TORCH_WHL_URL=<your-jp6-pytorch-wheel-url> \
  bash docker/scripts/build.sh orin
```

- Base image: `nvcr.io/nvidia/l4t-jetpack:r36.4.0`
- CUDA 12.6, cuDNN 8.9+
- PyTorch 2.5+ (build with `jetson-containers` or use NVIDIA wheel)

## flash-attn on Jetson

`flash-attn` is **not installed** in Orin images. The codebase automatically falls back to PyTorch's `scaled_dot_product_attention` (SDPA), which works on all GPU architectures including Orin's SM 8.7.

If you need flash-attn on Orin (for maximum performance), build from source:

```bash
TORCH_CUDA_ARCH_LIST="8.7" MAX_JOBS=1 NVCC_THREADS=1 \
  pip install flash-attn==2.5.5 --no-build-isolation
```

Note: This takes 2-4 hours on Orin and may fail on some JetPack versions.

## A100 Image Details

`fluxvla-a100` is the full development environment, matching the README installation guide:

| Component | Version |
|---|---|
| Base image | `nvidia/cuda:12.4.1-devel-ubuntu22.04` |
| Python | 3.10 |
| PyTorch | 2.6.0+cu124 |
| flash-attn | 2.5.5 |
| CUDA arch | SM 8.0 (A100) |
| LIBERO | included (for eval) |
| MuJoCo | 3.3.2 |
| TensorFlow | 2.15.0 |
| EGL | pre-configured for headless rendering |

All dependencies from `requirements.txt` are included, plus EGL configuration for headless LIBERO evaluation on A100 (no display needed).

> **Note**: Use `--shm-size=64g` for multi-GPU training to avoid shared memory errors with PyTorch DataLoader workers.

## CUDA Extensions

The three CUDA extensions (`gemma_rotary_embedding_ext`, `rotary_pos_embedding_ext`, `matmul_bias_ext`) are built during `pip install -e .` in the Dockerfile.

- **A100**: `TORCH_CUDA_ARCH_LIST="8.0"`
- **Orin**: `TORCH_CUDA_ARCH_LIST="8.7"`

## Volumes

| Mount Point | Description |
|---|---|
| `/app/checkpoints` | Model checkpoints (`.safetensors`, `.pt`) |
| `/app/datasets` | Training datasets (LIBERO, ALOHA, etc.) |
| `/app/work_dirs` | Training outputs, evaluation logs |
| `/app/configs` | Configuration files (optional, already in image) |

## Networking

- **Server**: Exposes port `5555` (ZMQ REP socket)
- **Orin client**: Use `--network host` for direct network access to the server
- Ensure the Orin can reach the server IP on port 5555

## Troubleshooting

**CUDA not available in container:**
Ensure you run with `--runtime nvidia` (Jetson) or `--gpus all` (x86).

**NumPy version mismatch after build:**
Some dependencies may overwrite numpy. The Dockerfile re-pins `numpy==1.26.4` at the end, but if issues persist: `pip install numpy==1.26.4`.

**Shared memory errors during training:**
Use `--shm-size=64g` (or at least `--shm-size=16g`) when running training containers.

**wandb login inside container:**
Mount your wandb credentials or run `wandb login` interactively:
```bash
docker run --gpus all -it -v ~/.netrc:/root/.netrc fluxvla-a100 bash
```

**PyTorch wheel URL for Jetson:**
Find the correct wheel at https://developer.download.nvidia.com/compute/redist/jp/
or build using https://github.com/dusty-nv/jetson-containers.

**CUDA extension build fails on Orin:**
Set `FORCE_CUDA=1` and verify `TORCH_CUDA_ARCH_LIST="8.7"`.
