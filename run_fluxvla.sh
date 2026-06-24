#!/bin/bash
# FluxVLA Run Script for NVIDIA Orin
# This script sets up the environment and runs FluxVLA

set -e

# Activate conda environment
source /home/limx/miniconda3/bin/activate fluxvla

# Fix library path for libstdc++
export LD_LIBRARY_PATH=/home/limx/miniconda3/envs/fluxvla/lib:$LD_LIBRARY_PATH

# Set CUDA environment
export CUDA_VISIBLE_DEVICES=0

# Disable wandb by default
export WANDB_MODE=disabled

# Set EGL for headless rendering
export MUJOCO_GL=egl

echo "=========================================="
echo "FluxVLA Environment Ready"
echo "=========================================="
echo "PyTorch version:"
python3 -c "import torch; print(f\"  {torch.__version__}\")"
echo "CUDA available:"
python3 -c "import torch; print(f\"  {torch.cuda.is_available()}\")"
echo "FluxVLA:"
python3 -c "import fluxvla; print(\"  Imported successfully\")"
echo "=========================================="
echo ""
echo "Environment variables set:"
echo "  LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "  WANDB_MODE: $WANDB_MODE"
echo ""
echo "Ready to run FluxVLA commands!"
echo ""
echo "Example commands:"
echo "  # Training"
echo "  torchrun --standalone --nnodes 1 --nproc-per-node 1 \\"
echo "    scripts/train.py \\"
echo "    --config configs/pi05/pi05_paligemma_libero_10_full_finetune.py \\"
echo "    --work-dir ./work_dirs/test_run"
echo ""
echo "  # Evaluation"
echo "  torchrun --standalone --nnodes 1 --nproc-per-node 1 \\"
echo "    scripts/eval.py \\"
echo "    --config configs/pi05/pi05_paligemma_libero_10_full_finetune.py \\"
echo "    --ckpt-path checkpoints/model.safetensors"
echo ""
echo "  # Inference"
echo "  python3 scripts/inference.py \\"
echo "    --config configs/pi05/pi05_paligemma_libero_10_full_finetune.py \\"
echo "    --ckpt-path checkpoints/model.safetensors"
echo ""

# If arguments provided, execute them
if [ $# -gt 0 ]; then
    echo "Executing: $@"
    echo "=========================================="
    exec "$@"
else
    # Start interactive bash
    exec bash
fi
