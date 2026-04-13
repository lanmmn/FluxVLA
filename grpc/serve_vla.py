"""Launch a gRPC server that serves a real VLA model for remote inference.

Usage::

    python grpc/serve_vla.py \
        --config configs/pi05/pi05_paligemma_libero_10_full_finetune.py \
        --ckpt-path /path/to/checkpoint.pt \
        --host 0.0.0.0 --port 50051
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

# Ensure project root is importable
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
# Ensure grpc/ is importable for pb2 modules
_grpc_dir = str(Path(__file__).resolve().parent)
if _grpc_dir not in sys.path:
    sys.path.insert(0, _grpc_dir)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Serve a VLA model via gRPC for remote inference")
    parser.add_argument("--config", required=True,
                        help="Path to mmengine config file")
    parser.add_argument("--ckpt-path", required=True,
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0",
                        help="Device to run model on (default: cuda:0)")
    parser.add_argument("--dtype", default="bf16",
                        choices=["bf16", "fp16", "fp32"],
                        help="Mixed precision dtype (default: bf16)")
    return parser.parse_args()


def main():
    args = parse_args()

    # --- Load config and build model ---
    from mmengine import Config
    from fluxvla.engines import build_vla_from_cfg

    cfg = Config.fromfile(args.config)

    print(f"[serve_vla] Building VLA model from config ...")
    if hasattr(cfg, "inference_model"):
        vla = build_vla_from_cfg(cfg.inference_model)
    else:
        vla = build_vla_from_cfg(cfg.model)

    # --- Load checkpoint ---
    ckpt_path = args.ckpt_path
    assert Path(ckpt_path).exists(), f"Checkpoint not found: {ckpt_path}"
    print(f"[serve_vla] Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    vla.load_state_dict(state_dict, strict=True)

    # --- Load norm_stats if available ---
    data_stat_path = os.path.join(
        Path(ckpt_path).resolve().parent.parent, "dataset_statistics.json")
    if os.path.isfile(data_stat_path):
        with open(data_stat_path, "r") as f:
            vla.norm_stats = json.load(f)
        print(f"[serve_vla] Loaded norm_stats from {data_stat_path}")

    # --- Create pipeline ---
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16,
                 "fp32": torch.float32}
    from grpc_server import VLAInferPipeline, serve

    pipeline = VLAInferPipeline(
        vla_model=vla,
        device=args.device,
        mixed_precision_dtype=dtype_map[args.dtype],
    )
    print(f"[serve_vla] Model on {args.device} ({args.dtype}), "
          f"ready to serve.")

    # --- Start gRPC server ---
    server = serve(
        host=args.host,
        port=args.port,
        max_workers=args.workers,
        vla_pipeline=pipeline,
    )
    print(f"[serve_vla] gRPC server listening on {args.host}:{args.port}")
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(grace=2)
        print("[serve_vla] Server stopped.")


if __name__ == "__main__":
    main()
