"""Launch a ZMQ server that serves a real VLA model for remote inference.

Usage::

    python zmq_msgpack/serve_vla_zmq.py \
        --config configs/pi05/pi05_paligemma_libero_10_full_finetune.py \
        --ckpt-path /path/to/checkpoint.pt \
        --host 0.0.0.0 --port 5555
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from safetensors.torch import load_file

import torch

# Ensure project root is importable
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Serve a VLA model via ZMQ for remote inference")
    parser.add_argument("--config", required=True,
                        help="Path to mmengine config file")
    parser.add_argument("--ckpt-path", required=True,
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--device", default="cuda:0",
                        help="Device to run model on (default: cuda:0)")
    parser.add_argument("--dtype", default="bf16",
                        choices=["bf16", "fp16", "fp32"],
                        help="Mixed precision dtype (default: bf16)")
    parser.add_argument("--dataset-key", default=None,
                        choices=["inference", "eval"],
                        help="Config key to load dataset pipeline from "
                             "(default: auto-detect from config)")
    return parser.parse_args()


def main():
    args = parse_args()

    # --- Load config and build model ---
    from mmengine import Config
    from fluxvla.engines import build_vla_from_cfg

    cfg = Config.fromfile(args.config)

    print(f"[serve_vla_zmq] Building VLA model from config ...")
    if hasattr(cfg, "inference_model"):
        vla = build_vla_from_cfg(cfg.inference_model)
    else:
        vla = build_vla_from_cfg(cfg.model)

    # --- Load checkpoint ---
    ckpt_path = args.ckpt_path
    assert Path(ckpt_path).exists(), f"Checkpoint not found: {ckpt_path}"
    print(f"[serve_vla_zmq] Loading checkpoint: {ckpt_path}")
    if ckpt_path.endswith(".safetensors"):
        checkpoint = load_file(ckpt_path, device="cpu")
    else:
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
        print(f"[serve_vla_zmq] Loaded norm_stats from {data_stat_path}")

    # --- Build dataset preprocessing pipeline (server-side) ---
    from fluxvla.engines import build_dataset_from_cfg

    dataset = None
    dataset_key = args.dataset_key
    if dataset_key is None:
        # Auto-detect: prefer inference, fallback to eval
        if hasattr(cfg, "inference") and "dataset" in cfg.inference:
            dataset_key = "inference"
        elif hasattr(cfg, "eval") and "dataset" in cfg.eval:
            dataset_key = "eval"

    if dataset_key:
        dataset_cfg = dict(getattr(cfg, dataset_key).dataset)
        # Inject norm_stats path so transforms can normalize states
        if "norm_stats" not in dataset_cfg:
            dataset_cfg["norm_stats"] = data_stat_path
        if "model_path" not in dataset_cfg:
            dataset_cfg["model_path"] = os.path.dirname(
                os.path.dirname(ckpt_path))
        dataset = build_dataset_from_cfg(dataset_cfg)
        print(f"[serve_vla_zmq] Dataset pipeline built from "
              f"cfg.{dataset_key}.dataset")
    else:
        print("[serve_vla_zmq] WARNING: No dataset pipeline found in config. "
              "Server expects pre-processed tensor batches from client.")

    # --- Create pipeline and server ---
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16,
                 "fp32": torch.float32}
    from fluxvla.remote.vla_server import VLAInferPipeline, create_vla_server

    pipeline = VLAInferPipeline(
        vla_model=vla,
        device=args.device,
        mixed_precision_dtype=dtype_map[args.dtype],
    )
    print(f"[serve_vla_zmq] Model on {args.device} ({args.dtype}), "
          f"ready to serve.")

    server = create_vla_server(
        pipeline=pipeline,
        host=args.host,
        port=args.port,
        dataset=dataset,
    )
    print(f"[serve_vla_zmq] ZMQ server starting on "
          f"tcp://{args.host}:{args.port}")
    try:
        server.run()  # Blocking
    except KeyboardInterrupt:
        server.close()
        print("[serve_vla_zmq] Server stopped.")


if __name__ == "__main__":
    main()
