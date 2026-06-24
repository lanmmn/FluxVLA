#!/usr/bin/env python3
"""GR00t Eagle 3B 实际模型推理测试 - 最终版本

参考 PI0.5 测试脚本结构，修复了所有输入格式问题。

使用方法::

    cd /home/limx/sober/FluxVLA
    python scripts/test_gr00t_final.py \\
        --config configs/gr00t/gr00t_eagle_3b_ur3_full_finetune.py \\
        --no-weights --predict-runs 10

加载权重测试::

    python scripts/test_gr00t_final.py \\
        --config configs/gr00t/gr00t_eagle_3b_ur3_full_finetune.py \\
        --ckpt /mnt/nvme/gr00t_eagle_3b_ur3_finetune_test_10241025/checkpoints/step-004378-epoch-02-loss=0.3078.pt \\
        --predict-runs 10
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

import torch
from mmengine import Config

# Flash Attention 未被强制禁用；运行时可通过环境变量 `ATTN_IMPLEMENTATION` / `TRANSFORMERS_ATTN_IMPLEMENTATION`
# 来选择具体实现（例如: flash_attention_2）。


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='GR00t Eagle 3B dummy-tensor forward / inference test.')
    p.add_argument(
        '--config',
        type=str,
        required=True,
        help='配置文件路径（.py 或 .yaml）。',
    )
    p.add_argument(
        '--ckpt',
        type=str,
        default=None,
        help='模型权重文件路径。',
    )
    p.add_argument(
        '--no-weights',
        action='store_true',
        help='不加载权重（仅随机初始化，用于快速验证）。',
    )
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--num-views', type=int, default=2, help='UR3 相机数量')
    p.add_argument('--lang-len', type=int, default=48)
    p.add_argument('--image-size', type=int, default=224)
    p.add_argument(
        '--embodiment-id',
        type=int,
        default=None,
        help='机器人 embodiment ID；默认读取配置里的 dataset.embodiment_id。',
    )
    p.add_argument('--seed', type=int, default=0)
    p.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
    )
    p.add_argument(
        '--predict-runs',
        type=int,
        default=50,
        help='推理重复次数（用于统计延迟）。',
    )
    p.add_argument(
        '--warmup',
        type=int,
        default=2,
        help='预热次数（不计入统计）。',
    )
    return p.parse_args()


def _make_dummy_batch(
    *,
    device: torch.device,
    batch_size: int,
    num_views: int,
    image_size: int,
    lang_len: int,
    state_dim: int,
    action_dim: int,
    vocab_size: int,
    embodiment_id: int,
) -> dict[str, torch.Tensor]:
    """构造与 GR00t predict_action 兼容的 dummy 张量。

    参考 PI0.5 的输入格式，确保所有维度正确。
    """
    # 图像: (B, V*3, H, W) - 与 SigLIPViTBackbone 格式一致
    images = torch.randn(
        batch_size,
        num_views * 3,
        image_size,
        image_size,
        device=device,
        dtype=torch.float32,
    )

    # 图像 mask: (B, num_views)
    img_masks = torch.ones(
        batch_size,
        num_views,
        dtype=torch.bool,
        device=device,
    )

    # 语言 tokens
    lang_tokens = torch.randint(
        low=0,
        high=min(32000, max(1, vocab_size - 1)),
        size=(batch_size, lang_len),
        device=device,
        dtype=torch.long,
    )
    lang_masks = torch.ones(batch_size, lang_len, dtype=torch.bool, device=device)

    # 状态: (B, state_dim) - 注意使用 state_dim 而不是 action_dim
    states = torch.randn(batch_size, state_dim, device=device, dtype=torch.float32)

    # 噪声: (B, action_dim, action_dim) - 用于 flow matching
    noise = torch.randn(
        batch_size,
        action_dim,
        action_dim,
        device=device,
        dtype=torch.float32,
    )

    embodiment_ids = torch.full(
        (batch_size,),
        fill_value=embodiment_id,
        device=device,
        dtype=torch.long,
    )

    return {
        'images': images,
        'img_masks': img_masks,
        'lang_tokens': lang_tokens,
        'lang_masks': lang_masks,
        'states': states,
        'noise': noise,
        'embodiment_ids': embodiment_ids,
    }


def _cfg_get_nested(cfg_obj, *keys, default=None):
    value = cfg_obj
    for key in keys:
        if not hasattr(value, 'get'):
            return default
        value = value.get(key, default)
        if value is default:
            return default
    return value


def _default_embodiment_id(cfg) -> int:
    for path in (
        ('inference', 'dataset', 'embodiment_id'),
        ('dataset', 'embodiment_id'),
    ):
        value = _cfg_get_nested(cfg, *path)
        if value is not None:
            return int(value)

    datasets = _cfg_get_nested(
        cfg, 'train_dataloader', 'dataset', 'datasets', default=[])
    for dataset in datasets:
        for transform in dataset.get('transforms', []):
            if 'embodiment_id' in transform:
                return int(transform['embodiment_id'])
    return 0


def main() -> int:
    args = _parse_args()

    print(
        'Attention backend:',
        os.environ.get('ATTN_IMPLEMENTATION')
        or os.environ.get('TRANSFORMERS_ATTN_IMPLEMENTATION')
        or 'flash_attention_2',
    )

    # 导入 FluxVLA
    from fluxvla.engines import build_vla_from_cfg, set_seed_everywhere

    set_seed_everywhere(args.seed)
    device = torch.device(args.device)

    cfg_path = Path(args.config).expanduser().resolve()
    if not cfg_path.is_file():
        print(f'ERROR: config not found: {cfg_path}', file=sys.stderr)
        return 1

    cfg = Config.fromfile(str(cfg_path))
    if 'model' not in cfg:
        print('ERROR: config missing `model` field', file=sys.stderr)
        return 1

    model_cfg = cfg.model.to_dict()

    # 如果不加载权重，清空预训练路径
    if args.no_weights or not args.ckpt:
        model_cfg['pretrained_name_or_path'] = None
        model_cfg['name_mapping'] = None

    print('Building GR00t Eagle 3B model...')
    vla = build_vla_from_cfg(model_cfg)
    vla = vla.to(device)
    if device.type == 'cuda' and hasattr(vla, 'to_bfloat16'):
        vla.to_bfloat16()

    # 加载权重
    if args.ckpt and not args.no_weights:
        ckpt_path = Path(args.ckpt).expanduser().resolve()
        if not ckpt_path.is_file():
            print(f'ERROR: checkpoint not found: {ckpt_path}', file=sys.stderr)
            return 1
        print(f'Loading weights: {ckpt_path}')
        state = torch.load(str(ckpt_path), map_location='cpu')
        missing, unexpected = vla.load_state_dict(state, strict=False)
        print(f'load_state_dict: missing={len(missing)}, unexpected={len(unexpected)}')

    vla.eval()

    if device.type == 'cuda':
        print(f'CUDA: {torch.cuda.get_device_name(device)}')
    else:
        print('Device: CPU')

    # 从配置获取正确的维度
    state_dim = model_cfg['vla_head']['state_dim']  # 64
    action_dim = model_cfg['vla_head']['action_dim']  # 32
    ori_action_dim = model_cfg['vla_head']['ori_action_dim']  # 7
    embodiment_id = args.embodiment_id
    if embodiment_id is None:
        embodiment_id = _default_embodiment_id(cfg)
    vocab_size = 257152  # Eagle 默认词汇表大小

    print(f'Model params: state_dim={state_dim}, action_dim={action_dim}, ori_action_dim={ori_action_dim}, embodiment_id={embodiment_id}')

    # 构造 dummy batch
    batch = _make_dummy_batch(
        device=device,
        batch_size=args.batch_size,
        num_views=args.num_views,
        image_size=args.image_size,
        lang_len=args.lang_len,
        state_dim=state_dim,
        action_dim=action_dim,
        vocab_size=vocab_size,
        embodiment_id=embodiment_id,
    )

    print(f'Benchmark predict_action: warmup={args.warmup}, runs={args.predict_runs} (dummy tensors)...')

    times = []
    actions = None

    with torch.inference_mode():
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == 'cuda'
        ):
            # 预热
            for _ in range(args.warmup):
                actions = vla.predict_action(**batch)
                if device.type == 'cuda':
                    torch.cuda.synchronize()

            # 正式测试
            for _ in range(args.predict_runs):
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                actions = vla.predict_action(**batch)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)

    ms = [t * 1000.0 for t in times]
    print(f'predict_action output shape: {tuple(actions.shape)}')
    print(
        f'latency_ms: min={min(ms):.3f} max={max(ms):.3f} '
        f'mean={statistics.mean(ms):.3f} median={statistics.median(ms):.3f}',
        end='',
    )
    if len(ms) > 1:
        print(f' stdev={statistics.stdev(ms):.3f}')
    else:
        print()

    total = sum(times)
    print(f'total_wall_predict={total*1000:.3f} ms ({args.predict_runs} runs, excl. warmup)')

    print('OK')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
