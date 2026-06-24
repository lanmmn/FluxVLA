#!/usr/bin/env python3
from __future__ import annotations
import argparse, statistics, sys, time
from pathlib import Path
import torch
from mmengine import Config

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, required=True)
    p.add_argument('--ckpt', type=str, default=None)
    p.add_argument('--no-weights', action='store_true')
    p.add_argument('--predict-runs', type=int, default=10)
    p.add_argument('--warmup', type=int, default=2)
    args = p.parse_args()

    from fluxvla.engines import build_vla_from_cfg, set_seed_everywhere
    set_seed_everywhere(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cfg = Config.fromfile(str(Path(args.config).expanduser().resolve()))
    model_cfg = cfg.model.to_dict()
    if args.no_weights or not args.ckpt:
        model_cfg['pretrained_name_or_path'] = None
        model_cfg['name_mapping'] = None

    print('构建 GR00t Eagle 3B 模型...')
    vla = build_vla_from_cfg(model_cfg)
    vla = vla.to(device)
    if device.type == 'cuda':
        vla.to(torch.bfloat16)

    if args.ckpt and not args.no_weights:
        print(f'加载权重: {args.ckpt}')
        state = torch.load(args.ckpt, map_location='cpu')
        vla.load_state_dict(state, strict=False)

    vla.eval()
    device_name = torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'
    print(f'设备: {device_name}')

    action_dim = model_cfg['vla_head']['ori_action_dim']
    n_steps = model_cfg['vla_head']['action_dim']
    print(f'模型参数: ori_action_dim={action_dim}, action_dim={n_steps}')

    images = torch.randn(1, 6, 224, 224, device=device, dtype=torch.float32)
    img_masks = torch.ones(1, 2, dtype=torch.bool, device=device)
    lang_tokens = torch.randint(0, 32000, (1, 48), device=device, dtype=torch.long)
    lang_masks = torch.ones(1, 48, dtype=torch.bool, device=device)
    states = torch.randn(1, n_steps, device=device, dtype=torch.float32)
    noise = torch.randn(1, n_steps, n_steps, device=device, dtype=torch.float32)
    batch = {'images': images, 'img_masks': img_masks, 'lang_tokens': lang_tokens,
             'lang_masks': lang_masks, 'states': states, 'noise': noise}

    print(f'开始推理测试: warmup={args.warmup}, runs={args.predict_runs}')
    times = []
    with torch.inference_mode():
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            for _ in range(args.warmup):
                _ = vla.predict_action(**batch)
                if device.type == 'cuda': torch.cuda.synchronize()
            for _ in range(args.predict_runs):
                if device.type == 'cuda': torch.cuda.synchronize()
                t0 = time.perf_counter()
                actions = vla.predict_action(**batch)
                if device.type == 'cuda': torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)

    ms = [t * 1000.0 for t in times]
    print(f'输出形状: {tuple(actions.shape)}')
    print(f'延迟 (ms): min={min(ms):.3f} max={max(ms):.3f} mean={statistics.mean(ms):.3f} median={statistics.median(ms):.3f} stdev={statistics.stdev(ms) if len(ms) > 1 else 0:.3f}')
    print(f'总耗时: {sum(times)*1000:.3f} ms ({args.predict_runs} 次)')
    print('✅ GR00t Eagle 3B 推理测试完成！')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
