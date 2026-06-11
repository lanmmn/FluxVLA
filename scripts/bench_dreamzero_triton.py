# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Correctness + speed benchmark for DreamZero Triton inference kernels.

Compares the fused Triton kernels in
``fluxvla/ops/triton/dreamzero_triton_ops.py`` against eager PyTorch
references at DreamZero (Wan2.1) DiT shapes.

Usage:
    python scripts/bench_dreamzero_triton.py
    python scripts/bench_dreamzero_triton.py --dim 1536 --seq 512 --batch 1
"""

import argparse
import importlib.util
import os

import torch
import torch.nn.functional as F

# Load the kernel module directly by path to avoid triggering the heavy
# ``fluxvla.ops`` package __init__ (which JIT-builds CUDA C++ extensions).
_KERNELS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'fluxvla', 'ops', 'triton', 'dreamzero_triton_ops.py')
_spec = importlib.util.spec_from_file_location('dreamzero_triton_ops',
                                               _KERNELS_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
adaln_modulate = _mod.adaln_modulate
gated_residual = _mod.gated_residual
gelu_tanh = _mod.gelu_tanh
rmsnorm = _mod.rmsnorm

WARMUP = 25
ITERS = 100


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    return F.cosine_similarity(a, b, dim=0).item()


def _maxdiff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def _time(fn) -> float:
    """Return median latency in ms over ITERS runs."""
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(ITERS):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]


# ---------------- eager references (match Wan DiT) ----------------

def eager_adaln_modulate(x, scale, shift, eps):
    n = F.layer_norm(x, (x.shape[-1], ), eps=eps)
    return (n * (1 + scale) + shift)


def eager_rmsnorm(x, weight, eps):
    n = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    return (n.type_as(x) * weight)


def eager_gated_residual(x, gate, y):
    return x + gate * y


def eager_gelu_tanh(x):
    return F.gelu(x, approximate='tanh')


def bench_op(name, triton_fn, eager_fn, gold_fn):
    gold = gold_fn()
    t_out = triton_fn()
    e_out = eager_fn()
    cos_t = _cosine(t_out, gold)
    cos_e = _cosine(e_out, gold)
    md_t = _maxdiff(t_out, gold)
    t_ms = _time(triton_fn)
    e_ms = _time(eager_fn)
    speedup = e_ms / t_ms
    print(f'{name:<22} | triton {t_ms:7.4f}ms | eager {e_ms:7.4f}ms | '
          f'{speedup:5.2f}x | cos(triton)={cos_t:.5f} '
          f'cos(eager)={cos_e:.5f} | maxdiff={md_t:.4f}')
    return t_ms, e_ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dim', type=int, default=5120, help='DiT hidden dim')
    ap.add_argument('--ffn', type=int, default=13824, help='FFN inner dim')
    ap.add_argument('--seq', type=int, default=512, help='token seq length')
    ap.add_argument('--batch', type=int, default=1)
    ap.add_argument('--eps', type=float, default=1e-6)
    args = ap.parse_args()

    assert torch.cuda.is_available(), 'CUDA required'
    dev = 'cuda'
    dt = torch.bfloat16
    B, L, D, F_ = args.batch, args.seq, args.dim, args.ffn
    torch.manual_seed(0)

    print(f'\nDreamZero Triton kernel benchmark  '
          f'(B={B} L={L} D={D} ffn={F_} dtype=bf16)')
    print(f'  device={torch.cuda.get_device_name(0)}  '
          f'warmup={WARMUP} iters={ITERS}\n')
    print('-' * 110)

    x = torch.randn(B, L, D, device=dev, dtype=dt)
    scale = torch.randn(B, 1, D, device=dev, dtype=dt) * 0.1
    shift = torch.randn(B, 1, D, device=dev, dtype=dt) * 0.1
    gate = torch.randn(B, 1, D, device=dev, dtype=dt) * 0.1
    y = torch.randn(B, L, D, device=dev, dtype=dt)
    w = torch.randn(D, device=dev, dtype=dt)
    x_ffn = torch.randn(B, L, F_, device=dev, dtype=dt)

    totals = [0.0, 0.0]

    t, e = bench_op(
        'adaln_modulate',
        lambda: adaln_modulate(x, scale, shift, args.eps),
        lambda: eager_adaln_modulate(x, scale, shift, args.eps),
        lambda: eager_adaln_modulate(x.float(), scale.float(),
                                     shift.float(), args.eps))
    totals[0] += t * 2  # norm1 + norm2 per block
    totals[1] += e * 2

    t, e = bench_op(
        'rmsnorm (q/k)',
        lambda: rmsnorm(x, w, args.eps),
        lambda: eager_rmsnorm(x, w, args.eps),
        lambda: eager_rmsnorm(x.float(), w.float(), args.eps))
    totals[0] += t * 2  # norm_q + norm_k per block
    totals[1] += e * 2

    t, e = bench_op(
        'gated_residual',
        lambda: gated_residual(x, gate, y),
        lambda: eager_gated_residual(x, gate, y),
        lambda: eager_gated_residual(x.float(), gate.float(), y.float()))
    totals[0] += t * 2  # gate_msa + gate_mlp per block
    totals[1] += e * 2

    t, e = bench_op(
        'gelu_tanh (ffn)',
        lambda: gelu_tanh(x_ffn),
        lambda: eager_gelu_tanh(x_ffn),
        lambda: eager_gelu_tanh(x_ffn.float()))
    totals[0] += t  # 1 GELU per block
    totals[1] += e

    print('-' * 110)
    print(f'{"per-block elementwise total":<22} | '
          f'triton {totals[0]:7.4f}ms | eager {totals[1]:7.4f}ms | '
          f'{totals[1] / totals[0]:5.2f}x')
    print(f'  (counts per DiTBlock: 2x adaln, 2x rmsnorm, 2x gate, 1x gelu)\n')


if __name__ == '__main__':
    main()
