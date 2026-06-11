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
"""Triton inference kernels for the DreamZero (Wan2.1) DiT block.

These fuse the memory-bound element-wise / normalization ops that
surround the GEMM + attention in each ``DiTBlock`` forward
(see ``models/third_party_models/dreamzero/modules/wan_video_dit.py``):

- ``adaln_modulate``   : LayerNorm(no affine) * (1 + scale) + shift   (norm1/norm2)
- ``rmsnorm``          : (x / rms(x)) * weight                         (norm_q/norm_k)
- ``gated_residual``   : x + gate * y                                  (GateModule)
- ``gelu_tanh``        : tanh-approx GELU                              (FFN activation)

All kernels are forward-only (inference), output bf16, and assume the
caller pre-allocates outputs (CUDA-Graph friendly).  ``scale``/``shift``/
``gate`` are per-batch ``(B, D)`` tensors broadcast over the sequence
dimension, matching the Wan DiT modulation layout.
"""

import torch
import triton
import triton.language as tl

try:  # Triton >= 3.x canonical extern-math import (see tutorials/07)
    from triton.language.extra import libdevice
except ImportError:  # pragma: no cover - older layouts
    from triton.language.math import libdevice


def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def _num_warps_for(block: int) -> int:
    if block <= 1024:
        return 4
    if block <= 4096:
        return 8
    return 16


@triton.jit
def _adaln_modulate_kernel(x_ptr, scale_ptr, shift_ptr, out_ptr,
                           n_rows, seq_len, D,
                           BLOCK: tl.constexpr, eps: tl.constexpr):
    """LayerNorm (no affine) then ``* (1 + scale) + shift``.

    x:     (n_rows, D)        n_rows = B * seq_len
    scale: (B, D)             broadcast over seq
    shift: (B, D)             broadcast over seq
    """
    row = tl.program_id(0)
    if row >= n_rows:
        return
    cols = tl.arange(0, BLOCK)
    mask = cols < D

    x = tl.load(x_ptr + row * D + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / D
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / D
    xn = xc * tl.rsqrt(var + eps)

    b = row // seq_len
    scale = tl.load(scale_ptr + b * D + cols, mask=mask, other=0.0).to(
        tl.float32)
    shift = tl.load(shift_ptr + b * D + cols, mask=mask, other=0.0).to(
        tl.float32)
    out = xn * (1.0 + scale) + shift
    tl.store(out_ptr + row * D + cols, out.to(tl.bfloat16), mask=mask)


@triton.jit
def _rmsnorm_kernel(x_ptr, w_ptr, out_ptr, n_rows, D,
                    BLOCK: tl.constexpr, eps: tl.constexpr):
    """RMSNorm: ``(x * rsqrt(mean(x^2) + eps)) * weight``.

    x: (n_rows, D),  weight: (D,)
    """
    row = tl.program_id(0)
    if row >= n_rows:
        return
    cols = tl.arange(0, BLOCK)
    mask = cols < D

    x = tl.load(x_ptr + row * D + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / D
    xn = x * tl.rsqrt(var + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + row * D + cols, (xn * w).to(tl.bfloat16), mask=mask)


@triton.jit
def _gated_residual_kernel(x_ptr, gate_ptr, y_ptr, out_ptr,
                           n_rows, seq_len, D, BLOCK: tl.constexpr):
    """``x + gate * y`` with gate broadcast over seq.

    x, y: (n_rows, D),  gate: (B, D)
    """
    row = tl.program_id(0)
    if row >= n_rows:
        return
    cols = tl.arange(0, BLOCK)
    mask = cols < D

    x = tl.load(x_ptr + row * D + cols, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + row * D + cols, mask=mask, other=0.0).to(tl.float32)
    b = row // seq_len
    g = tl.load(gate_ptr + b * D + cols, mask=mask, other=0.0).to(tl.float32)
    out = x + g * y
    tl.store(out_ptr + row * D + cols, out.to(tl.bfloat16), mask=mask)


@triton.jit
def _gelu_tanh_kernel(x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    """tanh-approx GELU, element-wise over a flat tensor."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    # 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 x^3)))
    c = 0.7978845608028654  # sqrt(2/pi)
    inner = c * (x + 0.044715 * x * x * x)
    out = 0.5 * x * (1.0 + libdevice.tanh(inner))
    tl.store(out_ptr + offs, out.to(tl.bfloat16), mask=mask)


# ----------------------------------------------------------------------
# Python wrappers
# ----------------------------------------------------------------------

def adaln_modulate(x: torch.Tensor, scale: torch.Tensor,
                   shift: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Fused AdaLN modulation for Wan DiT norm1/norm2.

    Args:
        x: (B, L, D) hidden states.
        scale: (B, 1, D) or (B, D) modulation scale.
        shift: (B, 1, D) or (B, D) modulation shift.
        eps: LayerNorm epsilon.

    Returns:
        (B, L, D) bf16 tensor = LayerNorm(x) * (1 + scale) + shift.
    """
    B, L, D = x.shape
    x2 = x.reshape(B * L, D).contiguous()
    scale2 = scale.reshape(B, D).contiguous()
    shift2 = shift.reshape(B, D).contiguous()
    out = torch.empty_like(x2, dtype=torch.bfloat16)
    block = _next_pow2(D)
    _adaln_modulate_kernel[(B * L, )](
        x2, scale2, shift2, out, B * L, L, D,
        BLOCK=block, eps=eps, num_warps=_num_warps_for(block))
    return out.view(B, L, D)


def rmsnorm(x: torch.Tensor, weight: torch.Tensor,
            eps: float = 1e-6) -> torch.Tensor:
    """Fused RMSNorm for Wan DiT norm_q/norm_k.

    Args:
        x: (B, L, D) hidden states.
        weight: (D,) scale.
        eps: epsilon.

    Returns:
        (B, L, D) bf16 tensor.
    """
    B, L, D = x.shape
    x2 = x.reshape(B * L, D).contiguous()
    out = torch.empty_like(x2, dtype=torch.bfloat16)
    block = _next_pow2(D)
    _rmsnorm_kernel[(B * L, )](
        x2, weight.contiguous(), out, B * L, D,
        BLOCK=block, eps=eps, num_warps=_num_warps_for(block))
    return out.view(B, L, D)


def gated_residual(x: torch.Tensor, gate: torch.Tensor,
                   y: torch.Tensor) -> torch.Tensor:
    """Fused gated residual ``x + gate * y`` for Wan DiT GateModule.

    Args:
        x: (B, L, D) residual stream.
        gate: (B, 1, D) or (B, D) gate.
        y: (B, L, D) branch output.

    Returns:
        (B, L, D) bf16 tensor.
    """
    B, L, D = x.shape
    x2 = x.reshape(B * L, D).contiguous()
    y2 = y.reshape(B * L, D).contiguous()
    gate2 = gate.reshape(B, D).contiguous()
    out = torch.empty_like(x2, dtype=torch.bfloat16)
    block = _next_pow2(D)
    _gated_residual_kernel[(B * L, )](
        x2, gate2, y2, out, B * L, L, D,
        BLOCK=block, num_warps=_num_warps_for(block))
    return out.view(B, L, D)


def gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    """tanh-approx GELU (Wan DiT FFN activation)."""
    x2 = x.contiguous()
    out = torch.empty_like(x2, dtype=torch.bfloat16)
    n = x2.numel()
    block = 1024
    grid = (triton.cdiv(n, block), )
    _gelu_tanh_kernel[grid](x2, out, n, BLOCK=block)
    return out.view_as(x)
