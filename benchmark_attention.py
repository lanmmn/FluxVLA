"""Benchmark: FlexAttention vs Eager vs SDPA for Pi0.5 Libero10 training.

Simulates the exact attention pattern used in pi05_flowmatching training:
  - Sequence length: 703 (512 img + 180 lang + 1 state + 10 action)
  - GQA: 8 query heads, 1 KV head, head_dim=256
  - Batch size: 8 (per-device in libero10 config)
  - bf16 mixed precision
  - Pi0.5 segment mask pattern

Note: torch.compile(flex_attention) hits triton shared memory limit on A100
with head_dim=256, so we benchmark the non-compiled version. The compiled
version would be faster if the environment supports it.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

# ── Constants matching pi05 libero10 config ──────────────────────────────
SEQ_LEN = 703         # 512 img + 180 lang + 1 state + 10 action
NUM_Q_HEADS = 8
NUM_KV_HEADS = 1
HEAD_DIM = 256
BATCH_SIZE = 8
NUM_LAYERS = 18
DTYPE = torch.bfloat16
DEVICE = "cuda"

IMG_LEN = 512
LANG_LEN = 180
STATE_LEN = 1
ACTION_LEN = 10

WARMUP_ITERS = 10
BENCH_ITERS = 50


# ── Helpers ──────────────────────────────────────────────────────────────
def build_pi05_masks(batch_size, seq_len, device):
    att_masks = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    att_masks[:, 0] = True
    att_masks[:, IMG_LEN] = True
    att_masks[:, IMG_LEN + LANG_LEN] = True
    att_masks[:, IMG_LEN + LANG_LEN + STATE_LEN] = True
    pad_masks = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
    return att_masks, pad_masks


def make_eager_4d_mask(pad_masks, att_masks):
    cumsum = torch.cumsum(att_masks.int(), dim=1)
    att_2d = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d = pad_masks[:, None, :] & pad_masks[:, :, None]
    att_2d = att_2d & pad_2d
    att_4d = att_2d[:, None, :, :]
    return torch.where(att_4d, 0.0, -2.3819763e38).to(DTYPE)


def make_sdpa_bool_mask(pad_masks, att_masks):
    """Build boolean mask for SDPA (True = attend, False = masked)."""
    cumsum = torch.cumsum(att_masks.int(), dim=1)
    att_2d = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d = pad_masks[:, None, :] & pad_masks[:, :, None]
    att_2d = att_2d & pad_2d
    # SDPA expects (B, 1, L, L) or (B, H, L, L)
    return att_2d[:, None, :, :].expand(-1, NUM_Q_HEADS, -1, -1)


def make_flex_block_mask(att_masks, pad_masks, device):
    B, L = att_masks.shape
    segment_ids = torch.cumsum(att_masks.int(), dim=1)

    def mask_mod(b, h, q_idx, kv_idx):
        q_seg = segment_ids[b, q_idx]
        kv_seg = segment_ids[b, kv_idx]
        segment_ok = q_seg >= kv_seg
        pad_ok = pad_masks[b, q_idx] & pad_masks[b, kv_idx]
        return segment_ok & pad_ok

    return create_block_mask(mask_mod, B=B, H=None, Q_LEN=L, KV_LEN=L, device=device)


def repeat_kv(hidden_states, n_rep):
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


# ── Attention forward functions ──────────────────────────────────────────
def eager_attn(q, k, v, mask, scaling, n_rep):
    """Eager attention — exact copy of the project's eager_attention_forward."""
    k_exp = repeat_kv(k, n_rep)
    v_exp = repeat_kv(v, n_rep)
    attn_weights = torch.matmul(q, k_exp.transpose(2, 3)) * scaling
    attn_weights = attn_weights + mask[:, :, :, :k_exp.shape[-2]]
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
    return torch.matmul(attn_weights, v_exp).transpose(1, 2).contiguous()


def flex_attn(q, k, v, block_mask, scaling):
    """FlexAttention — non-compiled (torch.compile hits shared mem limit on A100 with D=256)."""
    out = flex_attention(q, k, v, block_mask=block_mask, scale=scaling, enable_gqa=True)
    return out.transpose(1, 2).contiguous()


def sdpa_attn(q, k, v, bool_mask, scaling):
    """SDPA with flash attention backend where possible."""
    k_exp = repeat_kv(k, NUM_Q_HEADS // NUM_KV_HEADS)
    v_exp = repeat_kv(v, NUM_Q_HEADS // NUM_KV_HEADS)
    out = F.scaled_dot_product_attention(q, k_exp, v_exp, attn_mask=bool_mask, scale=scaling)
    return out.transpose(1, 2).contiguous()


def sdpa_attn_no_mask(q, k, v, _, scaling):
    """SDPA without mask (causal=False) — upper bound for SDPA speed."""
    k_exp = repeat_kv(k, NUM_Q_HEADS // NUM_KV_HEADS)
    v_exp = repeat_kv(v, NUM_Q_HEADS // NUM_KV_HEADS)
    out = F.scaled_dot_product_attention(q, k_exp, v_exp, scale=scaling)
    return out.transpose(1, 2).contiguous()


# ── Benchmark runner ─────────────────────────────────────────────────────
def benchmark_fn(name, fn, mask, scaling, extra_kwargs=None, iters=BENCH_ITERS):
    """Benchmark forward + backward for a single attention layer."""
    extra_kwargs = extra_kwargs or {}
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    q = torch.randn(BATCH_SIZE, NUM_Q_HEADS, SEQ_LEN, HEAD_DIM,
                     dtype=DTYPE, device=DEVICE, requires_grad=True)
    k = torch.randn(BATCH_SIZE, NUM_KV_HEADS, SEQ_LEN, HEAD_DIM,
                     dtype=DTYPE, device=DEVICE, requires_grad=True)
    v = torch.randn(BATCH_SIZE, NUM_KV_HEADS, SEQ_LEN, HEAD_DIM,
                     dtype=DTYPE, device=DEVICE, requires_grad=True)

    def run():
        out = fn(q, k, v, mask, scaling, **extra_kwargs)
        out.sum().backward()

    # Warmup
    print(f"  [{name}] warming up...", flush=True)
    for _ in range(WARMUP_ITERS):
        run()
    torch.cuda.synchronize()

    # Timed iterations
    torch.cuda.reset_peak_memory_stats()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        run()
        ends[i].record()
    torch.cuda.synchronize()

    times = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
    return times, peak_mem


def benchmark_mask_build(name, fn, att_masks, pad_masks, iters=BENCH_ITERS):
    for _ in range(5):
        fn(att_masks, pad_masks)
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn(att_masks, pad_masks)
        ends[i].record()
    torch.cuda.synchronize()
    return [s.elapsed_time(e) for s, e in zip(starts, ends)]


def benchmark_full_pass(name, fn, mask, scaling, extra_kwargs=None, iters=BENCH_ITERS):
    """Simulate 18-layer forward+backward (attention-only)."""
    extra_kwargs = extra_kwargs or {}
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    def make_qkv():
        return (
            torch.randn(BATCH_SIZE, NUM_Q_HEADS, SEQ_LEN, HEAD_DIM,
                         dtype=DTYPE, device=DEVICE, requires_grad=True),
            torch.randn(BATCH_SIZE, NUM_KV_HEADS, SEQ_LEN, HEAD_DIM,
                         dtype=DTYPE, device=DEVICE, requires_grad=True),
            torch.randn(BATCH_SIZE, NUM_KV_HEADS, SEQ_LEN, HEAD_DIM,
                         dtype=DTYPE, device=DEVICE, requires_grad=True),
        )

    def run():
        total = torch.tensor(0.0, device=DEVICE)
        for _ in range(NUM_LAYERS):
            q, k, v = make_qkv()
            out = fn(q, k, v, mask, scaling, **extra_kwargs)
            total = total + out.sum()
        total.backward()

    print(f"  [{name}] warming up 18-layer pass...", flush=True)
    for _ in range(WARMUP_ITERS):
        run()
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        run()
        ends[i].record()
    torch.cuda.synchronize()

    times = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
    return times, peak_mem


def stats(times):
    s = sorted(times)
    n = len(s)
    return {
        'median': s[n // 2],
        'mean': sum(s) / n,
        'p10': s[int(n * 0.1)],
        'p90': s[int(n * 0.9)],
    }


def print_row(name, times, peak_mem=None):
    s = stats(times)
    mem = f"  peak={peak_mem:,.0f} MB" if peak_mem else ""
    print(f"  {name:30s}  median={s['median']:8.2f} ms  mean={s['mean']:8.2f} ms  "
          f"p10={s['p10']:8.2f}  p90={s['p90']:8.2f}{mem}")


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 90)
    print("Benchmark: Eager vs FlexAttention vs SDPA  |  Pi0.5 Libero10")
    print(f"  B={BATCH_SIZE}, L={SEQ_LEN}, Hq={NUM_Q_HEADS}, Hkv={NUM_KV_HEADS}, "
          f"D={HEAD_DIM}, layers={NUM_LAYERS}")
    print(f"  dtype={DTYPE}, GPU={torch.cuda.get_device_name(0)}")
    print(f"  warmup={WARMUP_ITERS}, bench={BENCH_ITERS}")
    print(f"  Note: flex_attention is NON-compiled (torch.compile hits triton shared mem OOR)")
    print("=" * 90)

    scaling = HEAD_DIM ** -0.5
    n_rep = NUM_Q_HEADS // NUM_KV_HEADS
    att_masks, pad_masks = build_pi05_masks(BATCH_SIZE, SEQ_LEN, DEVICE)

    # ── 1. Mask construction ─────────────────────────────────────────
    print("\n[1] Mask Construction Time")
    print("-" * 70)
    eager_mask_t = benchmark_mask_build(
        "Eager", lambda a, p: make_eager_4d_mask(p, a), att_masks, pad_masks)
    flex_mask_t = benchmark_mask_build(
        "Flex", lambda a, p: make_flex_block_mask(a, p, DEVICE), att_masks, pad_masks)
    print_row("Eager 4D mask", eager_mask_t)
    print_row("Flex BlockMask", flex_mask_t)

    # Build masks once
    eager_mask = make_eager_4d_mask(pad_masks, att_masks)
    flex_mask = make_flex_block_mask(att_masks, pad_masks, DEVICE)
    sdpa_mask = make_sdpa_bool_mask(pad_masks, att_masks)

    eager_mask_bytes = eager_mask.nelement() * eager_mask.element_size()
    print(f"\n  Eager 4D mask:  shape={list(eager_mask.shape)}, "
          f"size={eager_mask_bytes / 1024:.1f} KB")
    print(f"  Flex BlockMask: block-level metadata only (~KB)")
    print(f"  SDPA bool mask: shape={list(sdpa_mask.shape)}, "
          f"size={sdpa_mask.nelement() * sdpa_mask.element_size() / 1024:.1f} KB")

    # ── 2. Single-layer attention ────────────────────────────────────
    print("\n[2] Single Layer Attention (fwd + bwd)")
    print("-" * 70)

    results = {}

    t, m = benchmark_fn("Eager", eager_attn, eager_mask, scaling, extra_kwargs={'n_rep': n_rep})
    results['eager'] = (t, m)
    print_row("Eager", t, m)

    t, m = benchmark_fn("Flex (non-compiled)", flex_attn, flex_mask, scaling)
    results['flex'] = (t, m)
    print_row("Flex (non-compiled)", t, m)

    t, m = benchmark_fn("SDPA (with mask)", sdpa_attn, sdpa_mask, scaling)
    results['sdpa'] = (t, m)
    print_row("SDPA (with mask)", t, m)

    t, m = benchmark_fn("SDPA (no mask)", sdpa_attn_no_mask, None, scaling)
    results['sdpa_nomask'] = (t, m)
    print_row("SDPA (no mask, upper bound)", t, m)

    eager_med = stats(results['eager'][0])['median']
    flex_med = stats(results['flex'][0])['median']
    sdpa_med = stats(results['sdpa'][0])['median']
    sdpa_nm_med = stats(results['sdpa_nomask'][0])['median']

    print(f"\n  Speedup vs Eager:")
    print(f"    Flex (non-compiled): {eager_med / flex_med:.2f}x")
    print(f"    SDPA (with mask):    {eager_med / sdpa_med:.2f}x")
    print(f"    SDPA (no mask):      {eager_med / sdpa_nm_med:.2f}x")

    # ── 3. Full 18-layer pass ────────────────────────────────────────
    print("\n[3] Full 18-Layer Attention Pass (fwd + bwd)")
    print("-" * 70)

    full_results = {}

    t, m = benchmark_full_pass("Eager", eager_attn, eager_mask, scaling,
                                extra_kwargs={'n_rep': n_rep})
    full_results['eager'] = (t, m)
    print_row("Eager 18-layer", t, m)

    t, m = benchmark_full_pass("Flex", flex_attn, flex_mask, scaling)
    full_results['flex'] = (t, m)
    print_row("Flex 18-layer", t, m)

    t, m = benchmark_full_pass("SDPA", sdpa_attn, sdpa_mask, scaling)
    full_results['sdpa'] = (t, m)
    print_row("SDPA 18-layer", t, m)

    t, m = benchmark_full_pass("SDPA no-mask", sdpa_attn_no_mask, None, scaling)
    full_results['sdpa_nomask'] = (t, m)
    print_row("SDPA 18-layer (no mask)", t, m)

    fe_med = stats(full_results['eager'][0])['median']
    ff_med = stats(full_results['flex'][0])['median']
    fs_med = stats(full_results['sdpa'][0])['median']
    fsn_med = stats(full_results['sdpa_nomask'][0])['median']

    print(f"\n  18-layer speedup vs Eager:")
    print(f"    Flex (non-compiled): {fe_med / ff_med:.2f}x")
    print(f"    SDPA (with mask):    {fe_med / fs_med:.2f}x")
    print(f"    SDPA (no mask):      {fe_med / fsn_med:.2f}x")

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"  {'Method':30s} {'1-layer (ms)':>14s} {'18-layer (ms)':>14s} "
          f"{'1L mem (MB)':>12s} {'18L mem (MB)':>12s}")
    print(f"  {'-'*30} {'-'*14} {'-'*14} {'-'*12} {'-'*12}")
    for key, label in [('eager', 'Eager'),
                       ('flex', 'Flex (non-compiled)'),
                       ('sdpa', 'SDPA (with mask)'),
                       ('sdpa_nomask', 'SDPA (no mask)')]:
        s1 = stats(results[key][0])['median']
        m1 = results[key][1]
        s18 = stats(full_results[key][0])['median']
        m18 = full_results[key][1]
        print(f"  {label:30s} {s1:14.2f} {s18:14.2f} {m1:12,.0f} {m18:12,.0f}")

    print(f"\n  Note: flex_attention is benchmarked WITHOUT torch.compile.")
    print(f"  With torch.compile, flex would likely be faster (uses fused triton kernels).")
    print(f"  Current limitation: head_dim=256 exceeds A100 shared memory for compiled flex.")
    print("=" * 90)


if __name__ == "__main__":
    main()
