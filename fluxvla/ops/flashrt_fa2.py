import math

import torch


_FLASHRT_FA2 = None


def _load_flashrt_fa2():
    global _FLASHRT_FA2
    if _FLASHRT_FA2 is None:
        try:
            from flash_rt import flash_rt_fa2
        except ImportError as exc:
            raise RuntimeError(
                'FLUXVLA_PI05_FA2=1 requires FlashRT built with '
                'flash_rt_fa2. Build FlashRT with GPU_ARCH=87, '
                'FA2_HDIMS including 256, and FA2_DTYPES including bf16.'
            ) from exc
        _FLASHRT_FA2 = flash_rt_fa2
    return _FLASHRT_FA2


def check_flashrt_fa2_available():
    _load_flashrt_fa2()


def _call_fa2_bf16(q, k, v, out, lse, lse_accum, out_accum):
    fa2 = _load_flashrt_fa2()
    batch, seqlen_q, num_heads_q, head_dim = q.shape
    seqlen_k = k.shape[1]
    num_heads_kv = k.shape[2]
    stream = torch.cuda.current_stream(q.device).cuda_stream
    fa2.fwd_bf16(
        q.data_ptr(),
        k.data_ptr(),
        v.data_ptr(),
        out.data_ptr(),
        lse.data_ptr(),
        lse_accum.data_ptr() if lse_accum is not None else 0,
        out_accum.data_ptr() if out_accum is not None else 0,
        batch=batch,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        head_dim=head_dim,
        q_strides=tuple(q.stride()[:3]),
        k_strides=tuple(k.stride()[:3]),
        v_strides=tuple(v.stride()[:3]),
        o_strides=tuple(out.stride()[:3]),
        softmax_scale=1.0 / math.sqrt(head_dim),
        num_sms=torch.cuda.get_device_properties(q.device).multi_processor_count,
        stream=stream,
    )


def pi05_encoder_attention(q_flat,
                           k_flat,
                           v_flat,
                           out,
                           lse,
                           lse_accum,
                           out_accum,
                           seq_len,
                           num_heads=8,
                           head_dim=256):
    q = q_flat[:seq_len * num_heads].view(1, seq_len, num_heads, head_dim)
    k = k_flat[:seq_len].view(1, seq_len, 1, head_dim)
    v = v_flat[:seq_len].view(1, seq_len, 1, head_dim)
    o = out[:, :seq_len]
    _call_fa2_bf16(q, k, v, o, lse, lse_accum, out_accum)
    return o.view(seq_len, num_heads * head_dim)


def pi05_decoder_attention(q_flat,
                           k_flat,
                           v_flat,
                           out,
                           lse,
                           lse_accum,
                           out_accum,
                           query_len,
                           key_len,
                           num_heads=8,
                           head_dim=256):
    q = q_flat[:query_len * num_heads].view(1, query_len, num_heads, head_dim)
    k = k_flat[:key_len].view(1, key_len, 1, head_dim)
    v = v_flat[:key_len].view(1, key_len, 1, head_dim)
    o = out[:, :query_len]
    _call_fa2_bf16(q, k, v, o, lse, lse_accum, out_accum)
    return o.view(query_len, num_heads * head_dim)