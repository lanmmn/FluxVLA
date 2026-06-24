"""Per-phase latency profiler for the Pi0.5 Triton inference pipeline.

Times vision_encoder / transformer_encoder / transformer_decoder separately
(eager, outside the unified CUDA graph) with cuda events, plus a per-decoder
single-step estimate, so we know where the ~220 ms actually goes before
optimizing.  Reuses the e2e build path.

Run inside ``fluxvla:orin``::

    python scripts/profile_pi05_phases.py
"""
import statistics
import time

import torch

from fluxvla.engines import set_seed_everywhere
from fluxvla.models.vlas import pi05_flowmatching_inference as P
from scripts.test_pi05_encoder_int8_e2e import build_model
from scripts.test_pi05_dummy_forward import _make_dummy_batch
import yaml

CFG_PATH = '/mnt/nvme/sober/checkpoints/pi05_aloha_fold_towel/config.yaml'


def _evt():
    return torch.cuda.Event(enable_timing=True)


class AttentionTimer:
    def __init__(self):
        self.original_encoder = P.pi05_encoder_attention
        self.original_decoder = P.pi05_decoder_attention
        self.encoder_ms = 0.0
        self.decoder_ms = 0.0
        self.encoder_calls = 0
        self.decoder_calls = 0

    def reset(self):
        self.encoder_ms = 0.0
        self.decoder_ms = 0.0
        self.encoder_calls = 0
        self.decoder_calls = 0

    def _time_cuda(self, fn, *args, **kwargs):
        a, b = _evt(), _evt()
        a.record()
        out = fn(*args, **kwargs)
        b.record()
        b.synchronize()
        return out, a.elapsed_time(b)

    def wrap_encoder(self, *args, **kwargs):
        out, ms = self._time_cuda(self.original_encoder, *args, **kwargs)
        self.encoder_ms += ms
        self.encoder_calls += 1
        return out

    def wrap_decoder(self, *args, **kwargs):
        out, ms = self._time_cuda(self.original_decoder, *args, **kwargs)
        self.decoder_ms += ms
        self.decoder_calls += 1
        return out

    def __enter__(self):
        P.pi05_encoder_attention = self.wrap_encoder
        P.pi05_decoder_attention = self.wrap_decoder
        return self

    def __exit__(self, exc_type, exc, tb):
        P.pi05_encoder_attention = self.original_encoder
        P.pi05_decoder_attention = self.original_decoder


def time_fn(fn, warmup=5, runs=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(runs):
        a, b = _evt(), _evt()
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return statistics.median(ts)


def time_fn_with_attention(fn, timer, warmup=3, runs=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    phase_ms = []
    encoder_attn_ms = []
    decoder_attn_ms = []
    encoder_calls = []
    decoder_calls = []
    for _ in range(runs):
        timer.reset()
        a, b = _evt(), _evt()
        a.record()
        fn()
        b.record()
        b.synchronize()
        phase_ms.append(a.elapsed_time(b))
        encoder_attn_ms.append(timer.encoder_ms)
        decoder_attn_ms.append(timer.decoder_ms)
        encoder_calls.append(timer.encoder_calls)
        decoder_calls.append(timer.decoder_calls)
    return {
        'phase_ms': statistics.median(phase_ms),
        'encoder_attn_ms': statistics.median(encoder_attn_ms),
        'decoder_attn_ms': statistics.median(decoder_attn_ms),
        'encoder_calls': int(statistics.median(encoder_calls)),
        'decoder_calls': int(statistics.median(decoder_calls)),
    }


def main():
    set_seed_everywhere(0)
    device = torch.device('cuda')
    print('CUDA:', torch.cuda.get_device_name(0))

    raw = yaml.safe_load(open(CFG_PATH, 'r'))
    nv = len(raw['inference']['dataset']['img_keys'])
    vla = build_model(True, device)
    print(f'int8 cfg: o={vla._int8_o} mlp={vla._int8_mlp} down={vla._int8_down}')

    vocab = int(getattr(vla.llm_backbone.config, 'vocab_size', 257152))
    batch = _make_dummy_batch(
        device=device, batch_size=1, num_views=nv, image_size=224,
        lang_len=min(32, int(vla.triton_max_prompt_len)),
        max_action_dim=int(vla.max_action_dim),
        n_action_steps=int(vla.n_action_steps), vocab_size=vocab)

    # Warm up the full pipeline once: prepares triton weights, fills buffers,
    # autotunes kernels, builds the CUDA graph.
    with torch.inference_mode():
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            for _ in range(3):
                _ = vla.predict_action(
                    images=batch['images'], lang_tokens=batch['lang_tokens'],
                    states=batch['states'], img_masks=batch['img_masks'],
                    lang_masks=batch['lang_masks'], noise=batch['noise'])
    torch.cuda.synchronize()

    w = vla._triton_weights
    b = vla._triton_bufs
    esl = vla._encoder_seq_len

    def full():
        vla._cuda_graph.replay()

    def vision():
        P.vision_encoder(w, b, vla.num_views, vla._num_vit_layers)

    def encoder():
        P.transformer_encoder(w, b, esl, vla._num_encoder_layers,
                              vla._use_flashrt_fa2)

    def decoder():
        P.transformer_decoder(w, b, esl, vla._num_decoder_layers,
                              vla._num_steps, vla._use_flashrt_fa2)

    with torch.inference_mode():
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            t_full = time_fn(full)
            t_vis = time_fn(vision)
            if vla._use_flashrt_fa2:
                with AttentionTimer() as attn_timer:
                    enc_stats = time_fn_with_attention(encoder, attn_timer)
                    dec_stats = time_fn_with_attention(decoder, attn_timer)
                t_enc = enc_stats['phase_ms']
                t_dec = dec_stats['phase_ms']
            else:
                enc_stats = None
                dec_stats = None
                t_enc = time_fn(encoder)
                t_dec = time_fn(decoder)

    print('\n' + '=' * 56)
    print(f'full pipeline (graph replay): {t_full:8.2f} ms')
    print(f'vision_encoder (eager):       {t_vis:8.2f} ms')
    print(f'transformer_encoder (eager):  {t_enc:8.2f} ms')
    print(f'transformer_decoder (eager):  {t_dec:8.2f} ms')
    if enc_stats is not None and dec_stats is not None:
        enc_attn = enc_stats['encoder_attn_ms']
        dec_attn = dec_stats['decoder_attn_ms']
        print(f'  encoder FlashRT FA2:       {enc_attn:8.2f} ms  '
              f'({enc_attn / max(t_enc, 1e-6) * 100:5.1f}%, '
              f'{enc_stats["encoder_calls"]} calls)')
        print(f'  decoder FlashRT FA2:       {dec_attn:8.2f} ms  '
              f'({dec_attn / max(t_dec, 1e-6) * 100:5.1f}%, '
              f'{dec_stats["decoder_calls"]} calls)')
    print(f'  decoder per step (~/{vla._num_steps}):     '
          f'{t_dec / vla._num_steps:8.2f} ms')
    print(f'  sum eager phases:           {t_vis + t_enc + t_dec:8.2f} ms')
    print('=' * 56)
    print('note: eager sum > graph replay due to per-kernel launch overhead;')
    print('use the phase *ratios* to locate the bottleneck.')


if __name__ == '__main__':
    main()
