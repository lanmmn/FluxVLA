"""Pre-flight smoke for ResidualFlowMatchingInferenceHead.

Validates the two deployment risks WITHOUT a robot / ROS / VLM:
  1. strict load of the vla_head.* subset of the v3 checkpoint;
  2. CUDA-graph capture of the fused loop *including* the residual MLP,
     triggered by one predict_action call on fake VLM features.

Run inside the orin docker. Prints SMOKE_OK on success.
"""
import sys
import torch
from mmengine.config import Config
from safetensors.torch import load_file
from fluxvla.engines import HEADS

CKPT = sys.argv[1] if len(sys.argv) > 1 else (
    "/data/ckpts/tiga_basket_flow_4to2_residual_stage1/checkpoints/"
    "step-000400-epoch-00-loss=0.0039.safetensors")
CFG = ("configs/gr00t/"
       "gr00t_hud04_rtc_no_done_rtc_kernel_inference_residual.py")

cfg = Config.fromfile(CFG)
head_cfg = dict(cfg.inference_model["vla_head"])
htype = head_cfg.pop("type")
print("building head:", htype)
head = HEADS.build(dict(type=htype, **head_cfg)).to(device="cuda", dtype=torch.bfloat16).eval()

# Load only the vla_head.* params (strip prefix), strict within the head.
full = load_file(CKPT, device="cpu")
head_sd = {k[len("vla_head."):]: v for k, v in full.items()
           if k.startswith("vla_head.")}
missing, unexpected = head.load_state_dict(head_sd, strict=False)
missing = [m for m in missing]
unexpected = [u for u in unexpected]
print("load missing:", missing)
print("load unexpected:", unexpected)
assert not missing and not unexpected, "KEY MISMATCH -> strict load would fail"
print("STRICT_LOAD_OK")

# Fake VLM features to drive one full denoise + graph capture.
B = 1
seq = head.max_input_seq_len
bed = head.backbone_embedding_dim
dev = "cuda"
input_features = torch.randn(B, seq, bed, dtype=torch.bfloat16, device=dev)
states = torch.randn(B, head.state_dim, dtype=torch.bfloat16, device=dev)
attention_mask = torch.ones(B, seq, dtype=torch.bool, device=dev)
embodiment_ids = torch.zeros(B, dtype=torch.long, device=dev)

with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    # first call: triggers _load_weights_and_buffer + record_graph (capture)
    out1 = head.predict_action(
        input_features=input_features, states=states,
        attention_mask=attention_mask, embodiment_ids=embodiment_ids,
        prev_actions=None, prefix_len=0, rtc_config=None)
    # second call: pure graph replay
    out2 = head.predict_action(
        input_features=input_features, states=states,
        attention_mask=attention_mask, embodiment_ids=embodiment_ids,
        prev_actions=None, prefix_len=0, rtc_config=None)

print("out shape:", tuple(out1.shape), "dtype:", out1.dtype)
print("finite:", bool(torch.isfinite(out1).all()))
print("replay deterministic (same seed buffers):",
      bool(torch.allclose(out1, out2)))
assert torch.isfinite(out1).all(), "non-finite output"
print("GRAPH_CAPTURE_OK")
print("SMOKE_OK")
