"""Fold a Latent Consistency Model's guidance embedding into its UNet weights.

An LCM UNet takes one input a Stable Diffusion UNet doesn't: `timestep_cond`, the embedding of
the guidance scale w that classifier-free guidance was distilled into. Apple's Core ML converter
and its Swift pipeline know nothing of that input. But w is a constant at inference, and the
embedding only ever enters the network as `cond_proj(emb)` added to the timestep embedding just
before `linear_1`, so `linear_1(t + cond_proj(emb)) == linear_1(t) + W1 @ cond_proj(emb)`: a
constant that can be added to `linear_1.bias` once. What comes out is a plain SD 1.5 UNet that
behaves exactly as the LCM would with that w.

    fold_lcm.py <diffusers source dir> <output dir> <w>
"""
import math
import sys

import torch
from diffusers import DiffusionPipeline

src, dst, w_value = sys.argv[1], sys.argv[2], float(sys.argv[3])
pipe = DiffusionPipeline.from_pretrained(src, torch_dtype=torch.float32, safety_checker=None)
unet = pipe.unet
dim = unet.config.time_cond_proj_dim
assert dim, "not an LCM UNet: no time_cond_proj_dim"

# diffusers' LatentConsistencyModelPipeline.get_guidance_scale_embedding, for one w.
w = torch.tensor([w_value]) * 1000.0
half = dim // 2
freqs = torch.exp(torch.arange(half, dtype=torch.float32) * -(math.log(10000.0) / (half - 1)))
emb = w[:, None] * freqs[None, :]
emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
assert emb.shape == (1, dim)

# A reference answer from the unfolded model, to check the fold against.
torch.manual_seed(0)
sample = torch.randn(1, unet.config.in_channels, 64, 64)
t = torch.tensor([759.0])
states = torch.randn(1, 77, unet.config.cross_attention_dim)
unet.eval()
with torch.no_grad():
    before = unet(sample, t, encoder_hidden_states=states, timestep_cond=emb).sample

te = unet.time_embedding
with torch.no_grad():
    cond = te.cond_proj(emb)[0]                      # [320]
    te.linear_1.bias += te.linear_1.weight @ cond    # W1 @ c
te.cond_proj = None
unet.register_to_config(time_cond_proj_dim=None)

with torch.no_grad():
    after = unet(sample, t, encoder_hidden_states=states).sample
diff = (before - after).abs().max().item()
print(f"fold check: max abs difference {diff:.3e}")
assert diff < 1e-3, "folding changed the model"

# Saved as the fp16 variant Apple's converter asks for by name.
pipe.to(torch.float16)
pipe.save_pretrained(dst, safe_serialization=True, variant="fp16")
print(f"saved folded pipeline to {dst}")
