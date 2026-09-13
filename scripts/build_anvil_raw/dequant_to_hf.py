#!/usr/bin/env python3
"""Turn a Gemma 4 GGUF back into a Hugging Face checkpoint, tensor by tensor.

llama.cpp's converter renames the tensors and (for Gemma 4) changes nothing
else, so undoing it is a name map and a dequantize. Every tensor is handled on
its own and the big embeddings in row chunks, so the whole model is never in
memory at once: this runs on a 16 GB Mac.

    dequant_to_hf.py model.gguf out_dir

out_dir must already hold config.json (the full Gemma4Config, text_config
included) and the tokenizer files; this adds the safetensors shards and index.
"""

import json
import os
import re
import sys

import numpy as np
import torch
from gguf import GGUFReader, MODEL_ARCH, get_tensor_name_map
from gguf.quants import dequantize
from safetensors.torch import save_file
from transformers import Gemma4Config, Gemma4ForConditionalGeneration

SHARD_BYTES = 2 * 1024**3
CHUNK_ROWS = 8192

src, out = sys.argv[1], sys.argv[2]
config = Gemma4Config(**json.load(open(os.path.join(out, "config.json"))))
n_layers = config.text_config.num_hidden_layers

# The shapes and names the checkpoint must end up with, without allocating it.
with torch.device("meta"):
    meta = Gemma4ForConditionalGeneration(config)
hf_shapes = {k: tuple(v.shape) for k, v in meta.state_dict().items()}
del meta

# GGUF name -> HF name, from llama.cpp's own table, run backwards. The table
# knows the text model as "model.X"; in the multimodal checkpoint it lives at
# "model.language_model.X".
reverse = {}
for hf_name, (_, gguf_name) in get_tensor_name_map(MODEL_ARCH.GEMMA4, n_layers).mapping.items():
    reverse.setdefault(gguf_name, []).append(hf_name)


def hf_name_for(gguf_full):
    base, _, suffix = gguf_full.rpartition(".")
    if base == "rope_freqs":
        return None  # llama.cpp invents this one; HF computes it.
    candidates = []
    for hf in reverse.get(base, []):
        full = hf.replace("model.", "model.language_model.", 1) if hf.startswith("model.") else hf
        candidates += [full + "." + suffix, full]
    if base == "output":
        candidates.insert(0, "lm_head.weight")
    for c in candidates:
        if c in hf_shapes:
            return c
    # The KV-shared layers at the top of the stack reuse keys and values from
    # the layers below. HF has no K/V projections there; the GGUF carries them
    # anyway, unused, and they are dropped here.
    m = re.match(r"blk\.(\d+)\.(attn_k|attn_v|attn_k_norm)$", base)
    if m and int(m.group(1)) >= n_layers - config.text_config.num_kv_shared_layers:
        return None
    raise KeyError(f"no HF tensor for {gguf_full}; tried {candidates}")


def dequantize_rows(t, rows):
    """A slice of rows, dequantized to float32, whatever the storage type."""
    # gguf's dequantize knows every storage type, the float ones included; BF16 in
    # particular arrives as raw bytes, which a plain cast would read as two columns.
    return np.ascontiguousarray(dequantize(t.data[rows], t.tensor_type), dtype=np.float32)


reader = GGUFReader(src)
shard, shard_bytes, shard_index = {}, 0, 0
weight_map, total = {}, 0
seen = set()


def flush():
    global shard, shard_bytes, shard_index
    if not shard:
        return
    name = f"model-{shard_index:05d}.safetensors"
    save_file(shard, os.path.join(out, name), metadata={"format": "pt"})
    for k in shard:
        weight_map[k] = name
    print(f"  wrote {name} ({shard_bytes / 1e9:.2f} GB, {len(shard)} tensors)", flush=True)
    shard, shard_bytes, shard_index = {}, 0, shard_index + 1


for t in reader.tensors:
    hf = hf_name_for(t.name)
    if hf is None:
        continue
    want = hf_shapes[hf]
    n_rows = int(t.shape[-1]) if len(t.shape) > 1 else 1
    if len(t.shape) == 1:
        arr = dequantize_rows(t, slice(None)).reshape(-1)
        tensor = torch.from_numpy(np.ascontiguousarray(arr)).to(torch.bfloat16).reshape(want)
    else:
        rows_total, cols = int(t.shape[1]), int(t.shape[0])
        tensor = torch.empty((want[0], want[1]), dtype=torch.bfloat16)
        assert cols == want[1], f"{t.name}: {cols} columns, HF wants {want}"
        assert rows_total <= want[0], f"{t.name}: {rows_total} rows, HF wants {want}"
        for start in range(0, rows_total, CHUNK_ROWS):
            stop = min(start + CHUNK_ROWS, rows_total)
            chunk = dequantize_rows(t, slice(start, stop)).reshape(stop - start, cols)
            tensor[start:stop] = torch.from_numpy(np.ascontiguousarray(chunk)).to(torch.bfloat16)
        if rows_total < want[0]:
            # llama.cpp drops rows past the tokenizer's vocabulary; HF keeps them.
            tensor[rows_total:] = 0
            print(f"  {hf}: padded {rows_total} -> {want[0]} rows")
    nbytes = tensor.numel() * 2
    if shard_bytes and shard_bytes + nbytes > SHARD_BYTES:
        flush()
    shard[hf] = tensor
    shard_bytes += nbytes
    total += nbytes
    seen.add(hf)
    print(f"{t.name:40s} {t.tensor_type.name:6s} -> {hf} {tuple(tensor.shape)}", flush=True)
flush()

with open(os.path.join(out, "model.safetensors.index.json"), "w") as f:
    json.dump({"metadata": {"total_size": total}, "weight_map": weight_map}, f, indent=2)

missing = [k for k in hf_shapes if k not in seen and "vision" not in k and "audio" not in k and k != "lm_head.weight"]
print(f"\n{len(seen)} tensors written, {total / 1e9:.2f} GB")
print("text tensors missing from the GGUF:", missing or "none")
