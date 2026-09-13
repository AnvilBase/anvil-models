#!/usr/bin/env python3
"""Re-bundle the exported pieces into a .litertlm with a different chat template.

Hugging Face's Gemma 4 template calls `.get()` on messages, which LiteRT-LM's
template engine doesn't have; Google's own bundles carry the canonical template
from the LiteRT-LM repository, which doesn't. The tflite pieces are untouched.

    repack.py hf out out/<work dir> chat_template.jinja
"""

import os
import sys

import torch
import transformers

from litert_torch.generative.export_hf.core import export_lib
from litert_torch.generative.export_hf.core import exportable_module
from litert_torch.generative.export_hf.core import exportable_module_config
from litert_torch.generative.export_hf.core import litert_lm_builder
from litert_torch.generative.export_hf.model_ext import patches as model_ext_patches

HF, OUT, WORK, TEMPLATE = sys.argv[1:5]

export_config = exportable_module.ExportableModuleConfig(
    model=HF,
    output_dir=OUT,
    work_dir=WORK,
    task=exportable_module_config.ExportTask.TEXT_GENERATION,
    keep_temporary_files=True,
    prefill_lengths=[128, 1024],
    cache_length=32000,
    enable_gpu_dynamic_cache=True,
    externalize_embedder=True,
    single_token_embedder=True,
    bundle_litert_lm=True,
    quantization_recipe="dynamic_wi4_afp32",
    sampler_top_k=64,
    sampler_top_p=0.95,
    sampler_temperature=1.0,
    experimental_lightweight_conversion=True,
    jinja_chat_template_override=TEMPLATE,
)

config = transformers.AutoConfig.from_pretrained(HF, dtype=torch.float32)
for c in (config, getattr(config, "text_config", None)):
    if c is not None and hasattr(c, "allow_global_per_layer_attribute_access"):
        c.allow_global_per_layer_attribute_access = True
tokenizer = transformers.AutoTokenizer.from_pretrained(HF)
with model_ext_patches.get_patch_context(config.model_type):
    with torch.device("meta"):
        model = transformers.Gemma4ForConditionalGeneration(config)
model.generation_config = transformers.GenerationConfig.from_pretrained(HF)

source = export_lib.SourceModelArtifacts(
    model=model, model_config=config, text_model_config=config.text_config, tokenizer=tokenizer
)
exported = export_lib.ExportedModelArtifacts(
    prefill_decode_model_path=os.path.join(WORK, "model_quantized.tflite"),
    embedder_model_path=os.path.join(WORK, "embedder_quantized.tflite"),
    additional_model_paths={"per_layer_embedder": os.path.join(WORK, "per_layer_embedder_quantized.tflite")},
    tokenizer_model_path=os.path.join(WORK, "tokenizer.json"),
)
for p in [exported.prefill_decode_model_path, exported.embedder_model_path,
          exported.tokenizer_model_path, *exported.additional_model_paths.values()]:
    assert os.path.exists(p), p
exported = litert_lm_builder.package_model(source, export_config, exported)
print("bundle:", exported.litert_lm_model_path, os.path.getsize(exported.litert_lm_model_path) / 1e9, "GB")
