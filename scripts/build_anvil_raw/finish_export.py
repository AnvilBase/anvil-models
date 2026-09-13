#!/usr/bin/env python3
"""Finish an export that died at the per-layer embedder.

The exporter's per-layer-embedder step copies the 11 GB embedding table into the
graph eagerly, which is what kills it on a 16 GB Mac; the main model and token
embedder got through because they use the lightweight (lazy-constant) path.
This does the same three remaining steps the exporter would have — per-layer
embedder, tokenizer, bundle — with the lightweight path, and with only the
per-layer table in memory rather than the whole model.

    finish_export.py hf out out/<work dir with model_quantized.tflite>
"""

import dataclasses
import gc
import os
import sys

import torch
import transformers
from safetensors import safe_open

from litert_torch.generative.export_hf.core import export_lib
from litert_torch.generative.export_hf.core import exportable_module
from litert_torch.generative.export_hf.core import exportable_module_config
from litert_torch.generative.export_hf.core import litert_lm_builder
from litert_torch.generative.export_hf.model_ext import patches as model_ext_patches
from litert_torch.generative.export_hf.model_ext.gemma4 import exportable_module as gemma4_exportable
from litert_torch._convert import interface as converter_utils

HF, OUT, WORK = sys.argv[1], sys.argv[2], sys.argv[3]

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
)
print("cache_length after magic-number rounding:", export_config.cache_length)

config = transformers.AutoConfig.from_pretrained(HF, dtype=torch.float32)
for c in (config, getattr(config, "text_config", None)):
    if c is not None and hasattr(c, "allow_global_per_layer_attribute_access"):
        c.allow_global_per_layer_attribute_access = True
config._attn_implementation = "lrt_transposed_attention"
tokenizer = transformers.AutoTokenizer.from_pretrained(HF)
text_config = config.text_config

# The model, with nothing in it: every parameter on the meta device. Only the
# per-layer embedding is then given real weights, since that is all this step
# touches. The bundling step reads config and generation_config from it too.
with model_ext_patches.get_patch_context(config.model_type):
    with torch.device("meta"):
        model = transformers.Gemma4ForConditionalGeneration(config)
model.generation_config = transformers.GenerationConfig.from_pretrained(HF)

table = transformers.models.gemma4.modeling_gemma4.Gemma4TextScaledWordEmbedding(
    text_config.vocab_size_per_layer_input,
    text_config.num_hidden_layers * text_config.hidden_size_per_layer_input,
    model.model.language_model.padding_idx,
    embed_scale=text_config.hidden_size_per_layer_input**0.5,
)
with safe_open(os.path.join(HF, "model-00001.safetensors"), framework="pt") as f:
    weight = f.get_tensor("model.language_model.embed_tokens_per_layer.weight")
with torch.no_grad():
    table.weight.copy_(weight.to(torch.float32))
del weight
print("per-layer table:", tuple(table.weight.shape), table.weight.dtype, flush=True)


# What the exportable module actually calls is model.model.language_model
# .get_per_layer_inputs(token_ids, None): one lookup and a reshape. torch.export
# would still walk every parameter of whatever module it is handed, and the
# meta model's have no data, so the exportable gets a stand-in holding only
# the real table, shaped like the path it takes through the model.
class LanguageModelStandIn(torch.nn.Module):
    def __init__(self, table, n_layers, per_layer_dim):
        super().__init__()
        self.embed_tokens_per_layer = table
        self.n_layers, self.per_layer_dim = n_layers, per_layer_dim

    def get_per_layer_inputs(self, input_ids, inputs_embeds):
        del inputs_embeds
        return self.embed_tokens_per_layer(input_ids).reshape(
            *input_ids.shape, self.n_layers, self.per_layer_dim
        )


class ModelStandIn(torch.nn.Module):
    def __init__(self, language_model):
        super().__init__()
        self.language_model = language_model


class OuterStandIn(torch.nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.model = inner


stand_in = OuterStandIn(
    ModelStandIn(
        LanguageModelStandIn(
            table, text_config.num_hidden_layers, text_config.hidden_size_per_layer_input
        )
    )
)

source = export_lib.SourceModelArtifacts(
    model=model, model_config=config, text_model_config=text_config, tokenizer=tokenizer
)
exported = export_lib.ExportedModelArtifacts(
    prefill_decode_model_path=os.path.join(WORK, "model_quantized.tflite"),
    embedder_model_path=os.path.join(WORK, "embedder_quantized.tflite"),
)
for p in (exported.prefill_decode_model_path, exported.embedder_model_path):
    assert os.path.exists(p), p

# --- per-layer embedder, the lightweight way ---------------------------------
module = gemma4_exportable.LiteRTExportableModuleForPerLayerEmbedder(stand_in)
converter = converter_utils.Converter()
for name, (sample, _) in module.get_sample_inputs(text_config, export_config).items():
    converter.add_signature(name, module.eval(), sample_kwargs=sample)
print("converting per-layer embedder...", flush=True)
lrt_model = converter.convert(lightweight_conversion=True, strict_export=False)
ple_path = os.path.join(WORK, "per_layer_embedder.tflite")
lrt_model.export(ple_path)
del lrt_model, converter
gc.collect()
print("quantizing per-layer embedder...", flush=True)
ple_path = export_lib.maybe_quantize_model(ple_path, export_config.quantization_recipe)
gc.collect()
exported = dataclasses.replace(exported, additional_model_paths={"per_layer_embedder": ple_path})
print("per-layer embedder:", ple_path, os.path.getsize(ple_path) / 1e9, "GB", flush=True)

# --- tokenizer and bundle, exactly as the exporter does them -------------------
exported = export_lib.export_tokenizer(source, export_config, exported)
print("tokenizer:", exported.tokenizer_model_path, flush=True)
exported = litert_lm_builder.package_model(source, export_config, exported)
print("bundle:", exported.litert_lm_model_path, os.path.getsize(exported.litert_lm_model_path) / 1e9, "GB")
