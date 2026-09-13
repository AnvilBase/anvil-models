#!/usr/bin/env python3
"""Export and quantize only the prefill/decode model, with a recipe of your choosing.

The exporter's own run does this as its first step, then dies on this Mac at
the per-layer embedder. This repeats just that first step so the main model
can be re-quantized — with Google's Gemma 4 recipe, which keeps the per-layer
projections at 8 bits — without redoing anything else.

    export_main_model.py hf out/<work dir> recipe.json
"""

import os
import sys

from litert_torch.generative.export_hf.core import export_lib
from litert_torch.generative.export_hf.core import exportable_module
from litert_torch.generative.export_hf.core import exportable_module_config

HF, WORK, RECIPE = sys.argv[1:4]
os.makedirs(WORK, exist_ok=True)

export_config = exportable_module.ExportableModuleConfig(
    model=HF,
    output_dir=os.path.dirname(WORK.rstrip("/")),
    work_dir=WORK,
    task=exportable_module_config.ExportTask.TEXT_GENERATION,
    keep_temporary_files=True,
    prefill_lengths=[128, 1024],
    cache_length=32000,
    enable_gpu_dynamic_cache=True,
    externalize_embedder=True,
    single_token_embedder=True,
    bundle_litert_lm=True,
    quantization_recipe=RECIPE,
    sampler_top_k=64,
    sampler_top_p=0.95,
    sampler_temperature=1.0,
    experimental_lightweight_conversion=True,
)
export_config.print_summary()

source = export_lib.load_model(HF, export_config, task=export_config.task)
export_config = export_lib.update_export_config(export_config, source)
exported = export_lib.ExportedModelArtifacts()
exported = export_lib.export_text_prefill_decode_model(source, export_config, exported)
print("prefill/decode:", exported.prefill_decode_model_path,
      os.path.getsize(exported.prefill_decode_model_path) / 1e9, "GB", flush=True)
