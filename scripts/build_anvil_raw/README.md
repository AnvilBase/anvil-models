# Building Anvil Raw

Anvil Raw is [HauhauCS's uncensored Gemma 4 E4B](https://huggingface.co/HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive),
which is published only as GGUF. The app's engine loads `.litertlm` alone, and
there is no GGUF-to-litertlm converter, so the file is rebuilt in stages: the
GGUF is dequantised back into a Hugging Face checkpoint, and Google's
`litert_torch` exporter turns that into a `.litertlm` the way Google builds its
own Gemma 4 files. Everything here ran on a 16 GB MacBook Air; the exporter
wants more, which is why two of its steps are done by hand.

```sh
cd work/anvil-raw
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install litert-torch litert-lm-builder litert-lm gguf "transformers>=5" torch safetensors sentencepiece

# 1. The source, at its highest-quality quantisation (8.1 GB).
curl -L -C - -o gguf/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf \
  https://huggingface.co/HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive/resolve/main/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf

# 2. Config and tokenizer from the base model (abliteration changes neither).
mkdir hf; for f in config.json tokenizer.json tokenizer_config.json chat_template.jinja generation_config.json; do
  curl -sL -o hf/$f https://huggingface.co/unsloth/gemma-4-E4B-it/resolve/main/$f; done

# 3. GGUF -> Hugging Face checkpoint, tensor by tensor (16 GB of bf16 safetensors).
python dequant_to_hf.py gguf/*.gguf hf

# 4. Main model + token embedder, 4-bit with Google's Gemma 4 recipe (per-layer
#    projections stay 8-bit; the generic int4 recipe wrecks the model).
python export_main_model.py hf out/main recipe_tf_lite_prefill_decode.json
python -m litert_torch.generative.export_hf hf out --task text_generation \
  --quantization_recipe dynamic_wi4_afp32 --prefill_lengths 128,1024 \
  --cache_length 32000 --enable_gpu_dynamic_cache True --externalize_embedder True \
  --single_token_embedder True --experimental_lightweight_conversion True   # dies at the per-layer embedder; keeps embedder_quantized.tflite

# 5. Per-layer embedder (the exporter runs out of memory on it), then the bundle
#    with LiteRT-LM's canonical Gemma 4 template — Hugging Face's uses `.get()`,
#    which LiteRT-LM's template engine doesn't have.
python finish_export.py hf out out/<work dir>
python repack.py hf out out/<work dir> chat_template_e2b_e4b.jinja

# 6. Ask it something before publishing.
python test_model.py out/model.litertlm
cp out/model.litertlm ../../build/anvil-raw.litertlm
```

The check that matters: run the same prompts greedily through the original GGUF
with llama.cpp (`llama-completion -no-cnv --temp 0`) and through the bundle.
They should agree on arithmetic and read the same. When the bundle rambles and
the GGUF doesn't, it is the quantisation recipe.
