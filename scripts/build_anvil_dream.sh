#!/bin/zsh
# Builds build/anvil-dream.aar, the Anvil Dream model, from LCM Dreamshaper v7.
#
# Anvil Dream is a Latent Consistency Model run through Apple's Core ML build of Stable Diffusion.
# Three steps, all on this Mac: fetch the diffusers weights from Hugging Face, fold the guidance
# scale into the U-Net so Apple's converter accepts it unchanged (scripts/fold_lcm_guidance.py),
# and convert with apple/ml-stable-diffusion — split-einsum attention for the Neural Engine, 6-bit
# palettized weights, batch of one (no classifier-free guidance: it was distilled in), the U-Net in
# two chunks — then pack the compiled resources as an Apple Archive the app unpacks on the phone.
#
# Needs: uv (https://docs.astral.sh/uv), Xcode's command line tools (for coremlcompiler and aa),
# about 20 GB free, and an hour. Then: scripts/publish_model.py anvil-dream --push
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p build work/anvil-dream-build
cd work/anvil-dream-build

GUIDANCE=8.0   # The guidance scale the model card recommends; folded into the weights.

if [ ! -d ml-stable-diffusion ]; then
  git clone --depth 1 https://github.com/apple/ml-stable-diffusion.git
fi

if [ ! -d env ]; then
  uv venv --python 3.11 env
  source env/bin/activate
  # Apple's converter is tested against torch 2.7; newer torch trips coremltools.
  uv pip install "torch==2.7.0" "numpy<2" "diffusers[torch]==0.30.2" "transformers==4.44.2" \
    "huggingface-hub==0.24.6" "coremltools>=8.0" scipy safetensors accelerate pytest \
    scikit-learn matplotlib requests
else
  source env/bin/activate
fi

# The converter imports diffusionkit at the top for SD3, which isn't needed here.
mkdir -p stubs/diffusionkit/tests
touch stubs/diffusionkit/__init__.py stubs/diffusionkit/tests/__init__.py
printf 'def convert_mmdit_to_mlpackage(*a, **k):\n    raise NotImplementedError\n\ndef convert_vae_to_mlpackage(*a, **k):\n    raise NotImplementedError\n' > stubs/diffusionkit/tests/torch2coreml.py
printf '__version__ = "stub"\n' > stubs/diffusionkit/version.py

if [ ! -f LCM_Dreamshaper_v7/unet/diffusion_pytorch_model.safetensors ]; then
  python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("SimianLuo/LCM_Dreamshaper_v7", local_dir="LCM_Dreamshaper_v7", allow_patterns=[
    "model_index.json", "unet/config.json", "unet/diffusion_pytorch_model.safetensors",
    "text_encoder/config.json", "text_encoder/model.safetensors",
    "vae/config.json", "vae/diffusion_pytorch_model.safetensors",
    "tokenizer/*", "scheduler/*", "feature_extractor/*"])
PY
fi

if [ ! -f folded/model_index.json ]; then
  python ../../scripts/fold_lcm_guidance.py LCM_Dreamshaper_v7 folded "$GUIDANCE"
fi

export PYTHONPATH="$PWD/ml-stable-diffusion:$PWD/stubs"
# Xcode's toolchain, not the command line tools: coremlcompiler lives only in the former.
export DEVELOPER_DIR="${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}"

# Convert and quantize. Chunking and bundling are done below rather than by the converter's own
# flags: it chunks the U-Net before quantizing it, which would ship the full-size chunks.
P=out/Stable_Diffusion_version_folded
if [ ! -d ${P}_unet.mlpackage ]; then
  python -m python_coreml_stable_diffusion.torch2coreml \
    --convert-unet --convert-text-encoder --convert-vae-decoder \
    --model-version folded \
    --unet-batch-one \
    --attention-implementation SPLIT_EINSUM_V2 \
    --quantize-nbits 6 --min-deployment-target iOS17 \
    --latent-h 64 --latent-w 64 \
    -o out
fi

# The quantized U-Net in two chunks, then everything compiled into the Resources folder the app
# expects, named the way Apple's Swift pipeline names them, with the tokenizer files beside.
rm -rf ${P}_unet_chunk1.mlpackage ${P}_unet_chunk2.mlpackage out/Resources
python -m python_coreml_stable_diffusion.chunk_mlprogram --mlpackage-path ${P}_unet.mlpackage -o out
mkdir -p out/Resources
for pair in text_encoder:TextEncoder vae_decoder:VAEDecoder unet_chunk1:UnetChunk1 unet_chunk2:UnetChunk2; do
  src=${pair%%:*}; dst=${pair##*:}
  xcrun coremlcompiler compile ${P}_${src}.mlpackage out/Resources
  mv out/Resources/Stable_Diffusion_version_folded_${src}.mlmodelc out/Resources/${dst}.mlmodelc
done
cp LCM_Dreamshaper_v7/tokenizer/vocab.json LCM_Dreamshaper_v7/tokenizer/merges.txt out/Resources/

# The archive holds the Resources folder's contents at its root: the compiled models and the
# tokenizer files, which is exactly what the app looks for once it has unpacked it.
rm -f ../../build/anvil-dream.aar
aa archive -d out/Resources -o ../../build/anvil-dream.aar -a lzfse
ls -la ../../build/anvil-dream.aar
echo "Built build/anvil-dream.aar. Publish it with: scripts/publish_model.py anvil-dream --push"
