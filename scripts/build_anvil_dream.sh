#!/bin/zsh
# Builds build/anvil-dream.aar, the Anvil Dream model, from Realism by Stable Yogi V5 XL Lightning.
#
# Anvil Dream is an SDXL model run through Apple's Core ML build of Stable Diffusion. It isn't
# converted here: LocalMuseAI publishes a Core ML conversion of the exact Civitai checkpoint —
# split-einsum attention for the Neural Engine, a 6-bit palettized U-Net in two chunks, an 8-bit
# second text encoder, 1024×1024 with a guidance batch of two — laid out the way Apple's Swift
# pipeline names things. This fetches that package at a pinned revision, leaves out the VAE encoder
# the app doesn't use, and packs the rest as the Apple Archive the app unpacks on the phone.
# PROVENANCE.json travels with it: the app reads the sampler, steps and guidance from it.
#
# Needs: curl, python3, Xcode's command line tools (for aa), and about 7 GB free.
# Then: scripts/publish_model.py anvil-dream --push
set -euo pipefail
cd "$(dirname "$0")/.."

REPO=LocalMuseAI/coreml-realism-by-stable-yogi-v5-xl-lightning-6bit
REVISION=04fb4eb9892dbe48adbf446ea3b099bb110c9d34
WORK=work/anvil-dream-build
PACKAGE=$WORK/package

mkdir -p build "$PACKAGE"
curl -sfL "https://huggingface.co/api/models/$REPO/tree/$REVISION?recursive=1" |
  python3 -c '
import json, sys
for item in json.load(sys.stdin):
    if item["type"] == "file" and not item["path"].startswith("VAEEncoder") and item["path"] != ".gitattributes":
        print(item["path"], item["size"])
' > "$WORK/files.txt"

# Resumable: a file already down at its full size is left alone.
while read -r file bytes; do
  mkdir -p "$PACKAGE/${file:h}"
  if [ -f "$PACKAGE/$file" ] && [ "$(stat -f%z "$PACKAGE/$file")" = "$bytes" ]; then continue; fi
  echo "fetching $file"
  curl -sfL --retry 3 -o "$PACKAGE/$file" "https://huggingface.co/$REPO/resolve/$REVISION/$file"
  [ "$(stat -f%z "$PACKAGE/$file")" = "$bytes" ] || { echo "$file came down the wrong size" >&2; exit 1; }
done < "$WORK/files.txt"

# The archive holds the package's contents at its root — the compiled models, the tokenizer files,
# PROVENANCE.json and LICENSE — which is exactly what the app looks for once it has unpacked it.
rm -f build/anvil-dream.aar
aa archive -d "$PACKAGE" -o build/anvil-dream.aar -a lzfse
ls -la build/anvil-dream.aar
echo "Built build/anvil-dream.aar. Publish it with: scripts/publish_model.py anvil-dream --push"
