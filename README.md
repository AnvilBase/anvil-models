# anvil-models

The model catalog for [Anvil](https://www.anvilai.com). The app reads
`https://www.anvilai.com/api/models`, which serves the `models.json` in this
repository with every download URL pointed back at anvilai.com. The bytes
themselves live in this repository's GitHub releases.

The catalog lists three models, in this order — the order the app shows them
in, on the install screen and in Settings › Models:

1. **Anvil Core** (`anvil-forge`, built from Gemma 4 E2B), the everyday model,
   free to everyone. The key stays `anvil-forge`: it is how installed copies
   are matched to the catalog.
2. **Anvil Raw** (`anvil-raw`), the unrestricted model, for Anvil Pro. Its
   entry carries `"pro": true`, which the app reads as: list it behind the Pro
   badge, and download or switch to it only while Pro is active. It is
   HauhauCS's uncensored Gemma 4 E4B, which is published only as GGUF; the
   `.litertlm` the app loads is rebuilt from it with the scripts in
   `scripts/build_anvil_raw/` (see the README there), and `source` points at
   the result in `build/`.
3. **Anvil Dream** (`anvil-dream`), the image model, for Anvil Pro.

Anvil Dream is a
different kind of model: `"kind": "image"`. It makes pictures rather than
text. The app runs it beside whichever chat model is loaded, through a
`generate_image` tool the chat model calls when someone asks for a picture. It
is [Realism by Stable Yogi V5 XL Lightning](https://civitai.com/models/166609?modelVersionId=1075465),
a photorealistic SDXL model made for a few steps, in
[LocalMuseAI's Core ML conversion](https://huggingface.co/LocalMuseAI/coreml-realism-by-stable-yogi-v5-xl-lightning-6bit)
for the Neural Engine, shipped as an Apple Archive (`.aar`) of the compiled
models, which the app unpacks on the phone. `scripts/build_anvil_dream.sh`
fetches that conversion at a pinned revision and packs it on a Mac, and `source`
in `sources.json` points at what it built.

Nothing large is committed here. A `.litertlm` file is several gigabytes; git
caps a file at 100 MB and a GitHub release asset at 2 GiB, so each model is
**split into 512 MB parts** and uploaded as release assets. The app downloads
the parts, checks each one against its SHA-256, appends them into a single file,
and deletes each part as it goes — so a phone only needs the model's own size
free, plus one part.

## What's in here

| File | Role |
| --- | --- |
| `models.json` | The catalog the app reads: every model in order, with its parts and their hashes. Generated — don't hand-edit it; `scripts/publish_model.py --sync-catalog` rewrites it from `sources.json`. |
| `sources.json` | Where each model comes from and what it's called in Anvil. This is the file you edit. |
| `scripts/publish_model.py` | Downloads (or picks up) a source model, splits it, uploads the parts, and rewrites `models.json`. |
| `scripts/build_anvil_raw/` | Builds Anvil Raw: dequantises HauhauCS's GGUF back to a checkpoint and exports it to `.litertlm` with Google's tooling and Gemma 4 recipe. |
| `scripts/build_anvil_dream.sh` | Builds Anvil Dream: fetches the Core ML conversion of Realism by Stable Yogi V5 XL Lightning at a pinned revision and packs it as `build/anvil-dream.aar`. |
| `scripts/fold_lcm_guidance.py` | Kept from Anvil Dream v1 (LCM Dreamshaper v7): folds an LCM's guidance scale into its U-Net so Apple's converter accepts it. |

## Publishing a model

```sh
scripts/publish_model.py anvil-forge --push
```

That downloads the source (resumable — run it again if it drops), splits it,
creates the `anvil-forge-v2` release, uploads the parts, rewrites `models.json`,
and pushes. The app picks it up on its next launch; there's nothing to deploy.

Useful flags: `--skip-upload` for a dry run that stops before touching GitHub,
`--keep-work` to leave the download and the parts in `work/`, and `--repo` to
publish somewhere else.

Requirements: `python3`, and the [GitHub CLI](https://cli.github.com) signed in
with write access to this repository (`gh auth login`). `work/` needs about
twice the model's size free while it runs.

## Changing the model

The entry in `sources.json` is the whole definition. Point `source` at a new
upstream file, bump `version`, and run the script:

```json
"anvil-forge": {
  "name": "Anvil Core",
  "version": "3",
  "summary": "The everyday model",
  "parameters": "2B",
  "fileName": "anvil-forge.litertlm",
  "source": "https://huggingface.co/…/the-upstream-file.litertlm",
  "license": "Apache-2.0",
  "licenseURL": "https://www.apache.org/licenses/LICENSE-2.0",
  "recommended": true
}
```

`version` is part of the release tag (`anvil-forge-v3`) and is how the app knows
an installed model is out of date. Bump it when the underlying file changes;
re-running the script with the same version replaces the assets in place. Keep
the key `anvil-forge` and the `fileName`: the app looks the model up by that id,
and an installed copy is matched to the catalog by it.

`parameters` is the model's size the way models are sized ("2B"), shown beside
the file size on the install screen. `pro` marks a model as part of Anvil Pro;
leave it off Anvil Core. `upstream` is documentation only — where a model
comes from when `source` can't point at it yet.

Only `.litertlm` files built for LiteRT-LM work — that's the format the app's
engine loads — and the script refuses anything else. A GGUF has to be converted
first, from the model's safetensors with Google's LiteRT-LM tooling; there is no
GGUF-to-litertlm path.

`kind` says which of the app's engines a model is for. Left out, it is `text`.
An `image` model's `fileName` ends in `.aar` and its `source` is usually a path
in this repository rather than a URL: the file is built, not fetched.

## Building Anvil Dream

```sh
scripts/build_anvil_dream.sh            # a 3 GB download; needs Xcode's tools, 7 GB free
scripts/publish_model.py anvil-dream --push
```

Nothing is converted here. An SDXL model needs far more memory to convert than
the result takes to run, so Anvil Dream is LocalMuseAI's Core ML conversion of
the exact Civitai checkpoint: split-einsum attention for the Neural Engine, a
6-bit palettized U-Net in two chunks, an 8-bit second text encoder, 1024×1024,
and a U-Net batch of two for classifier-free guidance, laid out the way Apple's
Swift pipeline names things. The build fetches it at a pinned revision, leaves
out the VAE encoder the app doesn't use, and packs the rest.

The package's `PROVENANCE.json` travels in the archive, and the app reads it:
it names the sampler, steps and guidance the model's creator recommends —
Euler ancestral, seven steps, guidance 1.5 — and the app samples with exactly
those. The result is about 3 GB and makes a 1024×1024 picture in seven passes
of the network. The app needs the build that can run SDXL; one from before it
would try to run this archive as Stable Diffusion 1.5.

Version 1 was LCM Dreamshaper v7, built from the diffusers weights by folding
its guidance scale into the U-Net (`scripts/fold_lcm_guidance.py`) and
converting with Apple's converter; the script that did it is in this
repository's history.

## Putting a model together by hand

The parts are plain byte ranges, in order:

```sh
gh release download anvil-forge-v2 --repo AnvilBase/anvil-models --pattern '*.part*'
cat anvil-forge.litertlm.part* > anvil-forge.litertlm
shasum -a 256 anvil-forge.litertlm   # compare with sha256 in models.json
```

## Licensing

Anvil renames the model, it doesn't retrain it. The entry in `models.json`
records the `license` the model is redistributed under, and the release carries
the same, so the licence travels with the file.

Anvil Core is Gemma 4 E2B, redistributed under the
[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). Apache-2.0
asks that the licence and any notices travel with the work; it does not require
that a product keep the upstream's name, and section 6 grants no trademark
rights, so renaming is the correct thing to do rather than a liberty taken. The
`source` field in `sources.json` is what the file was built from — that's the
build recipe and stays accurate.

Anvil Raw is Gemma 4 E4B with its refusals removed, redistributed under the
[Gemma Terms of Use](https://ai.google.dev/gemma/terms) rather than Apache-2.0.
Those terms travel with every copy and include Google's Prohibited Use Policy,
which binds whoever runs the model whatever the weights will say; the app
offers the model under those terms and the catalog entry records them.

Anvil Dream is Realism by Stable Yogi V5 XL Lightning, a fine-tune of SDXL,
redistributed under the
[CreativeML Open RAIL++-M License](https://github.com/Stability-AI/generative-models/blob/main/model_licenses/LICENSE-SDXL1.0)
it inherits from SDXL. That licence permits redistribution and commercial use,
requires that a copy of it travel with the model — the archive carries the
package's `LICENSE` — and binds whoever uses the model to its use restrictions,
whatever the weights will draw.

Check the upstream licence before swapping in a different model. Not every
open model is Apache-2.0, and some carry terms that do dictate naming and
attribution.
