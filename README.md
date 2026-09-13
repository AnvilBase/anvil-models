# anvil-models

The model catalog for [Anvil](https://www.anvilai.com). The app reads
`https://www.anvilai.com/api/models`, which serves the `models.json` in this
repository with every download URL pointed back at anvilai.com. The bytes
themselves live in this repository's GitHub releases.

The catalog offers one model to everyone: the **Anvil Model** (`anvil-forge`,
built from Gemma 4 E2B). The app's install screen shows it by that name with its
size and parameter count, and installs nothing else. Keep it that way — the app
is built around there being one model to download, not a choice.

A second model, **Anvil Core** (`anvil-core`), is for Anvil Pro. Its entry
carries `"pro": true`, which the app reads as: list it in Settings › Model
behind the Pro badge, and download or switch to it only while Pro is active.
It is built from HauhauCS's uncensored Gemma 4 E4B, which is published only as
GGUF. The app's engine loads `.litertlm` alone, so the entry sits in
`sources.json` with no `source` until a `.litertlm` build of it exists; the
publish script refuses to publish it before then, and it is not in
`models.json`.

Nothing large is committed here. A `.litertlm` file is several gigabytes; git
caps a file at 100 MB and a GitHub release asset at 2 GiB, so each model is
**split into 512 MB parts** and uploaded as release assets. The app downloads
the parts, checks each one against its SHA-256, appends them into a single file,
and deletes each part as it goes — so a phone only needs the model's own size
free, plus one part.

## What's in here

| File | Role |
| --- | --- |
| `models.json` | The catalog the app reads: the model, with its parts and their hashes. Generated — don't hand-edit it. |
| `sources.json` | Where each model comes from and what it's called in Anvil. This is the file you edit. |
| `scripts/publish_model.py` | Downloads a source model, splits it, uploads the parts, and rewrites `models.json`. |

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
  "name": "Anvil Model",
  "version": "3",
  "summary": "Runs entirely on your phone",
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
leave it off the Anvil Model. `upstream` is documentation only — where a model
comes from when `source` can't point at it yet.

Only `.litertlm` files built for LiteRT-LM work — that's the format the app's
engine loads — and the script refuses anything else. A GGUF has to be converted
first, from the model's safetensors with Google's LiteRT-LM tooling; there is no
GGUF-to-litertlm path.

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

The Anvil Model is Gemma 4 E2B, redistributed under the
[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). Apache-2.0
asks that the licence and any notices travel with the work; it does not require
that a product keep the upstream's name, and section 6 grants no trademark
rights, so renaming is the correct thing to do rather than a liberty taken. The
`source` field in `sources.json` is what the file was built from — that's the
build recipe and stays accurate.

Anvil Core is Gemma 4 E4B with its refusals removed, redistributed under the
[Gemma Terms of Use](https://ai.google.dev/gemma/terms) rather than Apache-2.0.
Those terms travel with every copy and include Google's Prohibited Use Policy,
which binds whoever runs the model whatever the weights will say; the app
offers the model under those terms and the catalog entry records them.

Check the upstream licence before swapping in a different model. Not every
open model is Apache-2.0, and some carry terms that do dictate naming and
attribution.
