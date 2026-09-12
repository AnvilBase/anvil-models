# anvil-models

The model catalog for [Anvil](https://www.anvilai.com). The app reads
`https://www.anvilai.com/api/models`, which serves the `models.json` in this
repository with every download URL pointed back at anvilai.com. The bytes
themselves live in this repository's GitHub releases.

Nothing large is committed here. A `.litertlm` file is several gigabytes; git
caps a file at 100 MB and a GitHub release asset at 2 GiB, so each model is
**split into 512 MB parts** and uploaded as release assets. The app downloads
the parts, checks each one against its SHA-256, appends them into a single file,
and deletes each part as it goes — so a phone only needs the model's own size
free, plus one part.

## What's in here

| File | Role |
| --- | --- |
| `models.json` | The catalog the app reads: one entry per model, with its parts and their hashes. Generated — don't hand-edit it. |
| `sources.json` | Where each model comes from and what it's called in Anvil. This is the file you edit. |
| `scripts/publish_model.py` | Downloads a source model, splits it, uploads the parts, and rewrites `models.json`. |

## Publishing a model

```sh
scripts/publish_model.py anvil-lite --push
```

That downloads the source (resumable — run it again if it drops), splits it,
creates the `anvil-lite-v1` release, uploads the parts, rewrites `models.json`,
and pushes. The app picks it up on its next launch; there's nothing to deploy.

Useful flags: `--skip-upload` for a dry run that stops before touching GitHub,
`--keep-work` to leave the download and the parts in `work/`, and `--repo` to
publish somewhere else.

Requirements: `python3`, and the [GitHub CLI](https://cli.github.com) signed in
with write access to this repository (`gh auth login`). `work/` needs about
twice the model's size free while it runs.

## Adding a new model

Add an entry to `sources.json` and run the script with its key:

```json
"anvil-lite": {
  "name": "Anvil Lite",
  "version": "1",
  "summary": "Runs entirely on your phone. Text, photos, and web search.",
  "fileName": "anvil-lite.litertlm",
  "source": "https://huggingface.co/…/gemma-4-E4B-it.litertlm",
  "basedOn": "Gemma 4 E4B (litert-lm)",
  "license": "Apache-2.0",
  "recommended": true,
  "minimumFreeBytes": 4500000000
}
```

`version` is part of the release tag (`anvil-lite-v1`) and is how the app knows
an installed model is out of date. Bump it when the underlying file changes;
re-running the script with the same version replaces the assets in place.

Only `.litertlm` files built for LiteRT-LM work — that's the format the app's
engine loads.

## Putting a model together by hand

The parts are plain byte ranges, in order:

```sh
gh release download anvil-lite-v1 --repo AnvilBase/anvil-models --pattern '*.part*'
cat anvil-lite.litertlm.part* > anvil-lite.litertlm
shasum -a 256 anvil-lite.litertlm   # compare with sha256 in models.json
```

## Licensing

Anvil renames these models, it doesn't retrain them. Each entry in `models.json`
carries the `basedOn` model and its `license`, and the app shows both before
downloading. Anvil Lite and Anvil Nano are Gemma models, used under the
[Gemma Terms of Use](https://ai.google.dev/gemma/terms) — which permit
redistribution, including renamed, as long as those terms travel with the file
and recipients are told what it's based on.
