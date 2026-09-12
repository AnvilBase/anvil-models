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
scripts/publish_model.py anvil-forge --push
```

That downloads the source (resumable — run it again if it drops), splits it,
creates the `anvil-forge-v1` release, uploads the parts, rewrites `models.json`,
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
"anvil-forge": {
  "name": "Anvil Forge",
  "version": "1",
  "summary": "Runs entirely on your phone. Text, photos, and web search.",
  "fileName": "anvil-forge.litertlm",
  "source": "https://huggingface.co/…/the-upstream-file.litertlm",
  "license": "Apache-2.0",
  "licenseURL": "https://www.apache.org/licenses/LICENSE-2.0",
  "recommended": true,
  "minimumFreeBytes": 4500000000
}
```

`version` is part of the release tag (`anvil-forge-v1`) and is how the app knows
an installed model is out of date. Bump it when the underlying file changes;
re-running the script with the same version replaces the assets in place.

Only `.litertlm` files built for LiteRT-LM work — that's the format the app's
engine loads.

## Putting a model together by hand

The parts are plain byte ranges, in order:

```sh
gh release download anvil-forge-v1 --repo AnvilBase/anvil-models --pattern '*.part*'
cat anvil-forge.litertlm.part* > anvil-forge.litertlm
shasum -a 256 anvil-forge.litertlm   # compare with sha256 in models.json
```

## Licensing

Anvil renames these models, it doesn't retrain them. Every entry in
`models.json` records the `license` the model is redistributed under, and the
release carries the same, so the licence travels with the file.

Anvil Forge is redistributed under the
[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). Apache-2.0
asks that the licence and any notices travel with the work; it does not require
that a product keep the upstream's name, and section 6 grants no trademark
rights, so renaming is the correct thing to do rather than a liberty taken. The
`source` field in `sources.json` is what the file was built from — that's the
build recipe and stays accurate.

Check the upstream licence before adding a model. Not every open model is
Apache-2.0, and some carry terms that do dictate naming and attribution.
