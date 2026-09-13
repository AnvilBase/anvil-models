#!/usr/bin/env python3
"""Publish one model from sources.json to the anvil-models GitHub release.

A .litertlm file is several gigabytes, and GitHub caps a release asset at 2 GiB,
so the file is split into parts. The Anvil app downloads the parts, checks each
one against the SHA-256 in models.json, and appends them back into one file.

    scripts/publish_model.py anvil-forge

Steps: download the source (resumable), split it into parts, hash everything,
create or reuse the release, upload the parts, and rewrite models.json. Run it
again after an interrupted run; finished work is reused.

Options:
    --skip-upload   Do everything except touching GitHub. Useful for a dry run.
    --push          git commit and push models.json when the upload succeeds.
    --repo OWNER/N  Publish somewhere other than AnvilBase/anvil-models.
    --keep-work     Leave the downloaded source and the parts on disk.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCES = ROOT / "sources.json"
CATALOG = ROOT / "models.json"
WORK = ROOT / "work"
DEFAULT_REPO = "AnvilBase/anvil-models"
UPLOAD_CONCURRENCY = 3
READ_SIZE = 8 * 1024 * 1024


# --- small helpers ----------------------------------------------------------


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} TB"


def progress(done: int, total: int, started: float, label: str) -> None:
    if not sys.stderr.isatty():
        return
    elapsed = max(time.monotonic() - started, 0.001)
    rate = done / elapsed
    share = f"{done / total * 100:5.1f}%" if total else "  ?  "
    left = f" · {human((total - done) / rate)[:-1]}s left" if total and rate else ""
    sys.stderr.write(f"\r  {label} {share}  {human(done)} of {human(total)}  {human(rate)}/s{left}   ")
    sys.stderr.flush()


def done_line() -> None:
    if sys.stderr.isatty():
        sys.stderr.write("\n")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(READ_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    print("  $ " + " ".join(cmd))
    return subprocess.run(cmd, check=True, **kwargs)


# --- download ---------------------------------------------------------------


def remote_size(url: str) -> int | None:
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(request) as response:
            length = response.headers.get("Content-Length")
            return int(length) if length else None
    except urllib.error.URLError as err:
        sys.exit(f"Could not reach {url}: {err}")


def download(url: str, destination: Path) -> None:
    """Fetch the source file, resuming a partial download with a Range request."""
    expected = remote_size(url)
    have = destination.stat().st_size if destination.exists() else 0

    if expected and have == expected:
        print(f"  already downloaded ({human(have)})")
        return
    if expected and have > expected:
        print("  local file is larger than the source; starting over")
        destination.unlink()
        have = 0

    headers = {"Range": f"bytes={have}-"} if have else {}
    if have:
        print(f"  resuming at {human(have)}")

    request = urllib.request.Request(url, headers=headers)
    started = time.monotonic()
    with urllib.request.urlopen(request) as response:
        # A server that ignores the Range header replies 200 with the whole file.
        if have and response.status != 206:
            print("  the server ignored the resume request; starting over")
            have = 0
        mode = "ab" if have else "wb"
        total = have + int(response.headers.get("Content-Length", 0))
        with destination.open(mode) as f:
            written = have
            while chunk := response.read(READ_SIZE):
                f.write(chunk)
                written += len(chunk)
                progress(written, total, started, "downloading")
    done_line()

    actual = destination.stat().st_size
    if expected and actual != expected:
        sys.exit(f"Download is {human(actual)}, expected {human(expected)}. Run again to resume.")


# --- split ------------------------------------------------------------------


def split(source: Path, parts_dir: Path, file_name: str, part_size: int) -> tuple[list[dict], str]:
    """Cut the file into parts, hashing each part and the whole file in one pass."""
    if parts_dir.exists():
        shutil.rmtree(parts_dir)
    parts_dir.mkdir(parents=True)

    total = source.stat().st_size
    whole = hashlib.sha256()
    parts: list[dict] = []
    started = time.monotonic()
    read = 0

    with source.open("rb") as f:
        index = 0
        while True:
            part_path = parts_dir / f"{file_name}.part{index:03d}"
            part_digest = hashlib.sha256()
            written = 0
            with part_path.open("wb") as out:
                while written < part_size:
                    chunk = f.read(min(READ_SIZE, part_size - written))
                    if not chunk:
                        break
                    out.write(chunk)
                    part_digest.update(chunk)
                    whole.update(chunk)
                    written += len(chunk)
                    read += len(chunk)
                    progress(read, total, started, "splitting  ")
            if written == 0:
                part_path.unlink()
                break
            parts.append(
                {"name": part_path.name, "sizeBytes": written, "sha256": part_digest.hexdigest()}
            )
            if written < part_size:
                break
            index += 1
    done_line()
    return parts, whole.hexdigest()


# --- release ----------------------------------------------------------------


def release_exists(repo: str, tag: str) -> bool:
    result = subprocess.run(
        ["gh", "release", "view", tag, "--repo", repo],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def publish_release(repo: str, tag: str, spec: dict, parts: list[dict], parts_dir: Path) -> None:
    if not shutil.which("gh"):
        sys.exit("The GitHub CLI (gh) is not installed. brew install gh, then gh auth login.")

    if not release_exists(repo, tag):
        origin = (
            f"Based on {spec['basedOn']}, licensed under {spec['license']}."
            if spec.get("basedOn")
            else f"An open model redistributed under {spec['license']}"
            f" ({spec.get('licenseURL', '')}).".rstrip()
        )
        notes = (
            f"{spec['name']} for the Anvil app.\n\n"
            f"{origin}\n\n"
            f"The model is split into {len(parts)} parts because a GitHub release asset "
            "is capped at 2 GiB. The app downloads the parts listed in `models.json`, "
            "checks each one against its SHA-256, and appends them into a single "
            f"`{spec['fileName']}`. Downloading a part by hand and concatenating them "
            "in order produces the same file."
        )
        run(["gh", "release", "create", tag, "--repo", repo, "--title", f"{spec['name']} v{spec['version']}", "--notes", notes])
    else:
        print(f"  release {tag} already exists; uploading into it")

    # Several at once: one upload rarely saturates a connection, and each part is still its own
    # command, so an interrupted run picks up from whichever parts already landed.
    def upload(part):
        subprocess.run(
            ["gh", "release", "upload", tag, str(parts_dir / part["name"]), "--repo", repo, "--clobber"],
            check=True, capture_output=True, text=True)
        return part["name"]

    with concurrent.futures.ThreadPoolExecutor(max_workers=UPLOAD_CONCURRENCY) as pool:
        futures = {pool.submit(upload, part): part for part in parts}
        done = 0
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]["name"]
            try:
                future.result()
            except subprocess.CalledProcessError as err:
                sys.exit(f"Uploading {name} failed:\n{err.stderr}")
            done += 1
            print(f"  uploaded {name}  ({done}/{len(parts)})")


# --- catalog ----------------------------------------------------------------


def write_catalog(model_id: str, spec: dict, tag: str, parts: list[dict], sha: str, size: int) -> None:
    catalog = json.loads(CATALOG.read_text()) if CATALOG.exists() else {"schemaVersion": 1, "models": []}
    entry = {
        "id": model_id,
        "name": spec["name"],
        "version": spec["version"],
        "summary": spec["summary"],
        "parameters": spec.get("parameters"),
        "fileName": spec["fileName"],
        "sizeBytes": size,
        "sha256": sha,
        "minimumFreeBytes": spec.get("minimumFreeBytes", int(size * 1.15)),
        "recommended": bool(spec.get("recommended", False)),
        # Anvil Pro only. The app shows the model behind the paywall until Pro is active.
        "pro": bool(spec.get("pro", False)),
        "basedOn": spec.get("basedOn"),
        "license": spec["license"],
        "licenseURL": spec.get("licenseURL"),
        "release": tag,
        "parts": parts,
    }
    models = [m for m in catalog.get("models", []) if m.get("id") != model_id]
    models.append(entry)
    # Recommended first, then alphabetically, so the app can render the list as it comes.
    models.sort(key=lambda m: (not m.get("recommended"), m.get("name", "")))
    catalog["schemaVersion"] = 1
    catalog["models"] = models
    CATALOG.write_text(json.dumps(catalog, indent=2) + "\n")
    print(f"  wrote {CATALOG.relative_to(ROOT)}")


# --- main -------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", help="a key from sources.json, e.g. anvil-forge")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--skip-upload", action="store_true", help="do everything except touch GitHub")
    parser.add_argument("--push", action="store_true", help="commit and push models.json afterwards")
    parser.add_argument("--keep-work", action="store_true", help="keep the download and the parts")
    args = parser.parse_args()

    sources = json.loads(SOURCES.read_text())
    spec = sources["models"].get(args.model)
    if spec is None:
        sys.exit(f"{args.model} is not in sources.json. Known: {', '.join(sources['models'])}")

    # The app's engine is LiteRT-LM and loads nothing but .litertlm. A GGUF would split, upload and
    # download fine, and then fail to load on every phone — so it is refused here, before any of that.
    source = spec.get("source")
    if not source:
        upstream = spec.get("upstream", "an upstream file")
        sys.exit(
            f"{args.model} has no source to publish yet: it needs a .litertlm build of {upstream}. "
            "Point \"source\" at one in sources.json and run again."
        )
    if not source.split("?")[0].endswith(".litertlm"):
        sys.exit(f"{args.model}'s source is not a .litertlm file, which is the only format the app loads: {source}")

    part_size = int(sources.get("partSizeBytes", 512 * 1024 * 1024))
    if part_size > 2 * 1024**3:
        sys.exit("partSizeBytes must stay under 2 GiB, the GitHub release asset limit.")

    work = WORK / args.model
    work.mkdir(parents=True, exist_ok=True)
    source_file = work / spec["fileName"]
    parts_dir = work / "parts"
    tag = f"{args.model}-v{spec['version']}"

    print(f"\n{spec['name']}  ({args.model}, release {tag})")
    print(f"\nsource  {source}")
    download(source, source_file)

    size = source_file.stat().st_size
    print(f"\nsplitting {human(size)} into {part_size // 1024 // 1024} MB parts")
    parts, sha = split(source_file, parts_dir, spec["fileName"], part_size)
    print(f"  {len(parts)} parts · sha256 {sha}")

    if args.skip_upload:
        print("\nskipping the upload (--skip-upload)")
    else:
        print(f"\nuploading to {args.repo}")
        publish_release(args.repo, tag, spec, parts, parts_dir)

    print("\ncatalog")
    write_catalog(args.model, spec, tag, parts, sha, size)

    if args.push and not args.skip_upload:
        print("\npushing the catalog")
        run(["git", "-C", str(ROOT), "add", "models.json"])
        run(["git", "-C", str(ROOT), "commit", "-m", f"Publish {spec['name']} v{spec['version']}"])
        run(["git", "-C", str(ROOT), "push"])

    if not args.keep_work:
        shutil.rmtree(work, ignore_errors=True)
        print(f"\nremoved {work.relative_to(ROOT)} (keep it next time with --keep-work)")

    print(f"\nDone. {spec['name']} is {len(parts)} parts in {tag}.")
    if not args.push:
        print("Commit and push models.json to put it in front of the app.")


if __name__ == "__main__":
    main()
