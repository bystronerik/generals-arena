#!/usr/bin/env python3
"""Copy joe-rs's artifact into unclejoe's, and refuse a partial copy.

    python bots/unclejoe/tools/sync_artifact.py
    python bots/unclejoe/tools/sync_artifact.py --check

unclejoe has no converter of its own. It never reads joe's `.eqx`; it tracks
**joe-rs**, one hop further down the chain joe -> joe-rs -> unclejoe, and this
script is the only knob that moves it. The reason it tracks joe-rs rather than
joe is the measurement: the joe-rs vs unclejoe contrast is supposed to isolate
the tactics layer, and that requires the two bots to run identical weights,
not merely weights from the same checkpoint.

Nothing detects a skipped sync at play time — a seat with older weights loads,
plays, and looks healthy — so the sync is a step in the "After a joe
re-export" checklist in the joe-rs export doc, and this script's job is to
make the *failure* of that step loud. It verifies the digest the manifest
claims against the bytes on both sides, and writes the weights through a
temporary file so an interrupted copy cannot leave a half-written
`model.safetensors` beside a manifest that swears it is complete.

`--check` reports without writing: it answers "is unclejoe on joe-rs's current
weights?" and exits non-zero when it is not.

A sync forks unclejoe's content hash, exactly as a re-conversion forks
joe-rs's. That is intended, and the version registry records it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

BOT_DIR = Path(__file__).resolve().parents[1]
REPO = BOT_DIR.parents[1]
SRC_DIR = REPO / "bots" / "joe-rs" / "artifact"
DST_DIR = BOT_DIR / "artifact"
MEMBERS = ("model.safetensors", "manifest.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_sha(manifest: Path) -> str:
    """The `safetensors_sha256` the manifest pins, or a fatal error."""
    payload = json.loads(manifest.read_text())
    claimed = payload.get("safetensors_sha256")
    if not isinstance(claimed, str) or not claimed:
        raise SystemExit(f"{manifest}: no safetensors_sha256 to verify against")
    return claimed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="report whether the copy is current; write nothing",
    )
    args = parser.parse_args(argv)

    for name in MEMBERS:
        if not (SRC_DIR / name).is_file():
            raise SystemExit(f"missing {SRC_DIR / name}: convert joe-rs's artifact first")

    # The source has to be self-consistent before it is worth copying. A
    # mismatch here means joe-rs itself is mid-re-export — a new manifest
    # beside old weights, or the reverse — and copying either half forward
    # would spread that instead of reporting it.
    claimed = manifest_sha(SRC_DIR / "manifest.json")
    source_sha = sha256(SRC_DIR / "model.safetensors")
    if source_sha != claimed:
        raise SystemExit(
            "joe-rs's artifact disagrees with its own manifest "
            f"(manifest {claimed}, file {source_sha}); re-run the joe-rs converter"
        )

    current = (
        sha256(DST_DIR / "model.safetensors")
        if (DST_DIR / "model.safetensors").is_file()
        else None
    )
    if current == claimed and (DST_DIR / "manifest.json").is_file():
        if manifest_sha(DST_DIR / "manifest.json") == claimed:
            print(f"unclejoe artifact already current at {claimed[:12]}")
            return 0

    if args.check:
        print(
            f"unclejoe artifact is STALE: joe-rs is at {claimed[:12]}, "
            f"unclejoe at {(current or 'nothing')[:12]}"
        )
        return 1

    DST_DIR.mkdir(parents=True, exist_ok=True)
    staged = DST_DIR / "model.safetensors.part"
    shutil.copyfile(SRC_DIR / "model.safetensors", staged)
    written = sha256(staged)
    if written != claimed:
        staged.unlink(missing_ok=True)
        raise SystemExit(f"copied weights hash {written}, manifest claims {claimed}")
    staged.replace(DST_DIR / "model.safetensors")
    # The manifest lands after the weights it pins: the failure this ordering
    # leaves behind is a manifest older than the file, which `--check` reports;
    # the other order leaves a manifest claiming bytes that are not there.
    shutil.copyfile(SRC_DIR / "manifest.json", DST_DIR / "manifest.json")
    print(f"synced unclejoe artifact from joe-rs at {claimed[:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
