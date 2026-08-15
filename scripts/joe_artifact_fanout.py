#!/usr/bin/env python3
"""Copy joe-rs's artifact into every downstream bot, and refuse a partial copy.

    python scripts/joe_artifact_fanout.py --check      # report, write nothing
    python scripts/joe_artifact_fanout.py              # sync every downstream
    python scripts/joe_artifact_fanout.py --bot morpheus-joe

The weights fan out along `joe -> joe-rs -> {unclejoe, morpheus-joe}`. Only
joe-rs converts; the forks **track joe-rs**, one hop down, and never read joe's
`.eqx`. The reason is the measurement, not tidiness: the joe-rs / unclejoe
contrast is supposed to isolate the tactics layer and the joe-rs /
morpheus-joe contrast the search stack, and either one only isolates anything
if both arms run byte-identical weights rather than weights from the same
checkpoint.

Nothing detects a skipped sync at play time — a seat with older weights loads,
plays, and looks healthy — so this script's job is to make the *failure* of
that step loud. It verifies the digest each manifest claims against the bytes
on both sides, and writes the weights through a temporary file so an
interrupted copy cannot leave a half-written `model.safetensors` beside a
manifest that swears it is complete.

`--check` reports without writing and exits non-zero when any downstream copy
is stale. `tests/test_joe_source_fanout.py` runs the same comparison as part of
the default suite, which is the part that actually fires: the checklist in
`docs/bots/joe-rs/export.md` is discipline, and discipline is what failed here
twice.

A sync forks the downstream bot's content hash, exactly as a re-conversion
forks joe-rs's. That is intended, the version registry records it, and it
**invalidates any rating contrast that spans the sync** — so freeze the
weights for the duration of a measurement round (joe-net-plan §8.7, §9).

`bots/unclejoe/tools/sync_artifact.py` still syncs unclejoe alone and is
unchanged; this script is the one that walks every downstream at once.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "bots" / "joe-rs" / "artifact"

# Every bot that carries a copy of joe-rs's weights. Adding a fork here and to
# `tests/test_joe_source_fanout.py` is the whole registration step.
DOWNSTREAM = ("unclejoe", "morpheus-joe")

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


def source_digest() -> str:
    """joe-rs's current weights digest, after checking it against its manifest.

    A mismatch here means joe-rs itself is mid-re-export — a new manifest
    beside old weights, or the reverse — and copying either half forward would
    spread that instead of reporting it.
    """
    for name in MEMBERS:
        if not (SOURCE / name).is_file():
            raise SystemExit(f"missing {SOURCE / name}: convert joe-rs's artifact first")
    claimed = manifest_sha(SOURCE / "manifest.json")
    actual = sha256(SOURCE / "model.safetensors")
    if actual != claimed:
        raise SystemExit(
            "joe-rs's artifact disagrees with its own manifest "
            f"(manifest {claimed}, file {actual}); re-run the joe-rs converter"
        )
    return claimed


def downstream_digest(bot: str) -> str | None:
    """What `bot` is actually carrying, or `None` if it carries nothing usable.

    Both halves have to agree before the copy counts as current. A manifest
    that pins bytes other than the ones beside it is exactly the half-written
    state the staged write below exists to prevent, and it reads as stale.
    """
    art = REPO / "bots" / bot / "artifact"
    weights, manifest = art / "model.safetensors", art / "manifest.json"
    if not weights.is_file() or not manifest.is_file():
        return None
    actual = sha256(weights)
    return actual if manifest_sha(manifest) == actual else None


def sync(bot: str, claimed: str) -> None:
    art = REPO / "bots" / bot / "artifact"
    art.mkdir(parents=True, exist_ok=True)
    staged = art / "model.safetensors.part"
    shutil.copyfile(SOURCE / "model.safetensors", staged)
    written = sha256(staged)
    if written != claimed:
        staged.unlink(missing_ok=True)
        raise SystemExit(f"{bot}: copied weights hash {written}, manifest claims {claimed}")
    staged.replace(art / "model.safetensors")
    # The manifest lands after the weights it pins: the failure this ordering
    # leaves behind is a manifest older than the file, which `--check` reports;
    # the other order leaves a manifest claiming bytes that are not there.
    shutil.copyfile(SOURCE / "manifest.json", art / "manifest.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="report which downstream copies are stale; write nothing",
    )
    parser.add_argument(
        "--bot",
        choices=DOWNSTREAM,
        help="one downstream bot instead of all of them",
    )
    args = parser.parse_args(argv)

    claimed = source_digest()
    targets = (args.bot,) if args.bot else DOWNSTREAM
    print(f"joe-rs is at {claimed[:12]}")

    stale = []
    for bot in targets:
        current = downstream_digest(bot)
        if current == claimed:
            print(f"  {bot}: current")
            continue
        stale.append(bot)
        print(f"  {bot}: STALE at {(current or 'nothing')[:12]}")

    if not stale:
        return 0
    if args.check:
        return 1
    for bot in stale:
        sync(bot, claimed)
        print(f"synced {bot} from joe-rs at {claimed[:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
