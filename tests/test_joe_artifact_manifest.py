"""
Does each converted artifact describe the bytes beside it?

Two bots convert an `.eqx` checkpoint to safetensors: joe-rs (from joe's
artifact) and unclejoe (from its own X16 export). A manifest that landed
before the weights it pins, or beside a truncated file, is the failure mode
this file exists for: the bot loads, plays, and looks perfectly healthy, and
nothing at play time can tell.

This file once also compared **downstream copies** against joe-rs. Two forks
carried a byte copy of joe-rs's weights, and a copy that silently fell behind
voided any rating contrast that spanned the sync (`joe-net-plan.md` §8.7).
`morpheus-joe` was removed on 2026-08-18 and the weight-copying `unclejoe`
on 2026-08-20, so that chain is one bot long and the comparison has no
subject; `scripts/joe_artifact_fanout.py`, which copied the weights forward,
went with them, and a new *weight-copying* fork restores both. The 2026-08-25
`unclejoe` is not that fork: it owns its weights lineage (the X16 run), so
nothing copies weights to it and only the manifest-vs-bytes half applies.

Cheap by construction: one `sha256` call and one small JSON read per bot,
well inside the 15 s suite budget (AGENTS.md).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_sha(bot: str) -> str | None:
    manifest = REPO / "bots" / bot / "artifact" / "manifest.json"
    if not manifest.is_file():
        return None
    return json.loads(manifest.read_text()).get("safetensors_sha256")


@pytest.mark.parametrize("bot", ("joe-rs", "unclejoe"))
def test_each_manifest_describes_the_weights_beside_it(bot):
    """
    The half of a copy that a digest comparison between manifests cannot see.

    Comparing manifests proves two files agree about which checkpoint they
    name. It does not prove either one is describing the bytes on its own disk
    — a manifest that landed before its weights, or beside a truncated file,
    passes that comparison and fails this one.
    """
    art = REPO / "bots" / bot / "artifact"
    weights = art / "model.safetensors"
    if not weights.is_file():
        pytest.skip(f"no weights at {weights}")
    assert sha256(weights) == manifest_sha(bot), (
        f"bots/{bot}/artifact/model.safetensors does not hash to what its own "
        f"manifest.json claims. Re-run the converter that produced it: "
        f"`bots/{bot}/tools/convert_artifact.py`, after the same bot's "
        f"quantize step (joe-rs quantizes via `bots/joe/tools/"
        f"quantize_artifact.py`; unclejoe carries its own)."
    )
