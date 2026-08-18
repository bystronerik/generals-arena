"""
Joe's net fans out to another bot. Does the copy still say what joe-rs says?

Only joe-rs converts joe's `.eqx` checkpoint. Everything downstream is a copy,
and a copy that silently falls behind is the failure mode this file exists for.
A seat carrying last month's weights loads, plays, and looks perfectly healthy.
Nothing at play time can tell.

**The weights** (`joe-net-plan.md` §8.7). The measurement is what forces this:
the joe-rs / unclejoe contrast is supposed to isolate the tactics layer, and it
isolates nothing unless both arms run **byte-identical** weights rather than
weights from the same checkpoint. The repo has been caught out here twice,
which is why the check is a test and not a line in a checklist —
`scripts/joe_artifact_fanout.py` is how you fix a red result, not how you
notice one.

This file once also checked source identity, for `morpheus-joe`: that fork
carried a byte-identical copy of `bots/joe-rs/src/`, and the copy *was* its
parity argument for the forward pass. The fork was removed on 2026-08-18, so
the check has no subject. `unclejoe` carries the same copies and was
deliberately never checked for source identity — it is under active
development, and gating someone else's working tree on a fork's argument would
be the wrong coupling. Its weights are checked, because those are shared.

Cheap by construction: a few `sha256` calls and two small JSON reads, well
inside the 15 s suite budget (AGENTS.md).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
JOE_RS = REPO / "bots" / "joe-rs"

# Every bot carrying a copy of joe-rs's weights.
WEIGHT_FORKS = ("unclejoe",)


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


@pytest.mark.parametrize("bot", WEIGHT_FORKS)
def test_every_downstream_bot_is_on_joe_rs_weights(bot):
    """
    §8.7. The staleness that has already happened twice, as a red suite instead
    of a silent regression in a rating round.
    """
    upstream = manifest_sha("joe-rs")
    assert upstream, "bots/joe-rs/artifact/manifest.json has no safetensors_sha256"
    downstream = manifest_sha(bot)
    assert downstream, f"bots/{bot}/artifact/manifest.json is missing or has no digest"
    assert downstream == upstream, (
        f"{bot} is on {downstream[:12]} but joe-rs is on {upstream[:12]}. Run "
        "`python scripts/joe_artifact_fanout.py`. If this fires part-way through "
        "a measurement round, the round is void: a sync forks the bot's content "
        "hash, and a contrast that spans one compares two different programs."
    )


@pytest.mark.parametrize("bot", ("joe-rs",) + WEIGHT_FORKS)
def test_each_manifest_describes_the_weights_beside_it(bot):
    """
    The half of a copy that a digest comparison between manifests cannot see.

    Comparing manifests proves the two files agree about which checkpoint they
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
        "manifest.json claims. Re-run the copy that produced it: "
        "`python scripts/joe_artifact_fanout.py` downstream, or joe-rs's "
        "converter at the top of the chain."
    )
