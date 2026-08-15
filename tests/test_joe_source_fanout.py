"""
Joe's net fans out to two other bots. Do the copies still say what joe-rs says?

Only joe-rs converts joe's `.eqx` checkpoint. Everything downstream — the
weights and, for `morpheus-joe`, the source files that read them — is a copy,
and a copy that silently falls behind is the failure mode this file exists for.
A seat carrying last month's weights, or a "fixed" copy of `gemm.rs`, loads,
plays, and looks perfectly healthy. Nothing at play time can tell.

Two things are checked, for two different reasons.

**The weights** (`joe-net-plan.md` §8.7). The measurement is what forces this:
the joe-rs / unclejoe contrast is supposed to isolate the tactics layer, and
the joe-rs / morpheus-joe contrast the search stack. Neither isolates anything
unless both arms run **byte-identical** weights rather than weights from the
same checkpoint. The repo has been caught out here twice, which is why the
check is a test and not a line in a checklist —
`scripts/joe_artifact_fanout.py` is how you fix a red result, not how you
notice one.

**The source** (§8.2). `bots/morpheus-joe/crates/joenet/src/` is a
byte-identical copy of `bots/joe-rs/src/`, and that identity *is* the fork's
parity argument for its forward pass: joe-rs already carries a JAX-oracle
corpus proving its forward matches joe to pinned relative bounds (2.331e-6
logit, 2.244e-6 bin; 731/731 frames of decision parity; 4,760 turns of
byte-equal wire replay), and rebuilding that corpus for a third bot buys
nothing. Equal files plus a green joe-rs corpus means the fork's forward pass
is joe's forward pass. Unequal files mean the argument is void, and this test
names the file that broke it.

The consequence is a rule rather than a preference: **the fork may not
"improve" those files.** A change lands in joe-rs first and is re-copied down.
Batching the leaf forwards (§5.2) is the concrete case, and it is scoped as
upstream work for exactly this reason.

`unclejoe` carries the same copies and is deliberately **not** checked for
source identity here — it is under active development, and gating someone
else's working tree on this fork's argument would be the wrong coupling. Its
weights are checked, because those are shared.

Cheap by construction: eleven `sha256` calls over ~1,900 lines and four small
JSON reads, well inside the 15 s suite budget (AGENTS.md).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
JOE_RS = REPO / "bots" / "joe-rs"

# Every bot carrying a copy of joe-rs's weights.
WEIGHT_FORKS = ("unclejoe", "morpheus-joe")

# The files `bots/morpheus-joe/crates/joenet/src/` copies from
# `bots/joe-rs/src/`. The plan names five — `nn/{net,gemm,safetensors}.rs`,
# `board/obs.rs`, `xla_math.rs`; the fork copies the modules around them too,
# so the whole subtree is one copy and not five files inside a rewrite.
JOENET_SOURCES = (
    "xla_math.rs",
    "io/mod.rs",
    "io/json.rs",
    "io/wire.rs",
    "board/mod.rs",
    "board/action.rs",
    "board/obs.rs",
    "nn/mod.rs",
    "nn/gemm.rs",
    "nn/net.rs",
    "nn/safetensors.rs",
)


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


@pytest.mark.parametrize("name", JOENET_SOURCES)
def test_the_fork_carries_joes_source_byte_for_byte(name):
    """
    §8.2. A failure here does not mean "a file drifted"; it means the fork's
    forward pass is no longer covered by any corpus.
    """
    upstream = JOE_RS / "src" / name
    copy = REPO / "bots" / "morpheus-joe" / "crates" / "joenet" / "src" / name
    assert upstream.is_file(), f"{upstream} is missing; joe-rs is the source of truth"
    assert copy.is_file(), (
        f"{copy} is missing. `crates/joenet/src/` is a copy of joe-rs's tree; "
        "re-copy the file rather than writing one."
    )
    assert sha256(copy) == sha256(upstream), (
        f"{name} differs between bots/joe-rs/src/ and the fork's copy. The fork's "
        "forward pass is proved *transitively* — by being byte-identical to the "
        "one joe-rs's JAX-oracle corpus covers — so this failure voids that "
        "proof. Land the change in joe-rs, re-run its parity gate "
        "(`pytest -m joe bots/joe-rs`), then re-copy down."
    )


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
