"""
Does the intake check still notice a bot that cannot play?

M8 found the packaging path scoring a bundle with `artifact/` deleted as "two
well-formed actions", byte-identical to a healthy one. That is not a bug in the
smoke test so much as the shape of the bot: `Seat::new` returning `Err`
degrades the seat to passing every turn rather than exiting, because the judge
forfeits a game on an early exit and charges one fault out of fifty for a bad
reply (RULES.md §08). Every way the submission can be broken — missing weights,
an unreadable `deployment.json`, a build with no hardware FMA — therefore
produces a bot that answers the protocol perfectly and loses every game.

`morpheus-joe selfcheck` is the one place that refuses, and `build.sh` runs it
at intake so a broken submission is rejected instead of rated. These tests are
here because a checker nobody checks is the thing it was written to prevent.

**Two of them are N1's own gate.** The phase asks for proof that `Net::load`
accepts the synced artifact and refuses a mismatched one, and the two failures
worth proving are the two the fork newly exposes: a **stale or interrupted
copy** from `scripts/joe_artifact_fanout.py`, which shows up as a digest that
disagrees with the manifest beside it, and a **schema mismatch**, which is what
a manifest from a differently-shaped checkpoint looks like. Neither is
hypothetical — the weights arrive here by a file copy, and joe's net is one
re-export away from a different `depth` or `embed_dim`.

Marked `morpheus`, so it is a gate rather than part of the default suite:
`pytest -m morpheus bots/morpheus-joe`. Each case copies the release tree and
runs the binary, which needs a build the default suite cannot assume
(AGENTS.md, "Test suite budget"). The copies are 34 MB rather than 1 MB now, so
the artifact is **hard-linked** where a case does not modify it.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.morpheus

BOT_DIR = Path(__file__).resolve().parents[1]
BINARY = BOT_DIR / "target" / "release" / "morpheus-joe"

# joe-rs's artifact, one hop up the chain this bot's weights come down.
# `scripts/joe_artifact_fanout.py` is the only thing that moves them.
JOE_RS_ARTIFACT = BOT_DIR.parent / "joe-rs" / "artifact"


def _run(root: Path) -> subprocess.CompletedProcess:
    """
    Run the copy of the binary that lives under `root`, not the repo's.

    The artifact and the config are resolved from the **binary's** own
    directory before the working directory is consulted, which is what makes
    the submission zip's flat layout work. Invoking the repo binary from
    elsewhere would therefore keep finding the repo's files and quietly test
    nothing.
    """
    return subprocess.run(
        [str(root / "target" / "release" / "morpheus-joe"), "selfcheck"],
        cwd=str(root),
        capture_output=True,
        text=True,
    )


def _layout(root: Path, *, link_artifact: bool = True) -> None:
    """A minimal healthy bot directory: binary, config, artifact."""
    (root / "target" / "release").mkdir(parents=True)
    shutil.copy(BINARY, root / "target" / "release" / "morpheus-joe")
    shutil.copy(BOT_DIR / "deployment.json", root / "deployment.json")
    (root / "artifact").mkdir()
    shutil.copy(BOT_DIR / "artifact" / "manifest.json", root / "artifact" / "manifest.json")
    weights = BOT_DIR / "artifact" / "model.safetensors"
    target = root / "artifact" / "model.safetensors"
    if link_artifact:
        # 34 MB per case, four cases. Hard-linking is safe because no case
        # below writes to the weights — the two that break an artifact break
        # the *manifest* beside it, or replace the weights with a new file.
        os.link(weights, target)
    else:
        shutil.copy(weights, target)


@pytest.fixture(scope="module")
def binary() -> Path:
    if not BINARY.is_file():
        pytest.skip(
            f"no release binary at {BINARY}; build with "
            f"`cargo build --release --manifest-path {BOT_DIR}/Cargo.toml`"
        )
    return BINARY


def _facts(result: subprocess.CompletedProcess) -> dict[str, str]:
    return dict(line.split(" ", 1) for line in result.stdout.splitlines() if " " in line)


def test_a_healthy_bot_passes_and_reports_what_it_loaded(binary):
    result = _run(BOT_DIR)
    assert result.returncode == 0, result.stderr
    facts = _facts(result)
    assert facts["selfcheck"] == "ok"
    # The flag whose absence costs 49x per forward, reported rather than assumed.
    assert facts["hardware_fma"] == "true"
    assert facts["tensor_schema"] == "joe-net-v1"
    # Not a literal: this bot's weights are joe-rs's, and pinning the digest
    # here would mean editing two files on every re-export instead of one.
    # `tests/test_joe_source_fanout.py` is what makes them agree; this asserts
    # the *bot* reports the same checkpoint the chain says it should.
    claimed = json.loads((JOE_RS_ARTIFACT / "manifest.json").read_text())
    assert facts["safetensors_sha256"] == claimed["safetensors_sha256"]
    assert facts["checkpoint"] == claimed["checkpoint"]["run_name"]
    # A decision, not a skip: the position has a general on thirteen army with
    # four empty neighbours, and `1 ...` is what a bot that failed to start
    # says. It holds at N1 with a flat prior, because the tactics layer and the
    # hard rules still run.
    assert facts["decision"].split()[0] == "0"


def test_a_bot_without_its_weights_fails_instead_of_passing_every_turn(
    binary, tmp_path
):
    """
    The exact failure the old smoke could not see. Note what is *not* asserted:
    that the bot exits or misbehaves. It plays on, legally, forever — which is
    why this has to be caught here and not in a match.
    """
    _layout(tmp_path)
    assert _run(tmp_path).returncode == 0, "the copied layout should be healthy"

    shutil.rmtree(tmp_path / "artifact")
    broken = _run(tmp_path)
    assert broken.returncode != 0
    assert "selfcheck FAILED" in broken.stdout
    assert "artifact" in broken.stderr


def test_an_interrupted_fan_out_is_caught_by_the_digest(binary, tmp_path):
    """
    N1's gate, half one: a manifest that does not describe the bytes beside it.

    This is the failure the fan-out can actually produce. The sync writes the
    weights through a temporary file so a killed copy cannot leave a truncated
    `model.safetensors`, but nothing stops a hand-edited or half-restored
    artifact directory, and `Net::load` alone would not notice: it checks
    shapes and names, and a *different* valid checkpoint has the same ones.
    """
    _layout(tmp_path, link_artifact=False)
    assert _run(tmp_path).returncode == 0

    manifest = tmp_path / "artifact" / "manifest.json"
    payload = json.loads(manifest.read_text())
    payload["safetensors_sha256"] = "0" * 64
    manifest.write_text(json.dumps(payload))

    broken = _run(tmp_path)
    assert broken.returncode != 0
    assert "selfcheck FAILED" in broken.stdout
    assert "SHA-256" in broken.stderr
    # The message has to say what to do about it, because the person reading it
    # is looking at a bot that would otherwise have played.
    assert "joe_artifact_fanout" in broken.stderr


def test_a_manifest_from_a_differently_shaped_net_is_refused(binary, tmp_path):
    """
    N1's gate, half two: the schema check, reached through the real loader.

    Changing `depth` is the cheapest way to describe a checkpoint this binary
    cannot run. The shapes are compiled in — five blocks, 384 wide, eight heads
    — so a six-block checkpoint is not a slower bot, it is a bot reading the
    wrong tensors, and it must not start.
    """
    _layout(tmp_path, link_artifact=False)
    manifest = tmp_path / "artifact" / "manifest.json"
    payload = json.loads(manifest.read_text())
    payload["network"]["depth"] = 6
    manifest.write_text(json.dumps(payload))
    # The digest still matches: only the manifest's claim about the *shape*
    # changed, which is exactly the case the digest cannot catch.
    payload["safetensors_sha256"] = json.loads(
        (BOT_DIR / "artifact" / "manifest.json").read_text()
    )["safetensors_sha256"]
    manifest.write_text(json.dumps(payload))

    broken = _run(tmp_path)
    assert broken.returncode != 0
    assert "selfcheck FAILED" in broken.stdout
    assert "depth" in broken.stderr


def test_a_bot_without_its_knobs_fails_rather_than_playing_placeholders(
    binary, tmp_path
):
    """
    Nastier than the missing artifact, and quieter. `try_load_deployment` falls
    back to the Part 07 placeholders — four times the particles, twice the
    search depth, a 125 ms deadline against 140 — so the bot plays well, plays
    within the protocol, and is not the program that was measured.
    """
    _layout(tmp_path)
    (tmp_path / "deployment.json").unlink()

    broken = _run(tmp_path)
    assert broken.returncode != 0
    assert "placeholder knobs" in broken.stderr
