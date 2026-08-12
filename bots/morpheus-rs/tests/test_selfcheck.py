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

`morpheus-rs selfcheck` is the one place that refuses, and `build.sh` runs it at
intake so a broken submission is rejected instead of rated. These tests are here
because a checker nobody checks is the thing it was written to prevent.

Marked `morpheus`, so it is a gate rather than part of the default suite:
`pytest -m morpheus bots/morpheus-rs`. Each case copies the release tree and
runs the binary, which costs 1.4 s and needs a build the default suite cannot
assume (AGENTS.md tester §4).
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.morpheus

BOT_DIR = Path(__file__).resolve().parents[1]
BINARY = BOT_DIR / "target" / "release" / "morpheus-rs"


def _run(root: Path) -> subprocess.CompletedProcess:
    """
    Run the copy of the binary that lives under `root`, not the repo's.

    The artifact and the config are resolved from the **binary's** own
    directory before the working directory is consulted, which is what makes
    the static variant's flat layout work. Invoking the repo binary from
    elsewhere would therefore keep finding the repo's files and quietly test
    nothing.
    """
    return subprocess.run(
        [str(root / "target" / "release" / "morpheus-rs"), "selfcheck"],
        cwd=str(root),
        capture_output=True,
        text=True,
    )


@pytest.fixture(scope="module")
def binary() -> Path:
    if not BINARY.is_file():
        pytest.skip(
            f"no release binary at {BINARY}; build with "
            f"`cargo build --release --manifest-path {BOT_DIR}/Cargo.toml`"
        )
    return BINARY


def test_a_healthy_bot_passes_and_reports_what_it_loaded(binary):
    result = _run(BOT_DIR)
    assert result.returncode == 0, result.stderr
    facts = dict(
        line.split(" ", 1) for line in result.stdout.splitlines() if " " in line
    )
    assert facts["selfcheck"] == "ok"
    # The flag whose absence costs 49x per forward, reported rather than assumed.
    assert facts["hardware_fma"] == "true"
    assert facts["weights_sha256"] == (
        "15509ff5cc1aa8bca16062363a9cd65cc4187a666eb71fc8dc200d8180abb0c9"
    )
    # A decision, not a skip: the position has a general on thirteen army with
    # four empty neighbours, and `1 ...` is what a bot that failed to start says.
    assert facts["decision"].split()[0] == "0"


def test_a_bot_without_its_weights_fails_instead_of_passing_every_turn(
    binary, tmp_path
):
    """
    The exact failure the old smoke could not see. Note what is *not* asserted:
    that the bot exits or misbehaves. It plays on, legally, forever — which is
    why this has to be caught here and not in a match.
    """
    shutil.copytree(BOT_DIR / "artifact", tmp_path / "artifact")
    (tmp_path / "target" / "release").mkdir(parents=True)
    shutil.copy(BINARY, tmp_path / "target" / "release" / "morpheus-rs")
    shutil.copy(BOT_DIR / "deployment.json", tmp_path / "deployment.json")

    assert _run(tmp_path).returncode == 0, "the copied layout should be healthy"

    shutil.rmtree(tmp_path / "artifact")
    broken = _run(tmp_path)
    assert broken.returncode != 0
    assert "selfcheck FAILED" in broken.stdout
    assert "artifact" in broken.stderr


def test_a_bot_without_its_knobs_fails_rather_than_playing_placeholders(
    binary, tmp_path
):
    """
    Nastier than the missing artifact, and quieter. `try_load_deployment` falls
    back to the Part 07 placeholders — four times the particles, twice the
    search depth, a 125 ms deadline against 140 — so the bot plays well, plays
    within the protocol, and is not the program that was measured.
    """
    shutil.copytree(BOT_DIR / "artifact", tmp_path / "artifact")
    (tmp_path / "target" / "release").mkdir(parents=True)
    shutil.copy(BINARY, tmp_path / "target" / "release" / "morpheus-rs")

    broken = _run(tmp_path)
    assert broken.returncode != 0
    assert "placeholder knobs" in broken.stderr
