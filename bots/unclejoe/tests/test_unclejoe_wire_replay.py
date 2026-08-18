"""End-to-end decision equality: the real binary, in wire mode, over whole
recorded games.

This is the fork's one sequence-level guard. joe-rs owns the parity harness —
the JAX-oracle surfaces, the tier-2 pins, the mutation kills — and unclejoe
does not duplicate it: the copied `board/`, `nn/`, `io/`, and `xla_math`
modules are byte-identical to the ones that harness already proves, so a
second copy would double the work after a joe re-export and learn nothing.
What is *not* proven elsewhere is that this crate still composes those modules
the way joe-rs does, and that is what this test checks.

Each corpus game's `.in.log` is exactly what the engine sent Python joe;
piping it to `unclejoe` in wire mode exercises the shipped path — stdio parse,
state accumulation under the bot's *own* state, forward, greedy decode, and
the pass clamp — and the stdout must match the JAX oracle's unpenalised
action (`all_action` in the capture) turn for turn: see `recorded_games()`.
The frames are fixed, so a flipped action desynchronizes nothing and every
turn stays comparable.

The corpus is joe-rs's, at `data/joe/joe-rs-parity/games`, and is read
**read-only**: producing it stays with joe-rs. A checkout without it falls
back to joe-rs's committed smoke slice, which is one sliced game and enough to
tell a broken composition from a working one.

Marker `unclejoe` (needs the release binary and recorded games).
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

BOT_DIR = Path(__file__).resolve().parents[1]
REPO = BOT_DIR.parents[1]
# UNCLEJOE_BIN aims the test at another build of the same crate.
BINARY = Path(os.environ.get("UNCLEJOE_BIN", BOT_DIR / "target" / "release" / "unclejoe"))
CORPUS = REPO / "data" / "joe" / "joe-rs-parity" / "games"
SMOKE = REPO / "bots" / "joe-rs" / "tests" / "fixtures"

pytestmark = pytest.mark.unclejoe


def recorded_games() -> list[Path]:
    """`.in.log` files with an oracle capture (`.npz`) beside them.

    Full corpus when present, else the committed smoke slice.

    The reference is the capture's `all_action`, **not** the `.out.log` —
    the same rule as `bots/joe-rs/tests/test_wire_replay.py`, for the same
    reason. Since 2026-08-16 Python joe applies a repetition penalty before
    its argmax (`bots/joe/agent.py`); with `UNCLEJOE_TACTICS=0` this test
    asks for the plain network path, which the unpenalised oracle records
    for every turn. The `.out.log` records the penalised program.
    """
    for directory in (CORPUS, SMOKE):
        games = [
            path
            for path in sorted(directory.glob("*.in.log"))
            if replies_for(path).is_file()
        ]
        if games:
            return games
    return []


def replies_for(in_log: Path) -> Path:
    return in_log.parent / (in_log.name.removesuffix(".in.log") + ".npz")


def oracle_replies(npz_path: Path) -> list[str]:
    """The unpenalised oracle action per turn, after agent.py's pass clamp."""
    import numpy as np

    with np.load(npz_path) as z:
        actions = z["all_action"]
    return [
        "1 0 0 0 0" if int(row[0]) == 1
        else " ".join(str(int(x)) for x in row)
        for row in actions
    ]


def test_wire_replay_matches_recorded_replies():
    if not BINARY.is_file():
        pytest.skip(f"no release binary at {BINARY}")
    games = recorded_games()
    if not games:
        pytest.skip("no recorded games with reply logs")

    total_turns = 0
    for in_log in games:
        frames = in_log.read_bytes()
        want = oracle_replies(replies_for(in_log))

        proc = subprocess.run(
            [str(BINARY)], input=frames, capture_output=True,
            # UNCLEJOE_TACTICS=0 is the kill-switch the tactics layer reads.
            # It means nothing to this build and everything to the next one:
            # this test's claim is about the network path, so it must keep
            # asking for the network path once a layer exists that can
            # override it.
            env={"UNCLEJOE_ARTIFACT": str(BOT_DIR / "artifact"),
                 "UNCLEJOE_TACTICS": "0",
                 "PATH": "/usr/bin:/bin"})
        assert proc.returncode == 0, proc.stderr.decode()[-2000:]
        got = proc.stdout.decode().splitlines()
        # The engine may have closed the seat before answering the final
        # frame when the game ended; unclejoe answers every frame it was fed.
        stem = in_log.name.removesuffix(".in.log")
        assert len(got) >= len(want), f"{stem}: {len(got)} replies vs {len(want)} recorded"
        for t, (g, w) in enumerate(zip(got, want)):
            assert g.split() == w.split(), (
                f"{stem} turn {t}: unclejoe replied {g!r}, Python joe replied {w!r}")
        total_turns += len(want)
    print(f"\n[wire-replay] {len(games)} games, {total_turns} turns, all replies equal")
