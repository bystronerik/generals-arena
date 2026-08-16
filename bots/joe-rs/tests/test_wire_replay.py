"""End-to-end decision parity: the real binary, in wire mode, over whole
recorded games (port-plan §6 tier 3, sequence-level; §8 J4).

Each corpus game's `.in.log` is exactly what the engine sent Python joe;
piping it to `joe-rs` in wire mode exercises the shipped path — stdio parse,
state accumulation under the bot's *own* state, forward, greedy decode, and
the pass clamp — and the stdout must match the JAX oracle turn for turn. A
single flipped action would desynchronize nothing here (the frames are
fixed), so every turn stays comparable.

The reference is the `.npz`'s `all_action`, **not** the `.out.log`. Since
2026-08-16 Python joe applies a repetition penalty before its argmax
(`bots/joe/agent.py`), which is a research layer the port deliberately does
not carry: joe-rs is a port of joe's *network*, not of joe's selection
layer. The `.out.log` therefore records a different program, and grading
joe-rs against it would fail the binary for a difference it is meant to
have. `all_action` is the same oracle, unpenalised, for every turn — so this
stays a full-strength end-to-end check. `capture_fixtures.py` still asserts
the `.out.log` against a penalised recomputation, which is what keeps the
recorded games honest about the deployed path.

Marker `joe` (needs the release binary and recorded games).
"""
from __future__ import annotations

import subprocess

import numpy as np
import pytest

from parity_lib import BINARY, BOT_DIR, corpus_games

pytestmark = pytest.mark.joe


def _oracle_replies(npz_path) -> list[list[str]]:
    """The unpenalised oracle action per turn, after agent.py's pass clamp."""
    with np.load(npz_path) as z:
        actions = z["all_action"]
    replies = []
    for row in actions:
        p, r, c, d, s = (int(x) for x in row)
        replies.append(["1", "0", "0", "0", "0"] if p == 1
                       else [str(p), str(r), str(c), str(d), str(s)])
    return replies


def test_wire_replay_matches_recorded_replies():
    if not BINARY.is_file():
        pytest.skip(f"no release binary at {BINARY}")
    # Real engine games only: the synthetic stitched stream has no `.in.log`
    # provenance from a live seat, and the surface tiers already cover it.
    games = [p for p in corpus_games()
             if (p.parent / (p.name.removesuffix(".npz") + ".out.log")).exists()]
    if not games:
        pytest.skip("no recorded games with reply logs")

    total_turns = 0
    for npz_path in games:
        stem = npz_path.name.removesuffix(".npz")
        in_log = (npz_path.parent / f"{stem}.in.log").read_bytes()
        want = _oracle_replies(npz_path)

        proc = subprocess.run(
            [str(BINARY)], input=in_log, capture_output=True,
            env={"JOE_RS_ARTIFACT": str(BOT_DIR / "artifact"),
                 "PATH": "/usr/bin:/bin"})
        assert proc.returncode == 0, proc.stderr.decode()[-2000:]
        got = proc.stdout.decode().splitlines()
        # The engine may have closed the seat before answering the final
        # frame when the game ended; joe-rs answers every frame it was fed.
        assert len(got) >= len(want), f"{stem}: {len(got)} replies vs {len(want)} recorded"
        for t, (g, w) in enumerate(zip(got, want)):
            assert g.split() == w, (
                f"{stem} turn {t}: joe-rs replied {g!r}, JAX oracle gave {' '.join(w)!r}")
        total_turns += len(want)
    print(f"\n[wire-replay] {len(games)} games, {total_turns} turns, all replies equal")
