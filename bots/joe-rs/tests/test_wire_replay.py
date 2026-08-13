"""End-to-end decision parity: the real binary, in wire mode, over whole
recorded games (port-plan §6 tier 3, sequence-level; §8 J4).

Each corpus game's `.in.log` is exactly what the engine sent Python joe;
piping it to `joe-rs` in wire mode exercises the shipped path — stdio parse,
state accumulation under the bot's *own* state, forward, greedy decode, and
the pass clamp — and the stdout must match Python joe's recorded replies
turn for turn. A single flipped action would desynchronize nothing here (the
frames are fixed), so every turn stays comparable.

Marker `joe` (needs the release binary and recorded games).
"""
from __future__ import annotations

import subprocess

import pytest

from parity_lib import BINARY, BOT_DIR, corpus_games

pytestmark = pytest.mark.joe


def test_wire_replay_matches_recorded_replies():
    if not BINARY.is_file():
        pytest.skip(f"no release binary at {BINARY}")
    games = [p for p in corpus_games()
             if (p.parent / (p.name.removesuffix(".npz") + ".out.log")).exists()]
    if not games:
        pytest.skip("no recorded games with reply logs")

    total_turns = 0
    for npz_path in games:
        stem = npz_path.name.removesuffix(".npz")
        in_log = (npz_path.parent / f"{stem}.in.log").read_bytes()
        want = (npz_path.parent / f"{stem}.out.log").read_text().splitlines()

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
            assert g.split() == w.split(), (
                f"{stem} turn {t}: joe-rs replied {g!r}, Python joe replied {w!r}")
        total_turns += len(want)
    print(f"\n[wire-replay] {len(games)} games, {total_turns} turns, all replies equal")
