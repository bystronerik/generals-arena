"""
At N2 this bot is joe wearing morpheus's I/O. Does it still reply as joe?

The second half of N2's gate, and the one that admits no interpretation. Each
corpus game's `.in.log` is exactly what the engine sent Python joe; piping it
to `morpheus-joe` in wire mode runs the shipped path — morpheus's stdio parse,
the observation bridge, joe's forward, joe's greedy decode, joe's pass clamp —
and stdout must match Python joe's recorded replies turn for turn.

Why this is worth 20 seconds a run: at N2 the fork has exactly one job, to feed
joe's network what joe's network expects, and every remaining difference
between the two bots is I/O. So a divergence here is a bridge bug, full stop —
there is no search, no shaping and no belief to argue about yet. Localizing one
after N3 lands would mean separating a bad tensor from a bad remap from a bad
prior, which is three phases of ambiguity this check buys off for one.

It overlaps `test_morpheus_joe_bridge.py` deliberately and does not replace it.
That file compares the tensor and would catch a wrong plane the argmax happens
to be insensitive to; this one compares the decision and would catch a fault
downstream of the tensor — the seat asking the net with the wrong penalties, a
lost pass clamp, a reply field widened wrong. Neither implies the other.

**This bot stops replying like joe at N3, by design.** The search then decides,
the mask becomes morpheus's `play_mask` rather than joe's, and the replies
diverge for good reasons. This file is scoped to the phase where equality is
the specification, and N3 retires it rather than relaxing it.

Marked `morpheus`: it needs a release binary and the derived corpus (AGENTS.md,
"Test suite budget").
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.morpheus

BOT_DIR = Path(__file__).resolve().parents[1]
REPO = BOT_DIR.parents[1]
BINARY = Path(
    os.environ.get("MORPHEUS_JOE_BINARY", BOT_DIR / "target" / "release" / "morpheus-joe")
)
CORPUS = REPO / "data" / "joe" / "joe-rs-parity" / "games"
SMOKE = REPO / "bots" / "joe-rs" / "tests" / "fixtures"

# The plan asks for "at least one full recorded game". Three, because one map
# exercises one board shape and one opening: a widening bug that only bites on
# a non-square board, or a history bug that needs a long game, would sit inside
# a single-game pass. ~7 s each on one core, which is what a forward costs
# times a game's length and not something a code change makes cheaper.
GAMES_WANTED = 3


def games() -> list[tuple[Path, Path]]:
    """`(input log, recorded replies)` pairs, corpus first, else the slice."""
    for directory in (CORPUS, SMOKE):
        if not directory.is_dir():
            continue
        found = []
        for in_log in sorted(directory.glob("*.in.log")):
            out_log = in_log.parent / f"{in_log.name.removesuffix('.in.log')}.out.log"
            if out_log.is_file():
                found.append((in_log, out_log))
        if found:
            return found[:GAMES_WANTED]
    return []


def test_every_reply_is_the_reply_joe_recorded():
    if not BINARY.is_file():
        pytest.skip(
            f"no release binary at {BINARY}; build with "
            f"`cargo build --release --manifest-path {BOT_DIR}/Cargo.toml`")
    pairs = games()
    if not pairs:
        pytest.skip(f"no recorded games with reply logs under {CORPUS} or {SMOKE}")

    total = 0
    for in_log, out_log in pairs:
        want = out_log.read_text().splitlines()
        proc = subprocess.run(
            [str(BINARY)],
            input=in_log.read_bytes(),
            capture_output=True,
            cwd=str(BOT_DIR),
        )
        assert proc.returncode == 0, proc.stderr.decode()[-2000:]
        got = proc.stdout.decode().splitlines()
        # The engine may close the seat before it answers the last frame it
        # sent; this bot answers every frame it is fed, so it may reply once
        # more than the log records. Fewer replies is a real failure.
        assert len(got) >= len(want), (
            f"{in_log.name}: {len(got)} replies against {len(want)} recorded")
        for turn, (mine, joes) in enumerate(zip(got, want)):
            assert mine.split() == joes.split(), (
                f"{in_log.name} turn {turn}: morpheus-joe replied {mine!r}, "
                f"joe replied {joes!r}. At N2 that is a bridge bug — compare "
                f"the augmented tensor at this turn with the `sequence` "
                f"surface first.")
        total += len(want)
    print(f"\n[wire-replay] {len(pairs)} games, {total} turns, every reply equal")
