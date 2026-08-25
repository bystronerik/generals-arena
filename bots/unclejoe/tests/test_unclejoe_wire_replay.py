"""End-to-end decision stability: the real binary, in wire mode, over whole
recorded games, against its own recorded replies (selection-plan S1).

Each corpus game's `.in.log` is exactly what the engine sent unclejoe;
piping it to `unclejoe` in wire mode exercises the shipped path — stdio
parse, state accumulation under the bot's *own* state, forward, argmax
selection (T = 0), and the pass clamp. unclejoe argmaxes, so its replies
should equal the oracle's greedy `all_action` by design — but the reference
here stays the binary's **own** replies, recorded by `capture_fixtures.py
--selection-golden` as `<name>.unclejoe.log`, graded byte-for-byte under
the bot's own accumulated state.

Selection is a deterministic function of the frame stream, so the whole
stdout must reproduce byte for byte, and an unexplained change anywhere in
the played path still fails here. The network itself stays graded against
the `.npz` oracle in `test_unclejoe_parity.py` (tiers 1–3), which this file does not
repeat.

Marker `joe` (needs the release binary and recorded games).
"""
from __future__ import annotations

import subprocess

import pytest

from unclejoe_parity_lib import BINARY, BOT_DIR, corpus_games

pytestmark = pytest.mark.joe


def test_wire_replay_matches_the_self_golden():
    if not BINARY.is_file():
        pytest.skip(f"no release binary at {BINARY}")
    replays = []
    for npz_path in corpus_games():
        stem = npz_path.name.removesuffix(".npz")
        in_log = npz_path.parent / f"{stem}.in.log"
        golden = npz_path.parent / f"{stem}.unclejoe.log"
        if in_log.exists() and golden.exists():
            replays.append((stem, in_log, golden))
    if not replays:
        pytest.skip("no self-goldens; run capture_fixtures.py --selection-golden")

    total_replies = 0
    for stem, in_log, golden in replays:
        proc = subprocess.run(
            [str(BINARY)], input=in_log.read_bytes(), capture_output=True,
            env={"JOE_RS_ARTIFACT": str(BOT_DIR / "artifact"),
                 "PATH": "/usr/bin:/bin"})
        assert proc.returncode == 0, proc.stderr.decode()[-2000:]
        got = proc.stdout.decode()
        want = golden.read_text()
        if got != want:
            # Localize before failing: the first differing turn names the
            # regression better than a whole-stream mismatch does.
            got_lines, want_lines = got.splitlines(), want.splitlines()
            for t, (g, w) in enumerate(zip(got_lines, want_lines)):
                assert g == w, (
                    f"{stem} turn {t}: unclejoe replied {g!r}, the golden "
                    f"recorded {w!r}")
            assert len(got_lines) == len(want_lines), (
                f"{stem}: {len(got_lines)} replies vs {len(want_lines)} recorded")
            pytest.fail(f"{stem}: stdout differs from the golden in whitespace only")
        total_replies += len(got.splitlines())
    print(f"\n[wire-replay] {len(replays)} games, {total_replies} replies, "
          f"byte-equal to the self-goldens")
