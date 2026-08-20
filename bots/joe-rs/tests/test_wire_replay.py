"""End-to-end decision stability: the real binary, in wire mode, over whole
recorded games, against its own recorded replies (selection-plan S1).

Each corpus game's `.in.log` is exactly what the engine sent Python joe;
piping it to `joe-rs` in wire mode exercises the shipped path — stdio parse,
state accumulation under the bot's *own* state, forward, Gumbel selection,
and the pass clamp. Since the selection layer shipped, no sibling predicts
the played move: the unpenalised oracle argmax (`all_action`) diverges
wherever the noise flips a near-tie, and deployed joe's `.out.log` records
the penalised program joe-rs deliberately is not. The reference is therefore
the binary's **own** replies, recorded by `capture_fixtures.py
--selection-golden` as `<name>.joe-rs.log`.

Selection is a deterministic function of the frame stream, so the whole
stdout must reproduce byte for byte, and an unexplained change anywhere in
the played path still fails here. The network itself stays graded against
the `.npz` oracle in `test_parity.py` (tiers 1–3), which this file does not
repeat.

Marker `joe` (needs the release binary and recorded games).
"""
from __future__ import annotations

import subprocess

import pytest

from parity_lib import BINARY, BOT_DIR, corpus_games

pytestmark = pytest.mark.joe


def test_wire_replay_matches_the_self_golden():
    if not BINARY.is_file():
        pytest.skip(f"no release binary at {BINARY}")
    replays = []
    for npz_path in corpus_games():
        stem = npz_path.name.removesuffix(".npz")
        in_log = npz_path.parent / f"{stem}.in.log"
        golden = npz_path.parent / f"{stem}.joe-rs.log"
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
                    f"{stem} turn {t}: joe-rs replied {g!r}, the golden "
                    f"recorded {w!r}")
            assert len(got_lines) == len(want_lines), (
                f"{stem}: {len(got_lines)} replies vs {len(want_lines)} recorded")
            pytest.fail(f"{stem}: stdout differs from the golden in whitespace only")
        total_replies += len(got.splitlines())
    print(f"\n[wire-replay] {len(replays)} games, {total_replies} replies, "
          f"byte-equal to the self-goldens")
