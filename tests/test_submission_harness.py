"""Table-driven negative controls for the submission-shaped harness."""
from __future__ import annotations

from pathlib import Path

import pytest

from arena.matches.submission import RejectReason, run_submission_match
from arena.paths import REPO_ROOT

pytestmark = pytest.mark.morpheus

FIXTURES = REPO_ROOT / "tests" / "fixtures" / "submission_bots"
SMOKE = REPO_ROOT / "bots" / "smoke" / "run.sh"
GAMES_DIR = REPO_ROOT / "data" / "games"

# Short clocks so timeout fixtures stay cheap. Memory cap sits above a normal
# Python process for a few turns and below the memory_bomb fixture.
_FIRST_S = 0.25
_NORMAL_S = 0.15
_EOF_GRACE_S = 0.4
_MEMORY_CAP = 40 * 1024 * 1024
_MAX_TURNS = 3


def _run_sh(name: str) -> Path:
    return FIXTURES / name / "run.sh"


def _games_snapshot() -> set[str]:
    if not GAMES_DIR.is_dir():
        return set()
    return {p.name for p in GAMES_DIR.rglob("*.json")}


@pytest.mark.parametrize(
    ("fixture", "reason"),
    [
        ("good", RejectReason.ACCEPTED),
        ("first_timeout", RejectReason.FIRST_REPLY_TIMEOUT),
        ("reply_timeout", RejectReason.NORMAL_REPLY_TIMEOUT),
        ("missing_reply", RejectReason.MISSING_REPLY),
        ("bad_output", RejectReason.MALFORMED_REPLY),
        ("memory_bomb", RejectReason.MEMORY_CAP),
        ("crash", RejectReason.CRASH),
        ("eof_hang", RejectReason.EOF_HANG),
        ("eof_nonzero", RejectReason.EOF_NONZERO),
    ],
)
def test_submission_harness_cases(fixture: str, reason: RejectReason) -> None:
    before = _games_snapshot()
    result = run_submission_match(
        _run_sh(fixture),
        SMOKE,
        seed=0,
        mode="competition",
        first_reply_s=_FIRST_S,
        normal_reply_s=_NORMAL_S,
        eof_grace_s=_EOF_GRACE_S,
        memory_cap_bytes=_MEMORY_CAP,
        max_turns=_MAX_TURNS,
    )
    after = _games_snapshot()

    assert after == before, "judged fixture runs must not write data/games/"
    assert result.reason == reason
    if reason is RejectReason.ACCEPTED:
        assert result.accepted
        assert result.faults == 0
        assert result.eof_voluntary
        assert result.eof_exit_code == 0
        assert not result.eof_forced
    else:
        assert not result.accepted
