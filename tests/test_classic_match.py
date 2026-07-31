"""Fast smoke for classic harness (tiny board, short truncation)."""
from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SMOKE_RUN = REPO_ROOT / "bots" / "smoke" / "run.sh"


@pytest.mark.slow
def test_classic_match_smoke_truncates():
    from arena.matches.classic import run_classic_match

    if not SMOKE_RUN.is_file():
        pytest.skip("smoke bot missing")

    winner, turns, truncated = run_classic_match(
        SMOKE_RUN,
        SMOKE_RUN,
        seed=0,
        env_overrides={
            "grid_dims": (8, 8),
            "truncation": 5,
            "num_castles_range": (1, 1),
            "castle_val_range": (10, 10),
        },
    )
    assert truncated is True
    assert turns == 5
    assert winner == -1
