"""Tests for global WDL curriculum stratification."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.morpheus

from training.morpheus.curriculum.stratify import (
    GameCandidate,
    StratifyError,
    split_equal_thirds,
    stratify_global_wdl,
)


def _cand(game_id: str, player: str, outcome: str) -> GameCandidate:
    return GameCandidate(
        path=Path(f"/tmp/{game_id}.traj.jsonl.gz"),
        game_id=game_id,
        player=player,
        outcome=outcome,
        source_label=f"{player}_reconstructions",
    )


def test_split_equal_thirds_remainder_by_sorted_name():
    quotas = split_equal_thirds(10, ["ResBot", "Kubic", "thor"])
    assert quotas == {"Kubic": 4, "ResBot": 3, "thor": 3}
    assert sum(quotas.values()) == 10


def test_stratify_global_wdl_balances_and_caps_top_players():
    candidates: list[GameCandidate] = []
    for i in range(20):
        candidates.append(_cand(f"loss-{i:03d}", "erik.bystron", "lose"))
    for i in range(30):
        candidates.append(_cand(f"resbot-win-{i:03d}", "ResBot", "win"))
    for i in range(30):
        candidates.append(_cand(f"kubic-win-{i:03d}", "Kubic", "win"))
    for i in range(30):
        candidates.append(_cand(f"thor-win-{i:03d}", "thor", "win"))
    for i in range(40):
        candidates.append(_cand(f"other-win-{i:03d}", "candide", "win"))

    result = stratify_global_wdl(
        candidates,
        top_win_players=["ResBot", "Kubic", "thor"],
        top_win_fraction=0.5,
    )
    assert result.n_pairs == 20
    assert result.n_wins == 20
    assert result.n_losses == 20
    assert result.top_win_quota == 10
    wins = [c for c in result.kept if c.outcome == "win"]
    top = {"ResBot", "Kubic", "thor"}
    top_wins = [c for c in wins if c.player in top]
    other_wins = [c for c in wins if c.player not in top]
    assert len(top_wins) == 10
    assert len(other_wins) == 10
    by_player = {p: 0 for p in top}
    for c in top_wins:
        by_player[c.player] += 1
    assert by_player == {"Kubic": 4, "ResBot": 3, "thor": 3}

    again = stratify_global_wdl(
        candidates,
        top_win_players=["ResBot", "Kubic", "thor"],
        top_win_fraction=0.5,
    )
    assert [c.game_id for c in again.kept] == [c.game_id for c in result.kept]


def test_stratify_fail_closed_when_top_pool_short():
    candidates = [_cand("l0", "erik.bystron", "lose"), _cand("w0", "ResBot", "win")]
    with pytest.raises(StratifyError, match="need"):
        stratify_global_wdl(
            candidates,
            top_win_players=["ResBot", "Kubic", "thor"],
            top_win_fraction=0.5,
        )


def test_discover_trajectory_dirs_reconstructions_only(tmp_path: Path):
    from training.morpheus.curriculum.build import discover_trajectory_dirs

    recon = tmp_path / "ResBot-win-reconstructions"
    recon.mkdir()
    (recon / "corpus-index.json").write_text("{}", encoding="utf-8")
    other = tmp_path / "morpheus-bootstrap"
    other.mkdir()
    (other / "x.traj.jsonl.gz").write_bytes(b"")
    found = discover_trajectory_dirs(tmp_path, reconstructions_only=True)
    assert found == [recon]
    found_all = discover_trajectory_dirs(tmp_path, reconstructions_only=False)
    assert set(found_all) == {recon, other}
