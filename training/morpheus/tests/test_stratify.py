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


def test_split_equal_across_six_players():
    quotas = split_equal_thirds(10, ["Kubic", "ResBot", "bca", "nanomena", "thor", "Mattz"])
    assert quotas == {
        "Kubic": 2,
        "Mattz": 2,
        "ResBot": 2,
        "bca": 2,
        "nanomena": 1,
        "thor": 1,
    }
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


def test_stratify_global_wdl_75_percent_six_top_players():
    top = ["Kubic", "ResBot", "bca", "nanomena", "thor", "Mattz"]
    candidates: list[GameCandidate] = []
    for i in range(40):
        candidates.append(_cand(f"loss-{i:03d}", "erik.bystron", "lose"))
    for player in top:
        for i in range(20):
            candidates.append(_cand(f"{player}-win-{i:03d}", player, "win"))
    for i in range(50):
        candidates.append(_cand(f"other-win-{i:03d}", "candide", "win"))

    result = stratify_global_wdl(
        candidates,
        top_win_players=top,
        top_win_fraction=0.75,
    )
    assert result.n_pairs == 40
    assert result.n_wins == 40
    assert result.n_losses == 40
    assert result.top_win_quota == 30
    wins = [c for c in result.kept if c.outcome == "win"]
    top_set = set(top)
    top_wins = [c for c in wins if c.player in top_set]
    other_wins = [c for c in wins if c.player not in top_set]
    assert len(top_wins) == 30
    assert len(other_wins) == 10
    by_player = {p: 0 for p in top}
    for c in top_wins:
        by_player[c.player] += 1
    assert by_player == split_equal_thirds(30, top)
    assert all(c.outcome != "draw" for c in result.kept)


def test_stratify_fail_closed_when_top_pool_short():
    candidates = [_cand("l0", "erik.bystron", "lose"), _cand("w0", "ResBot", "win")]
    with pytest.raises(StratifyError, match="feasible|need"):
        stratify_global_wdl(
            candidates,
            top_win_players=["ResBot", "Kubic", "thor"],
            top_win_fraction=0.5,
        )


def test_stratify_caps_n_when_top_player_short():
    top = ["Kubic", "ResBot", "bca", "nanomena", "thor", "Mattz"]
    candidates: list[GameCandidate] = []
    for i in range(100):
        candidates.append(_cand(f"loss-{i:03d}", "erik.bystron", "lose"))
    for player in top:
        # Mattz is the bottleneck at 5 wins.
        n_wins = 5 if player == "Mattz" else 40
        for i in range(n_wins):
            candidates.append(_cand(f"{player}-win-{i:03d}", player, "win"))
    for i in range(80):
        candidates.append(_cand(f"other-win-{i:03d}", "candide", "win"))

    result = stratify_global_wdl(
        candidates,
        top_win_players=top,
        top_win_fraction=0.75,
    )
    assert result.n_pairs == 42  # largest n with Mattz=5 under equal 75% sixths
    assert result.n_wins == result.n_losses == 42
    assert result.top_win_quota == 31
    assert any("capped n from 100 to 42" in note for note in result.notes)
    wins = [c for c in result.kept if c.outcome == "win"]
    by_player = {p: 0 for p in top}
    for c in wins:
        if c.player in by_player:
            by_player[c.player] += 1
    assert by_player == split_equal_thirds(31, top)
    assert by_player["Mattz"] == 5


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
