"""Part 11 self-play — mixture, reproducibility, shard replay, full stack."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from dataclasses import replace

pytestmark = pytest.mark.morpheus

from training.morpheus.self_play.driver import DriverConfig, play_matchup, run_batch
from training.morpheus.self_play.league import smoke_league
from training.morpheus.self_play.sampler import (
    MixtureConfig,
    PanelMember,
    mixture_proportions,
    sample_matchup,
)
from training.morpheus.self_play.schema import read_shard, value_targets_from_winner
from training.morpheus.self_play.verify import verify_directory, verify_shard

REPO = Path(__file__).resolve().parents[3]


def _mixture() -> MixtureConfig:
    return MixtureConfig(
        league_weight=0.7,
        panel_weight=0.3,
        panel_members=(
            PanelMember(bot_id="smoke", role="heuristic", weight=1.0),
        ),
    )


def test_mixture_proportions_match_config_over_batch():
    league = smoke_league()
    mixture = _mixture()
    rng = np.random.default_rng(123)
    matchups = [sample_matchup(rng, league, mixture) for _ in range(200)]
    props = mixture_proportions(matchups)
    # Binomial noise around 0.7 / 0.3 — keep a loose gate for n=200.
    assert abs(props["league"] - 0.7) < 0.12
    assert abs(props["fixed_panel"] - 0.3) < 0.12


def test_seeded_batch_is_reproducible(tmp_path: Path):
    cfg = DriverConfig(
        games=2,
        seed=11,
        max_turns=4,
        league_weight=1.0,
        panel_weight=0.0,
        panel_members=(
            {"bot_id": "smoke", "role": "heuristic", "weight": 1.0},
        ),
        n_particles=2,
        target_simulations=4,
        min_simulations=2,
        search_depth=2,
        pending_leaf_batch=2,
        output=str(tmp_path / "a"),
    )
    a = run_batch(cfg, output=tmp_path / "a", repo_root=REPO)
    b = run_batch(
        replace(cfg, output=str(tmp_path / "b")),
        output=tmp_path / "b",
        repo_root=REPO,
    )
    assert [m.to_dict() for m in a.matchups] == [m.to_dict() for m in b.matchups]
    for pa, pb in zip(a.shards, b.shards):
        sa, sb = read_shard(pa), read_shard(pb)
        assert sa.winner == sb.winner
        assert [(t.action_a, t.action_b) for t in sa.turns] == [
            (t.action_a, t.action_b) for t in sb.turns
        ]
        assert [(t.policy_a, t.policy_b) for t in sa.turns] == [
            (t.policy_a, t.policy_b) for t in sb.turns
        ]


def test_league_game_runs_full_stack_both_seats(tmp_path: Path):
    league = smoke_league()
    mixture = MixtureConfig(
        league_weight=1.0,
        panel_weight=0.0,
        panel_members=(PanelMember(bot_id="smoke", weight=1.0),),
    )
    matchup = sample_matchup(np.random.default_rng(0), league, mixture, map_seed=3)
    assert matchup.source == "league"
    shard = play_matchup(
        matchup,
        game_id="league_full_stack",
        runtime_kwargs={
            "n_particles": 2,
            "target_simulations": 4,
            "min_simulations": 2,
            "search_depth": 2,
            "pending_leaf_batch": 2,
        },
        max_turns=4,
    )
    assert shard.turns
    for frame in shard.turns:
        assert frame.policy_a is not None
        assert frame.policy_b is not None
    issues = verify_shard(shard)
    assert issues == []


def test_shard_replay_matches_actions_and_outcome(tmp_path: Path):
    cfg = DriverConfig(
        games=1,
        seed=5,
        max_turns=6,
        league_weight=1.0,
        panel_weight=0.0,
        panel_members=({"bot_id": "smoke", "weight": 1.0},),
        n_particles=2,
        target_simulations=4,
        min_simulations=2,
        search_depth=2,
        output=str(tmp_path),
    )
    result = run_batch(cfg, output=tmp_path, repo_root=REPO)
    report = verify_directory(tmp_path)
    assert report.ok, report.to_dict()
    shard = read_shard(result.shards[0])
    assert shard.value_targets == value_targets_from_winner(shard.winner)


def test_refuses_write_into_data_games(tmp_path: Path):
    cfg = DriverConfig(
        games=1,
        seed=0,
        max_turns=2,
        league_weight=1.0,
        panel_weight=0.0,
        panel_members=({"bot_id": "smoke", "weight": 1.0},),
        n_particles=2,
        target_simulations=2,
        min_simulations=1,
        search_depth=1,
        output=str(REPO / "data" / "games" / "self-play-should-fail"),
    )
    with pytest.raises(ValueError, match="data/games"):
        run_batch(cfg, repo_root=REPO)


def test_panel_matchup_records_learner_policy_only():
    league = smoke_league()
    mixture = MixtureConfig(
        league_weight=0.0,
        panel_weight=1.0,
        panel_members=(PanelMember(bot_id="smoke", weight=1.0),),
    )
    matchup = sample_matchup(np.random.default_rng(1), league, mixture, map_seed=9)
    assert matchup.source == "fixed_panel"
    shard = play_matchup(
        matchup,
        game_id="panel_one",
        runtime_kwargs={
            "n_particles": 2,
            "target_simulations": 4,
            "min_simulations": 2,
            "search_depth": 2,
        },
        max_turns=3,
    )
    learner = matchup.learner_seat
    for frame in shard.turns:
        policies = (frame.policy_a, frame.policy_b)
        assert policies[learner] is not None
        assert policies[1 - learner] is None
