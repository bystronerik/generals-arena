"""Part 11 league — immutable snapshots and epoch freeze."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.morpheus

from training.morpheus.self_play.league import (
    League,
    LeagueError,
    smoke_league,
    stub_snapshot,
)


def test_smoke_league_has_required_roles():
    league = smoke_league()
    assert league.frozen
    roles = {s.role for s in league.snapshots()}
    assert roles == {"learner", "best", "recent", "exploiter"}
    assert league.learner().snapshot_id == "learner"


def test_register_rejected_when_frozen():
    league = smoke_league()
    with pytest.raises(LeagueError, match="frozen"):
        league.register(
            stub_snapshot(snapshot_id="extra", role="recent", weight=1.0)
        )


def test_replace_rejected_when_frozen():
    league = smoke_league()
    with pytest.raises(LeagueError, match="frozen"):
        league.replace(
            stub_snapshot(snapshot_id="best", role="best", weight=2.0, stub_tag="x")
        )


def test_require_immutable_rejects_digest_mutation():
    league = smoke_league()
    snap = league.get("best")
    with pytest.raises(LeagueError, match="mutated"):
        league.require_immutable(snap.snapshot_id, "sha256:deadbeef")


def test_new_epoch_allows_replace_then_refreeze():
    league = smoke_league()
    league.begin_epoch()
    assert not league.frozen
    league.replace(
        stub_snapshot(snapshot_id="best", role="best", weight=2.0, stub_tag="v2")
    )
    league.freeze_epoch()
    assert league.get("best").weight == 2.0
    with pytest.raises(LeagueError, match="frozen"):
        league.replace(
            stub_snapshot(snapshot_id="best", role="best", weight=3.0, stub_tag="v3")
        )


def test_sample_opponent_deterministic():
    league = smoke_league()
    import numpy as np

    a = league.sample_opponent(np.random.default_rng(7), exclude_ids={"learner"})
    b = league.sample_opponent(np.random.default_rng(7), exclude_ids={"learner"})
    assert a.snapshot_id == b.snapshot_id
