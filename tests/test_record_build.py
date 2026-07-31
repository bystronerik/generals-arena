"""Building a stored record from a match result: engine truth, nothing else."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from arena.records.store import CURRENT_SCHEMA_VERSION
from arena.records.telemetry import engine_metrics, record_from_match_result

IDENTITY = {
    # Minted before the match, not by the record builder: a trajectory is keyed
    # by the game it belongs to.
    "game_id": "20260101T000000Z_blitz_vs_smoke_s7_abcd1234",
    "bot_a": "blitz",
    "bot_b": "smoke",
    "seed": 7,
    "mode": "competition",
    "round_name": "round5",
    "bot_a_content_hash": "0123456789ab",
    "bot_b_content_hash": "ba9876543210",
    "engine_version": "9e3b9d1",
}


def _result(**overrides) -> SimpleNamespace:
    base = dict(
        winner="a",
        turns=137,
        terminated=True,
        truncated=False,
        castles_built_a=2,
        castles_built_b=1,
        final_land_a=50,
        final_land_b=30,
        final_army_a=100,
        final_army_b=80,
        stderr="",
    )
    return SimpleNamespace(**{**base, **overrides})


def test_record_carries_identity_outcome_and_engine_metrics():
    record = record_from_match_result(_result(), **IDENTITY)

    assert record.game_id == IDENTITY["game_id"]
    assert (record.bot_a, record.bot_b, record.seed) == ("blitz", "smoke", 7)
    assert (record.winner, record.turns, record.truncated) == ("a", 137, False)
    assert (record.round, record.engine_version) == ("round5", "9e3b9d1")
    assert record.schema_version == CURRENT_SCHEMA_VERSION
    assert record.metrics == {
        "castles_built_a": 2,
        "castles_built_b": 1,
        "final_land_a": 50,
        "final_land_b": 30,
        "final_army_a": 100,
        "final_army_b": 80,
    }


def test_finals_come_from_the_engine_not_from_bot_stderr():
    """A bot that shouts telemetry at stderr changes nothing: nobody reads it."""
    noisy = _result(stderr="[telemetry] player=0 my_land=999 my_army=999\n")
    assert record_from_match_result(noisy, **IDENTITY).metrics["final_land_a"] == 50


def test_an_unrecorded_match_carries_no_probe_metrics():
    """Missing means not measured: per-turn keys need `--record`."""
    metrics = record_from_match_result(_result(), **IDENTITY).metrics
    assert not {"phase_a", "enemy_general_sighted_a", "strikes_a"} & set(metrics)


def test_land_margin_only_on_a_truncated_draw():
    stalled = _result(winner="draw", terminated=False, truncated=True, turns=1200)
    metrics = record_from_match_result(stalled, **IDENTITY).metrics
    assert metrics["land_margin_a"] == 20
    assert metrics["land_margin_b"] == -20

    assert "land_margin_a" not in record_from_match_result(_result(), **IDENTITY).metrics


def test_unmeasured_engine_values_are_omitted_not_zeroed():
    """Classic envs build no castles; a match with no steps has no finals."""
    empty = _result(
        castles_built_a=None,
        castles_built_b=None,
        final_land_a=None,
        final_land_b=None,
        final_army_a=None,
        final_army_b=None,
    )
    assert engine_metrics(empty) == {}


@pytest.mark.parametrize(
    "override, message",
    [
        ({"bot_a_content_hash": "unknown"}, "unknown"),
        ({"bot_b_content_hash": ""}, "non-empty"),
        ({"engine_version": ""}, "non-empty"),
    ],
)
def test_a_record_with_no_rating_identity_is_refused(override, message):
    with pytest.raises(ValueError, match=message):
        record_from_match_result(_result(), **{**IDENTITY, **override})
