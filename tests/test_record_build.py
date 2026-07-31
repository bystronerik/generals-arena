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


# --- series aggregation over recorded games ---------------------------------


def _record_a_game(directory, *, game_id="g", turns=4, trace_a=None):
    """Write a trajectory (and optionally a seat-A trace) as a match would."""
    import json

    from arena.records.trajectories import (
        TrajectoryRecorder,
        gzip_into_place,
        trace_path,
    )

    rec = TrajectoryRecorder(
        game_id=game_id, seed=0, mode="competition", round_name="r",
        engine_version="e", bot_a="metro", bot_b="blitz", directory=directory,
    )
    rec.set_dims(3, 3)
    for turn in range(1, turns + 1):
        rec.record_turn(turn, (0,) * 5, (0,) * 5, (turn, 1), (turn * 2, 2))
    rec.finish(winner="a", turns=turns, terminated=True, truncated=False)
    rec.write()

    if trace_a is not None:
        raw = directory / "scratch.jsonl"
        raw.write_text(
            "".join(json.dumps({"t": t + 1, **row}) + "\n" for t, row in enumerate(trace_a)),
            encoding="utf-8",
        )
        gzip_into_place(raw, trace_path(game_id, "a", directory))
    return game_id


def test_an_unrecorded_game_has_no_series_metrics(tmp_path):
    from arena.records.telemetry import series_metrics

    assert series_metrics(tmp_path, "never-recorded") == {}


def test_engine_series_reduce_for_both_seats(tmp_path):
    from arena.records.telemetry import series_metrics

    _record_a_game(tmp_path)
    metrics = series_metrics(tmp_path, "g")

    assert metrics["land_mean_a"] == 2.5  # 1,2,3,4
    assert metrics["land_max_a"] == 4
    assert metrics["land_argmax_turn_a"] == 4
    assert metrics["army_mean_b"] == 2.0
    # Seat-relative margins: A's lead is B's deficit.
    assert metrics["land_margin_a"] == 3
    assert metrics["land_margin_b"] == -3
    assert metrics["land_margin_auc_a"] == -metrics["land_margin_auc_b"]
    # A first led on turn 2 (1-1 is level, not ahead); B never led.
    assert metrics["land_margin_first_turn_a"] == 2
    assert "land_margin_first_turn_b" not in metrics


def test_a_seat_without_a_probe_contributes_engine_series_only(tmp_path):
    from arena.records.telemetry import series_metrics

    _record_a_game(tmp_path, trace_a=[{"phase": "open"}, {"phase": "raid"}])
    metrics = series_metrics(tmp_path, "g")

    assert metrics["phase_a"] == "raid"
    assert "phase_b" not in metrics  # blitz's trace was never written


def test_probe_series_reduce_through_the_declared_reducers(tmp_path):
    from arena.records.telemetry import series_metrics

    _record_a_game(
        tmp_path,
        trace_a=[
            {"enemy_general_sighted": 0, "guard": 1},
            {"enemy_general_sighted": 0, "guard": 5},
            {"enemy_general_sighted": 1, "guard": 3},
        ],
    )
    metrics = series_metrics(tmp_path, "g")

    assert metrics["enemy_general_sighted_a"] is True
    assert metrics["enemy_general_sighted_first_turn_a"] == 3
    assert (metrics["guard_a"], metrics["guard_mean_a"], metrics["guard_max_a"]) == (3, 3.0, 5)


def test_a_never_true_predicate_emits_nothing(tmp_path):
    from arena.records.telemetry import series_metrics

    _record_a_game(tmp_path, trace_a=[{"enemy_general_sighted": 0}] * 3)
    metrics = series_metrics(tmp_path, "g")
    assert metrics["enemy_general_sighted_a"] is False
    assert "enemy_general_sighted_first_turn_a" not in metrics


def test_an_undeclared_trace_key_raises_at_record_build(tmp_path):
    from arena.records.telemetry_schema import UnknownTelemetryKey
    from arena.records.telemetry import series_metrics

    _record_a_game(tmp_path, trace_a=[{"brand_new_counter": 3}])
    with pytest.raises(UnknownTelemetryKey, match="brand_new_counter"):
        series_metrics(tmp_path, "g")


def test_series_metrics_merge_into_the_record(tmp_path):
    result = _result()
    result.series_metrics = {"phase_a": "raid", "land_mean_a": 2.5}
    metrics = record_from_match_result(result, **IDENTITY).metrics

    assert metrics["phase_a"] == "raid"
    assert metrics["final_land_a"] == 50  # engine truth still there


def test_a_key_computed_twice_must_agree():
    """`land_margin_a` is both an engine final and a reducer over the series."""
    from arena.records.telemetry import ConflictingMetric

    stalled = _result(winner="draw", terminated=False, truncated=True, turns=1200)
    stalled.series_metrics = {"land_margin_a": 20}  # agrees with 50 - 30
    assert record_from_match_result(stalled, **IDENTITY).metrics["land_margin_a"] == 20

    stalled.series_metrics = {"land_margin_a": 999}
    with pytest.raises(ConflictingMetric, match="land_margin_a"):
        record_from_match_result(stalled, **IDENTITY)
