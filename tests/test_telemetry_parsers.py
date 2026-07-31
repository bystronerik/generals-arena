"""Table-driven tests for arena/records/telemetry.py bot-telemetry parsers."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from arena.records.store import CURRENT_SCHEMA_VERSION, GameRecord
from arena.records.telemetry import (
    BotTelemetry,
    apply_telemetry_to_record,
    parse_bot_telemetry,
    record_from_match_result,
)


TELEM_P0 = (
    "[telemetry] player=0 turn=1199 my_land=50 my_army=100 "
    "opp_land=30 opp_army=80 enemy_general_sighted=1 first_sighting_turn=200\n"
)
TELEM_P1 = "[telemetry] player=1 turn=1199 my_land=30 my_army=80 opp_land=50 opp_army=100\n"


def test_parse_bot_telemetry_last_line_per_player():
    combined = TELEM_P0 + (
        "[telemetry] player=0 turn=500 my_land=10 my_army=20 "
        "opp_land=5 opp_army=15\n"
    ) + TELEM_P1
    got = parse_bot_telemetry(combined)
    assert set(got) == {0, 1}
    assert got[0].my_land == 10
    assert got[0].enemy_general_sighted is None
    assert got[1].my_land == 30


def _minimal_record() -> GameRecord:
    return GameRecord(
        game_id="g1",
        seed=0,
        mode="competition",
        round="roundT",
        bot_a="a",
        bot_b="b",
        bot_a_content_hash="0123456789ab",
        bot_b_content_hash="ba9876543210",
        engine_version="9e3b9d1",
        winner="draw",
        turns=10,
        terminated=False,
        truncated=True,
        started_at="2026-01-01T00:00:00Z",
        finished_at="2026-01-01T00:01:00Z",
    )


def test_apply_telemetry_both_players():
    record = _minimal_record()
    telemetry = {
        0: BotTelemetry(50, 100, 30, 80, enemy_general_sighted=1, first_sighting_turn=200),
        1: BotTelemetry(30, 80, 50, 100),
    }
    apply_telemetry_to_record(record, telemetry)
    assert record.final_land_a == 50
    assert record.final_army_a == 100
    assert record.final_land_b == 30
    assert record.final_army_b == 80
    assert record.metrics["enemy_general_sighted_a"] is True
    assert record.metrics["first_sighting_turn_a"] == 200


def test_apply_telemetry_player0_only_fills_opponent_from_opp_fields():
    record = _minimal_record()
    apply_telemetry_to_record(
        record,
        {0: BotTelemetry(50, 100, 30, 80)},
    )
    assert record.final_land_b == 30
    assert record.final_army_b == 80


def test_apply_telemetry_noop_on_empty():
    record = _minimal_record()
    apply_telemetry_to_record(record, {})
    assert record.final_land_a is None
    assert record.metrics == {}


def test_parse_bot_telemetry_generic_extras():
    line = (
        "[telemetry] player=0 turn=1199 my_land=50 my_army=100 "
        "opp_land=30 opp_army=80 enemy_general_sighted=1 "
        "first_sighting_turn=200 first_city_capture_turn=450\n"
    )
    got = parse_bot_telemetry(line)
    assert got[0].extras == {
        "enemy_general_sighted": "1",
        "first_sighting_turn": "200",
        "first_city_capture_turn": "450",
    }


def test_apply_telemetry_generic_extras_and_land_margin():
    record = _minimal_record()
    apply_telemetry_to_record(
        record,
        {
            0: BotTelemetry(50, 100, 30, 80, extras={"first_city_capture_turn": "450"}),
            1: BotTelemetry(30, 80, 50, 100),
        },
    )
    assert record.metrics["first_city_capture_turn_a"] == 450
    assert record.metrics["land_margin_a"] == 20
    assert record.metrics["land_margin_b"] == -20


def test_record_from_match_result_fills_identity_timing_and_telemetry():
    """The shared builder used by run_and_store and tournament_worker."""
    result = SimpleNamespace(
        winner="a",
        turns=137,
        terminated=True,
        truncated=False,
        castles_built_a=2,
        castles_built_b=1,
        stderr=TELEM_P0 + TELEM_P1,
    )
    record = record_from_match_result(
        result,
        bot_a="blitz",
        bot_b="smoke",
        seed=7,
        mode="competition",
        round_name="round5",
        bot_a_content_hash="0123456789ab",
        bot_b_content_hash="ba9876543210",
        engine_version="9e3b9d1",
        started_at="2026-01-01T00:00:00Z",
        finished_at="2026-01-01T00:01:30Z",
    )
    assert (record.bot_a, record.bot_b, record.seed) == ("blitz", "smoke", 7)
    assert record.winner == "a" and record.turns == 137
    assert record.schema_version == CURRENT_SCHEMA_VERSION
    assert record.duration_seconds == 90.0
    assert (record.castles_built_a, record.castles_built_b) == (2, 1)
    assert (record.round, record.engine_version) == ("round5", "9e3b9d1")
    assert record.bot_a_content_hash == "0123456789ab"
    # telemetry merged from stderr, not passed in separately
    assert record.final_land_a == 50
    assert record.final_land_b == 30
    assert record.metrics["first_sighting_turn_a"] == 200


@pytest.mark.parametrize(
    "override, message",
    [
        ({"bot_a_content_hash": "unknown"}, "unknown"),
        ({"bot_b_content_hash": ""}, "non-empty"),
        ({"engine_version": ""}, "non-empty"),
    ],
)
def test_record_from_match_result_rejects_a_missing_identity(override, message):
    """A record with no rating identity must never reach the store."""
    result = SimpleNamespace(
        winner="draw",
        turns=1200,
        terminated=False,
        truncated=True,
        castles_built_a=None,
        castles_built_b=None,
        stderr="",
    )
    kwargs = {
        "bot_a": "blitz",
        "bot_b": "smoke",
        "seed": 0,
        "mode": "competition",
        "round_name": "round5",
        "bot_a_content_hash": "0123456789ab",
        "bot_b_content_hash": "ba9876543210",
        "engine_version": "9e3b9d1",
        "started_at": "2026-01-01T00:00:00Z",
        "finished_at": "2026-01-01T00:01:00Z",
        **override,
    }
    with pytest.raises(ValueError, match=message):
        record_from_match_result(result, **kwargs)
