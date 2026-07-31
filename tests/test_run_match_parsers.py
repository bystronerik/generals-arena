"""Table-driven tests for arena/run_match.py stdout parsers."""
from __future__ import annotations

import pytest

from arena.run_match import (
    BotTelemetry,
    apply_telemetry_to_record,
    parse_bot_telemetry,
    parse_castles_built,
    parse_matchup_output,
)
from arena.store import GameRecord


WIN_A = "[matchup] turn 42: player 0 captured opponent general\n"
WIN_B = "[matchup] turn 17: player 1 captured opponent general\n"
DRAW = "[matchup] turn 1200: truncated\n"
CASTLE_LINE = "[matchup] castles built: 2 (smoke) vs 1 (rush)\n"
TELEM_P0 = (
    "[telemetry] player=0 turn=1199 my_land=50 my_army=100 "
    "opp_land=30 opp_army=80 enemy_general_sighted=1 first_sighting_turn=200\n"
)
TELEM_P1 = "[telemetry] player=1 turn=1199 my_land=30 my_army=80 opp_land=50 opp_army=100\n"


@pytest.mark.parametrize(
    ("combined", "expected"),
    [
        (WIN_A, ("a", 42, True, False)),
        (WIN_B, ("b", 17, True, False)),
        (DRAW, ("draw", 1200, False, True)),
        (WIN_A + DRAW, ("a", 42, True, False)),
        (DRAW + WIN_B, ("b", 17, True, False)),
    ],
    ids=["win_a", "win_b", "draw", "win_then_draw", "draw_then_win"],
)
def test_parse_matchup_output(combined, expected):
    assert parse_matchup_output(combined) == expected


def test_parse_matchup_output_raises_on_missing_result():
    with pytest.raises(ValueError, match="could not parse matchup result"):
        parse_matchup_output("no result lines here\n")


@pytest.mark.parametrize(
    ("combined", "expected"),
    [
        ("", (None, None)),
        ("no castles\n", (None, None)),
        (CASTLE_LINE, (2, 1)),
        (CASTLE_LINE + "[matchup] castles built: 9 (a) vs 8 (b)\n", (9, 8)),
    ],
    ids=["empty", "missing", "single", "last_wins"],
)
def test_parse_castles_built(combined, expected):
    assert parse_castles_built(combined) == expected


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
        bot_a="a",
        bot_b="b",
        bot_a_commit_or_tag="abc",
        bot_b_commit_or_tag="abc",
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
