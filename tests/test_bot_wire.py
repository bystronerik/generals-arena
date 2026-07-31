"""Tests for shared bot stdio wire helpers."""
from __future__ import annotations

from _common.wire import Observation, _telemetry_line


class _FakeAgent:
    def telemetry_extras(self):
        return {
            "enemy_general_sighted": 1,
            "first_sighting_turn": 42,
            "custom_metric": 7,
        }


def test_telemetry_line_emits_all_extras_sorted():
    obs = Observation(
        H=5,
        W=5,
        turn=99,
        my_land=10,
        my_army=20,
        opp_land=8,
        opp_army=15,
        type_grid=[],
        owner_grid=[],
        army_grid=[],
    )
    line = _telemetry_line(0, obs, _FakeAgent())
    assert line.startswith("[telemetry] player=0 turn=99")
    assert "custom_metric=7" in line
    assert "enemy_general_sighted=1" in line
    assert "first_sighting_turn=42" in line
    assert line.index("custom_metric") < line.index("enemy_general_sighted")
    assert line.index("enemy_general_sighted") < line.index("first_sighting_turn")
