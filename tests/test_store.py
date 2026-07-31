"""GameRecord round-trip and schema migration tests."""
from __future__ import annotations

import pytest

from arena.store import GameRecord


V1_MINIMAL = {
    "game_id": "20260101T000000Z_smoke_vs_rush_s0_abcd1234",
    "seed": 0,
    "mode": "competition",
    "bot_a": "smoke",
    "bot_b": "rush",
    "bot_a_commit_or_tag": "abc1234",
    "bot_b_commit_or_tag": "abc1234",
    "winner": "a",
    "turns": 42,
    "terminated": True,
    "truncated": False,
    "started_at": "2026-01-01T00:00:00Z",
    "finished_at": "2026-01-01T00:01:00Z",
}

V2_FULL = {
    **V1_MINIMAL,
    "schema_version": 2,
    "duration_seconds": 60.0,
    "castles_built_a": 2,
    "castles_built_b": 1,
    "final_land_a": 50,
    "final_land_b": 30,
    "final_army_a": 100,
    "final_army_b": 80,
    "metrics": {"enemy_general_sighted_a": True},
}


@pytest.mark.parametrize("payload", [V1_MINIMAL, V2_FULL], ids=["v1", "v2"])
def test_game_record_round_trip(payload):
    record = GameRecord.from_dict(payload)
    restored = GameRecord.from_dict(record.to_dict())
    assert restored == record


def test_v1_defaults_schema_version():
    record = GameRecord.from_dict(V1_MINIMAL)
    assert record.schema_version == 1
    assert record.castles_built_a is None
    assert record.metrics == {}


def test_v2_preserves_telemetry_fields():
    record = GameRecord.from_dict(V2_FULL)
    assert record.schema_version == 2
    assert record.castles_built_a == 2
    assert record.final_army_b == 80
    assert record.metrics["enemy_general_sighted_a"] is True


def test_from_dict_rejects_missing_required():
    bad = dict(V1_MINIMAL)
    del bad["winner"]
    with pytest.raises(ValueError, match="missing fields"):
        GameRecord.from_dict(bad)


def test_from_dict_rejects_invalid_winner():
    bad = dict(V1_MINIMAL, winner="x")
    with pytest.raises(ValueError, match="invalid winner"):
        GameRecord.from_dict(bad)
