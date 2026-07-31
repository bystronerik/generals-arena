"""The one-shot v4 -> v5 projection: what it moves, drops, and refuses."""
from __future__ import annotations

import json

import pytest

from arena.records.store import GameRecord
from scripts.migrate_games_v5 import MigrationError, migrate_file, project_v5

V4_RECORD = {
    "game_id": "20260101T000000Z_metro_vs_smoke_s7_abcd1234",
    "seed": 7,
    "mode": "competition",
    "round": "round5",
    "bot_a": "metro",
    "bot_b": "smoke",
    "bot_a_content_hash": "0123456789ab",
    "bot_b_content_hash": "ba9876543210",
    "engine_version": "9e3b9d1f00112233445566778899aabbccddeeff",
    "winner": "draw",
    "turns": 1200,
    "terminated": False,
    "truncated": True,
    "started_at": "2026-01-01T00:00:00Z",
    "finished_at": "2026-01-01T00:01:00Z",
    "schema_version": 4,
    "duration_seconds": 60.0,
    "castles_built_a": 4,
    "castles_built_b": 0,
    "final_land_a": 50,
    "final_land_b": 30,
    "final_army_a": 100,
    "final_army_b": 80,
    "metrics": {
        "castles_built_a": 3,  # metro's own counter, not the engine tally
        "enemy_general_sighted_a": True,
        "land_margin_a": 20,
        "land_margin_b": -20,
    },
}


def test_projection_matches_the_expected_v5_record():
    assert project_v5(V4_RECORD) == {
        "game_id": V4_RECORD["game_id"],
        "seed": 7,
        "mode": "competition",
        "round": "round5",
        "bot_a": "metro",
        "bot_b": "smoke",
        "bot_a_content_hash": "0123456789ab",
        "bot_b_content_hash": "ba9876543210",
        "engine_version": V4_RECORD["engine_version"],
        "winner": "draw",
        "turns": 1200,
        "truncated": True,
        "schema_version": 5,
        "metrics": {
            "castles_built_a": 4,
            "castles_built_b": 0,
            "castles_built_probe_a": 3,
            "final_land_a": 50,
            "final_land_b": 30,
            "final_army_a": 100,
            "final_army_b": 80,
            "enemy_general_sighted_a": True,
            "land_margin_a": 20,
            "land_margin_b": -20,
        },
    }


def test_identity_and_outcome_are_untouched():
    """The fit reads only these, so ratings over the migrated pool are identical."""
    v5 = project_v5(V4_RECORD)
    for field in (
        "game_id", "seed", "mode", "round", "bot_a", "bot_b",
        "bot_a_content_hash", "bot_b_content_hash", "engine_version",
        "winner", "turns", "truncated",
    ):
        assert v5[field] == V4_RECORD[field], field


def test_shadowed_castle_counts_both_survive():
    """The engine tally and metro's belief disagree on real games; keep both."""
    metrics = project_v5(V4_RECORD)["metrics"]
    assert metrics["castles_built_a"] == 4
    assert metrics["castles_built_probe_a"] == 3


def test_projected_record_loads_as_v5():
    record = GameRecord.from_dict(project_v5(V4_RECORD))
    assert record.schema_version == 5
    assert record.terminated is False  # derived from winner


def test_projection_is_idempotent():
    once = project_v5(V4_RECORD)
    assert project_v5(once) == once


def test_pre_v4_records_are_refused():
    with pytest.raises(MigrationError, match="only v4"):
        project_v5(dict(V4_RECORD, schema_version=3))


def test_a_would_be_overwrite_raises_instead_of_losing_a_value():
    """Reinterpretation is the one thing a pure projection must never do."""
    hostile = dict(V4_RECORD, metrics={"final_land_a": 999})
    with pytest.raises(MigrationError, match="final_land_a"):
        project_v5(hostile)


def test_migrate_file_rewrites_in_place_and_is_idempotent(tmp_path):
    path = tmp_path / "game.json"
    path.write_text(json.dumps(V4_RECORD), encoding="utf-8")

    assert migrate_file(path, dry_run=False) is True
    assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] == 5
    assert migrate_file(path, dry_run=False) is False


def test_dry_run_writes_nothing(tmp_path):
    path = tmp_path / "game.json"
    path.write_text(json.dumps(V4_RECORD), encoding="utf-8")

    assert migrate_file(path, dry_run=True) is True
    assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] == 4
