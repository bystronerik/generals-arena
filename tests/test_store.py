"""GameRecord v4 round-trip, required identity fields, and path scanning."""
from __future__ import annotations

import json

import pytest

from arena.records.store import (
    CURRENT_SCHEMA_VERSION,
    GameRecord,
    list_game_paths,
    load_all_games,
    round_games_dir,
    save_game,
)


V4_MINIMAL = {
    "game_id": "20260101T000000Z_smoke_vs_rush_s0_abcd1234",
    "seed": 0,
    "mode": "competition",
    "round": "round5",
    "bot_a": "smoke",
    "bot_b": "rush",
    "bot_a_content_hash": "0123456789ab",
    "bot_b_content_hash": "ba9876543210",
    "engine_version": "9e3b9d1f00112233445566778899aabbccddeeff",
    "winner": "a",
    "turns": 42,
    "terminated": True,
    "truncated": False,
    "started_at": "2026-01-01T00:00:00Z",
    "finished_at": "2026-01-01T00:01:00Z",
    "schema_version": 4,
}

V4_FULL = {
    **V4_MINIMAL,
    "duration_seconds": 60.0,
    "castles_built_a": 2,
    "castles_built_b": 1,
    "final_land_a": 50,
    "final_land_b": 30,
    "final_army_a": 100,
    "final_army_b": 80,
    "metrics": {"enemy_general_sighted_a": True},
}


@pytest.mark.parametrize("payload", [V4_MINIMAL, V4_FULL], ids=["minimal", "full"])
def test_game_record_round_trip(payload):
    record = GameRecord.from_dict(payload)
    restored = GameRecord.from_dict(record.to_dict())
    assert restored == record


def test_minimal_defaults_optional_fields():
    record = GameRecord.from_dict(V4_MINIMAL)
    assert record.schema_version == CURRENT_SCHEMA_VERSION
    assert record.castles_built_a is None
    assert record.metrics == {}


def test_full_preserves_telemetry_fields():
    record = GameRecord.from_dict(V4_FULL)
    assert record.castles_built_a == 2
    assert record.final_army_b == 80
    assert record.metrics["enemy_general_sighted_a"] is True


def test_identity_fields_survive_the_round_trip():
    record = GameRecord.from_dict(V4_MINIMAL)
    assert record.bot_a_content_hash == "0123456789ab"
    assert record.bot_b_content_hash == "ba9876543210"
    assert record.engine_version.startswith("9e3b9d1")
    assert record.round == "round5"


@pytest.mark.parametrize("version", [1, 2, 3])
def test_pre_v4_records_are_rejected_loudly(version):
    """No record survives the refactor, so a stray one must not load quietly."""
    stale = dict(V4_MINIMAL, schema_version=version)
    with pytest.raises(ValueError, match="not readable"):
        GameRecord.from_dict(stale)


@pytest.mark.parametrize(
    "field", ["winner", "round", "engine_version", "bot_a_content_hash"]
)
def test_from_dict_rejects_missing_required(field):
    bad = dict(V4_MINIMAL)
    del bad[field]
    with pytest.raises(ValueError, match="missing fields"):
        GameRecord.from_dict(bad)


def test_from_dict_rejects_invalid_winner():
    bad = dict(V4_MINIMAL, winner="x")
    with pytest.raises(ValueError, match="invalid winner"):
        GameRecord.from_dict(bad)


def test_from_dict_rejects_the_unknown_hash_sentinel():
    """`"unknown"` would pool every unreadable closure into one rated entity."""
    bad = dict(V4_MINIMAL, bot_b_content_hash="unknown")
    with pytest.raises(ValueError, match="unknown"):
        GameRecord.from_dict(bad)


def test_from_dict_rejects_an_empty_hash():
    bad = dict(V4_MINIMAL, bot_a_content_hash="")
    with pytest.raises(ValueError, match="non-empty"):
        GameRecord.from_dict(bad)


def test_round_games_dir(tmp_path):
    path = round_games_dir("round3", games_root=tmp_path)
    assert path == tmp_path / "round3"


def test_round_games_dir_rejects_bad_name():
    with pytest.raises(ValueError, match="invalid round name"):
        round_games_dir("../escape")


def test_list_game_paths_recursive_skips_manifest(monkeypatch, tmp_path):
    monkeypatch.setattr("arena.records.store.GAMES_DIR", tmp_path)
    flat = tmp_path / "flat.json"
    flat.write_text(json.dumps(V4_MINIMAL) + "\n", encoding="utf-8")
    round_dir = tmp_path / "roundX"
    round_dir.mkdir()
    (round_dir / "manifest.json").write_text("{}", encoding="utf-8")
    game = dict(V4_MINIMAL, game_id="g2")
    (round_dir / "g2.json").write_text(json.dumps(game) + "\n", encoding="utf-8")

    paths = list_game_paths(tmp_path)
    names = {p.name for p in paths}
    assert names == {"flat.json", "g2.json"}
    assert "manifest.json" not in names

    records = load_all_games(tmp_path)
    assert len(records) == 2


def test_list_game_paths_round_dir_non_recursive(tmp_path):
    round_dir = tmp_path / "roundY"
    nested = round_dir / "nested"
    nested.mkdir(parents=True)
    (round_dir / "manifest.json").write_text("{}", encoding="utf-8")
    (round_dir / "top.json").write_text(json.dumps(V4_MINIMAL) + "\n", encoding="utf-8")
    (nested / "deep.json").write_text(json.dumps(V4_MINIMAL) + "\n", encoding="utf-8")

    paths = list_game_paths(round_dir)
    assert [p.name for p in paths] == ["top.json"]


def test_save_game_round_folder(tmp_path):
    record = GameRecord.from_dict(V4_FULL)
    path = save_game(record, tmp_path / "roundZ")
    assert path.parent.name == "roundZ"
    assert path.exists()
