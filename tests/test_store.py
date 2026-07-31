"""GameRecord round-trip and schema migration tests."""
from __future__ import annotations

import json

import pytest

from arena.store import (
    GameRecord,
    list_game_paths,
    load_all_games,
    round_games_dir,
    save_game,
)


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


V3_FULL = {
    **V2_FULL,
    "schema_version": 3,
    "bot_a_content_hash": "0123456789ab",
    "bot_b_content_hash": "ba9876543210",
}


@pytest.mark.parametrize(
    "payload", [V1_MINIMAL, V2_FULL, V3_FULL], ids=["v1", "v2", "v3"]
)
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


def test_pre_v3_records_have_no_content_hash():
    """The 7k+ games stored before schema v3 must still load, hashes absent."""
    for payload in (V1_MINIMAL, V2_FULL):
        record = GameRecord.from_dict(payload)
        assert record.bot_a_content_hash is None
        assert record.bot_b_content_hash is None


def test_v3_preserves_content_hashes():
    record = GameRecord.from_dict(V3_FULL)
    assert record.schema_version == 3
    assert record.bot_a_content_hash == "0123456789ab"
    assert record.bot_b_content_hash == "ba9876543210"


def test_from_dict_rejects_missing_required():
    bad = dict(V1_MINIMAL)
    del bad["winner"]
    with pytest.raises(ValueError, match="missing fields"):
        GameRecord.from_dict(bad)


def test_from_dict_rejects_invalid_winner():
    bad = dict(V1_MINIMAL, winner="x")
    with pytest.raises(ValueError, match="invalid winner"):
        GameRecord.from_dict(bad)


def test_round_games_dir(tmp_path):
    path = round_games_dir("round3", games_root=tmp_path)
    assert path == tmp_path / "round3"


def test_round_games_dir_rejects_bad_name():
    with pytest.raises(ValueError, match="invalid round name"):
        round_games_dir("../escape")


def test_list_game_paths_recursive_skips_manifest(monkeypatch, tmp_path):
    monkeypatch.setattr("arena.store.GAMES_DIR", tmp_path)
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps(V1_MINIMAL) + "\n", encoding="utf-8")
    round_dir = tmp_path / "roundX"
    round_dir.mkdir()
    (round_dir / "manifest.json").write_text("{}", encoding="utf-8")
    game = dict(V1_MINIMAL, game_id="g2")
    (round_dir / "g2.json").write_text(json.dumps(game) + "\n", encoding="utf-8")

    paths = list_game_paths(tmp_path)
    names = {p.name for p in paths}
    assert names == {"legacy.json", "g2.json"}
    assert "manifest.json" not in names

    records = load_all_games(tmp_path)
    assert len(records) == 2


def test_list_game_paths_round_dir_non_recursive(tmp_path):
    round_dir = tmp_path / "roundY"
    nested = round_dir / "nested"
    nested.mkdir(parents=True)
    (round_dir / "manifest.json").write_text("{}", encoding="utf-8")
    (round_dir / "top.json").write_text(json.dumps(V1_MINIMAL) + "\n", encoding="utf-8")
    (nested / "deep.json").write_text(json.dumps(V1_MINIMAL) + "\n", encoding="utf-8")

    paths = list_game_paths(round_dir)
    assert [p.name for p in paths] == ["top.json"]


def test_save_game_round_folder(tmp_path):
    record = GameRecord.from_dict(V2_FULL)
    path = save_game(record, tmp_path / "roundZ")
    assert path.parent.name == "roundZ"
    assert path.exists()
