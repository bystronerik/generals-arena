"""GameRecord v5 round-trip, required identity fields, and path scanning."""
from __future__ import annotations

import json

import pytest

from arena.records.store import (
    CURRENT_SCHEMA_VERSION,
    GameRecord,
    list_game_paths,
    round_games_dir,
    save_game,
)


V5_MINIMAL = {
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
    "truncated": False,
    "schema_version": 5,
}

# Everything observational is in `metrics` at v5 — including the engine-truth
# finals and castle tallies that used to be top-level fields.
V5_FULL = {
    **V5_MINIMAL,
    "metrics": {
        "castles_built_a": 2,
        "castles_built_b": 1,
        "final_land_a": 50,
        "final_land_b": 30,
        "final_army_a": 100,
        "final_army_b": 80,
        "enemy_general_sighted_a": True,
    },
}


@pytest.mark.parametrize("payload", [V5_MINIMAL, V5_FULL], ids=["minimal", "full"])
def test_game_record_round_trip(payload):
    record = GameRecord.from_dict(payload)
    restored = GameRecord.from_dict(record.to_dict())
    assert restored == record


def test_minimal_defaults_optional_fields():
    record = GameRecord.from_dict(V5_MINIMAL)
    assert record.schema_version == CURRENT_SCHEMA_VERSION
    assert record.metrics == {}


def test_full_preserves_observational_metrics():
    record = GameRecord.from_dict(V5_FULL)
    assert record.metrics["castles_built_a"] == 2
    assert record.metrics["final_army_b"] == 80
    assert record.metrics["enemy_general_sighted_a"] is True


def test_terminated_is_derived_not_stored():
    """`terminated` was exactly `winner != "draw"`, so v5 computes it."""
    assert GameRecord.from_dict(V5_MINIMAL).terminated is True
    drawn = GameRecord.from_dict(dict(V5_MINIMAL, winner="draw", truncated=True))
    assert drawn.terminated is False
    assert "terminated" not in drawn.to_dict()


def test_truncated_survives_because_winner_cannot_replace_it():
    """A stalled draw and a simultaneous capture differ only here."""
    stalled = GameRecord.from_dict(dict(V5_MINIMAL, winner="draw", truncated=True))
    mutual_kill = GameRecord.from_dict(dict(V5_MINIMAL, winner="draw", truncated=False))
    assert stalled.winner == mutual_kill.winner
    assert stalled.truncated != mutual_kill.truncated


def test_dropped_fields_are_not_stored():
    """v4 fields with no readers must not reappear through the round trip."""
    stored = GameRecord.from_dict(V5_FULL).to_dict()
    for field in ("started_at", "finished_at", "duration_seconds", "terminated"):
        assert field not in stored


def test_identity_fields_survive_the_round_trip():
    record = GameRecord.from_dict(V5_MINIMAL)
    assert record.bot_a_content_hash == "0123456789ab"
    assert record.bot_b_content_hash == "ba9876543210"
    assert record.engine_version.startswith("9e3b9d1")
    assert record.round == "round5"


@pytest.mark.parametrize("version", [1, 2, 3, 4])
def test_pre_v5_records_are_rejected_loudly(version):
    """The stored pool was projected once; there is no dual-version reader."""
    stale = dict(V5_MINIMAL, schema_version=version)
    with pytest.raises(ValueError, match="not readable"):
        GameRecord.from_dict(stale)


@pytest.mark.parametrize(
    "field", ["winner", "round", "engine_version", "bot_a_content_hash", "truncated"]
)
def test_from_dict_rejects_missing_required(field):
    bad = dict(V5_MINIMAL)
    del bad[field]
    with pytest.raises(ValueError, match="missing fields"):
        GameRecord.from_dict(bad)


def test_from_dict_rejects_invalid_winner():
    bad = dict(V5_MINIMAL, winner="x")
    with pytest.raises(ValueError, match="invalid winner"):
        GameRecord.from_dict(bad)


def test_from_dict_rejects_the_unknown_hash_sentinel():
    """`"unknown"` would pool every unreadable closure into one rated entity."""
    bad = dict(V5_MINIMAL, bot_b_content_hash="unknown")
    with pytest.raises(ValueError, match="unknown"):
        GameRecord.from_dict(bad)


def test_from_dict_rejects_an_empty_hash():
    bad = dict(V5_MINIMAL, bot_a_content_hash="")
    with pytest.raises(ValueError, match="non-empty"):
        GameRecord.from_dict(bad)


def test_round_games_dir(tmp_path):
    path = round_games_dir("round3", games_root=tmp_path)
    assert path == tmp_path / "round3"


def test_round_games_dir_rejects_bad_name():
    with pytest.raises(ValueError, match="invalid round name"):
        round_games_dir("../escape")


def test_list_game_paths_skips_manifest_and_subdirectories(tmp_path):
    """One directory at a time — the rating layer walks rounds itself."""
    round_dir = tmp_path / "roundY"
    nested = round_dir / "nested"
    nested.mkdir(parents=True)
    (round_dir / "manifest.json").write_text("{}", encoding="utf-8")
    (round_dir / "top.json").write_text(json.dumps(V5_MINIMAL) + "\n", encoding="utf-8")
    (nested / "deep.json").write_text(json.dumps(V5_MINIMAL) + "\n", encoding="utf-8")

    assert [p.name for p in list_game_paths(round_dir)] == ["top.json"]


def test_list_game_paths_on_a_missing_directory_is_empty(tmp_path):
    assert list_game_paths(tmp_path / "never-ran") == []


def test_save_game_round_folder(tmp_path):
    record = GameRecord.from_dict(V5_FULL)
    path = save_game(record, tmp_path / "roundZ")
    assert path.parent.name == "roundZ"
    assert path.exists()
