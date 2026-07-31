"""RatingBook.apply_game idempotence and basic Elo updates."""
from __future__ import annotations

from arena.records.ratings import INITIAL_RATING, RatingBook
from arena.records.store import GameRecord


def _record(game_id: str, winner: str = "a") -> GameRecord:
    return GameRecord(
        game_id=game_id,
        seed=0,
        mode="competition",
        round="roundT",
        bot_a="smoke",
        bot_b="rush",
        bot_a_content_hash="0123456789ab",
        bot_b_content_hash="ba9876543210",
        engine_version="9e3b9d1",
        winner=winner,  # type: ignore[arg-type]
        turns=10,
        terminated=True,
        truncated=False,
        started_at="2026-01-01T00:00:00Z",
        finished_at="2026-01-01T00:01:00Z",
    )


def test_apply_game_idempotent():
    book = RatingBook()
    record = _record("g1")
    rating_a_before = book.get("smoke").rating

    assert book.apply_game(record) is True
    rating_a_after_first = book.get("smoke").rating
    assert rating_a_after_first > rating_a_before

    assert book.apply_game(record) is False
    assert book.get("smoke").rating == rating_a_after_first
    assert book.games_played["smoke"]["wins"] == 1


def test_apply_game_draw_updates_both():
    book = RatingBook()
    record = _record("g2", winner="draw")
    assert book.apply_game(record) is True
    assert book.games_played["smoke"]["draws"] == 1
    assert book.games_played["rush"]["draws"] == 1
    assert book.get("smoke").rating == INITIAL_RATING


def test_apply_game_skip_if_rated_false():
    book = RatingBook()
    record = _record("g3")
    book.apply_game(record)
    rating = book.get("smoke").rating
    assert book.apply_game(record, skip_if_rated=False) is True
    assert book.get("smoke").rating != rating
