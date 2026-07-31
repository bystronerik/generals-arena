"""Elo ratings via elote EloCompetitor. Store games first, then rate."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from elote import EloCompetitor

from arena.reporting import leaderboard_table_lines
from arena.store import (
    GAMES_DIR,
    REPO_ROOT,
    GameRecord,
    load_all_games,
    utc_now_iso,
)

RATINGS_DIR = REPO_ROOT / "data" / "ratings"
STATE_FILENAME = "competitors.json"
LEADERBOARD_JSON = "leaderboard.json"
LEADERBOARD_MD = "leaderboard.md"

# Readable Elo scale; elote's library default is 400.
INITIAL_RATING = 1500.0


@dataclass
class LeaderboardRow:
    rank: int
    bot_id: str
    rating: float
    games: int
    wins: int
    losses: int
    draws: int


class RatingBook:
    """Explicit Elo book: one EloCompetitor per bot id."""

    def __init__(
        self,
        initial_rating: float = INITIAL_RATING,
        competitors: dict[str, EloCompetitor] | None = None,
        games_played: dict[str, dict[str, int]] | None = None,
        rated_game_ids: set[str] | None = None,
    ) -> None:
        self.initial_rating = initial_rating
        self.competitors: dict[str, EloCompetitor] = competitors or {}
        self.games_played: dict[str, dict[str, int]] = games_played or {}
        self.rated_game_ids: set[str] = rated_game_ids or set()

    def get(self, bot_id: str) -> EloCompetitor:
        if bot_id not in self.competitors:
            self.competitors[bot_id] = EloCompetitor(initial_rating=self.initial_rating)
            self.games_played[bot_id] = {"wins": 0, "losses": 0, "draws": 0}
        return self.competitors[bot_id]

    def apply_game(self, record: GameRecord, *, skip_if_rated: bool = True) -> bool:
        """Apply one stored game. Returns False if already rated and skip_if_rated."""
        if skip_if_rated and record.game_id in self.rated_game_ids:
            return False
        a = self.get(record.bot_a)
        b = self.get(record.bot_b)
        if record.winner == "a":
            a.beat(b)
            self.games_played[record.bot_a]["wins"] += 1
            self.games_played[record.bot_b]["losses"] += 1
        elif record.winner == "b":
            a.lost_to(b)
            self.games_played[record.bot_a]["losses"] += 1
            self.games_played[record.bot_b]["wins"] += 1
        elif record.winner == "draw":
            a.tied(b)
            self.games_played[record.bot_a]["draws"] += 1
            self.games_played[record.bot_b]["draws"] += 1
        else:
            raise ValueError(f"invalid winner: {record.winner!r}")
        self.rated_game_ids.add(record.game_id)
        return True

    def apply_games(self, records: list[GameRecord], *, skip_if_rated: bool = True) -> int:
        applied = 0
        for record in records:
            if self.apply_game(record, skip_if_rated=skip_if_rated):
                applied += 1
        return applied

    def leaderboard(self) -> list[LeaderboardRow]:
        rows: list[LeaderboardRow] = []
        for bot_id, competitor in self.competitors.items():
            stats = self.games_played.get(bot_id, {"wins": 0, "losses": 0, "draws": 0})
            games = stats["wins"] + stats["losses"] + stats["draws"]
            rows.append(
                LeaderboardRow(
                    rank=0,
                    bot_id=bot_id,
                    rating=float(competitor.rating),
                    games=games,
                    wins=stats["wins"],
                    losses=stats["losses"],
                    draws=stats["draws"],
                )
            )
        rows.sort(key=lambda r: (-r.rating, r.bot_id))
        for i, row in enumerate(rows, start=1):
            row.rank = i
        return rows

    def to_state(self) -> dict[str, Any]:
        return {
            "version": 1,
            "initial_rating": self.initial_rating,
            "updated_at": utc_now_iso(),
            "rated_game_ids": sorted(self.rated_game_ids),
            "games_played": self.games_played,
            "competitors": {
                bot_id: competitor.export_state()
                for bot_id, competitor in sorted(self.competitors.items())
            },
        }

    @classmethod
    def from_state(cls, data: dict[str, Any]) -> RatingBook:
        initial = float(data.get("initial_rating", INITIAL_RATING))
        competitors: dict[str, EloCompetitor] = {}
        for bot_id, state in data.get("competitors", {}).items():
            competitors[bot_id] = EloCompetitor.from_state(state)
        games_played = {
            bot_id: {
                "wins": int(stats.get("wins", 0)),
                "losses": int(stats.get("losses", 0)),
                "draws": int(stats.get("draws", 0)),
            }
            for bot_id, stats in data.get("games_played", {}).items()
        }
        rated = set(data.get("rated_game_ids", []))
        return cls(
            initial_rating=initial,
            competitors=competitors,
            games_played=games_played,
            rated_game_ids=rated,
        )


def state_path(ratings_dir: Path | None = None) -> Path:
    return (ratings_dir or RATINGS_DIR) / STATE_FILENAME


def load_book(ratings_dir: Path | None = None) -> RatingBook:
    path = state_path(ratings_dir)
    if not path.exists():
        return RatingBook()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"ratings state must be an object: {path}")
    return RatingBook.from_state(data)


def save_book(book: RatingBook, ratings_dir: Path | None = None) -> Path:
    directory = ratings_dir or RATINGS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = state_path(directory)
    path.write_text(json.dumps(book.to_state(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_leaderboard(book: RatingBook, ratings_dir: Path | None = None) -> tuple[Path, Path]:
    directory = ratings_dir or RATINGS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    rows = book.leaderboard()
    payload = {
        "updated_at": utc_now_iso(),
        "initial_rating": book.initial_rating,
        "rated_games": len(book.rated_game_ids),
        "bots": [
            {
                "rank": r.rank,
                "bot_id": r.bot_id,
                "rating": round(r.rating, 2),
                "games": r.games,
                "wins": r.wins,
                "losses": r.losses,
                "draws": r.draws,
            }
            for r in rows
        ],
    }
    json_path = directory / LEADERBOARD_JSON
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Arena leaderboard",
        "",
        f"Updated: {payload['updated_at']}",
        f"Rated games: {payload['rated_games']}",
        f"Initial Elo: {payload['initial_rating']}",
        "",
        *leaderboard_table_lines(rows),
        "",
    ]
    md_path = directory / LEADERBOARD_MD
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def rate_stored_game(
    record: GameRecord,
    *,
    ratings_dir: Path | None = None,
    persist: bool = True,
) -> RatingBook:
    """Apply one already-stored game, then optionally persist + leaderboard."""
    book = load_book(ratings_dir)
    book.apply_game(record)
    if persist:
        save_book(book, ratings_dir)
        write_leaderboard(book, ratings_dir)
    return book


def rebuild_from_games(
    *,
    games_dir: Path | None = None,
    ratings_dir: Path | None = None,
    initial_rating: float = INITIAL_RATING,
) -> RatingBook:
    """Rebuild ratings from every game under data/games/ (chrono order)."""
    book = RatingBook(initial_rating=initial_rating)
    book.apply_games(load_all_games(games_dir or GAMES_DIR), skip_if_rated=False)
    save_book(book, ratings_dir)
    write_leaderboard(book, ratings_dir)
    return book


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Rebuild Elo leaderboard from stored data/games/ records."
    )
    parser.add_argument(
        "--games-dir",
        type=Path,
        default=GAMES_DIR,
        help=f"directory of game JSON files (default: {GAMES_DIR})",
    )
    parser.add_argument(
        "--ratings-dir",
        type=Path,
        default=RATINGS_DIR,
        help=f"directory for ratings snapshots (default: {RATINGS_DIR})",
    )
    parser.add_argument(
        "--initial-rating",
        type=float,
        default=INITIAL_RATING,
        help=f"Elo for unseen bots when rebuilding (default: {INITIAL_RATING})",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        dest="print_table",
        help="print the Markdown leaderboard to stdout",
    )
    args = parser.parse_args(argv)

    book = rebuild_from_games(
        games_dir=args.games_dir,
        ratings_dir=args.ratings_dir,
        initial_rating=args.initial_rating,
    )
    json_path = args.ratings_dir / LEADERBOARD_JSON
    md_path = args.ratings_dir / LEADERBOARD_MD
    print(f"[ratings] rated {len(book.rated_game_ids)} game(s)")
    print(f"[ratings] wrote {json_path}")
    print(f"[ratings] wrote {md_path}")
    if args.print_table:
        print(md_path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
