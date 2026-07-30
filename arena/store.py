"""Game record schema and IO for `data/games/<game_id>.json`."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal

REPO_ROOT = Path(__file__).resolve().parent.parent
GAMES_DIR = REPO_ROOT / "data" / "games"

Winner = Literal["a", "b", "draw"]

REQUIRED_FIELDS = (
    "game_id",
    "seed",
    "mode",
    "bot_a",
    "bot_b",
    "bot_a_commit_or_tag",
    "bot_b_commit_or_tag",
    "winner",
    "turns",
    "terminated",
    "truncated",
    "started_at",
    "finished_at",
)


@dataclass
class GameRecord:
    """One stored competition match (see docs/arena/game-record-schema.md)."""

    game_id: str
    seed: int
    mode: str
    bot_a: str
    bot_b: str
    bot_a_commit_or_tag: str
    bot_b_commit_or_tag: str
    winner: Winner
    turns: int
    terminated: bool
    truncated: bool
    started_at: str
    finished_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GameRecord:
        missing = [f for f in REQUIRED_FIELDS if f not in data]
        if missing:
            raise ValueError(f"game record missing fields: {missing}")
        winner = data["winner"]
        if winner not in ("a", "b", "draw"):
            raise ValueError(f"invalid winner: {winner!r}")
        return cls(
            game_id=str(data["game_id"]),
            seed=int(data["seed"]),
            mode=str(data["mode"]),
            bot_a=str(data["bot_a"]),
            bot_b=str(data["bot_b"]),
            bot_a_commit_or_tag=str(data["bot_a_commit_or_tag"]),
            bot_b_commit_or_tag=str(data["bot_b_commit_or_tag"]),
            winner=winner,
            turns=int(data["turns"]),
            terminated=bool(data["terminated"]),
            truncated=bool(data["truncated"]),
            started_at=str(data["started_at"]),
            finished_at=str(data["finished_at"]),
        )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def make_game_id(bot_a: str, bot_b: str, seed: int, when: datetime | None = None) -> str:
    """Build a filesystem-safe unique id."""
    stamp = (when or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    a = _slug(bot_a)
    b = _slug(bot_b)
    short = uuid.uuid4().hex[:8]
    return f"{stamp}_{a}_vs_{b}_s{seed}_{short}"


def _slug(label: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", label.strip()).strip("-")
    return cleaned or "bot"


def game_path(game_id: str, games_dir: Path | None = None) -> Path:
    return (games_dir or GAMES_DIR) / f"{game_id}.json"


def save_game(record: GameRecord, games_dir: Path | None = None) -> Path:
    """Write the game record. Call this before updating ratings."""
    directory = games_dir or GAMES_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = game_path(record.game_id, directory)
    path.write_text(json.dumps(record.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_game(path: Path) -> GameRecord:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"game record must be an object: {path}")
    return GameRecord.from_dict(data)


def list_game_paths(games_dir: Path | None = None) -> list[Path]:
    directory = games_dir or GAMES_DIR
    if not directory.exists():
        return []
    return sorted(directory.glob("*.json"))


def load_all_games(games_dir: Path | None = None) -> list[GameRecord]:
    records = [load_game(p) for p in list_game_paths(games_dir)]
    records.sort(key=lambda r: (r.finished_at, r.game_id))
    return records


def bot_id_from_run_sh(run_sh: Path) -> str:
    return run_sh.resolve().parent.name


def git_commit_or_tag(repo_root: Path | None = None) -> str:
    """Best-effort version pin for the workspace (HEAD short SHA)."""
    root = repo_root or REPO_ROOT
    try:
        import subprocess

        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return result.stdout.strip() or "unknown"
    except OSError:
        pass
    return "unknown"


def validate_record_dict(data: dict[str, Any]) -> None:
    """Raise ValueError if `data` is not a valid game record."""
    GameRecord.from_dict(data)


def iter_winners(records: Iterable[GameRecord]) -> list[Winner]:
    return [r.winner for r in records]
