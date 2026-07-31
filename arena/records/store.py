"""Game record schema and IO for `data/games/<round>/<game_id>.json`."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from arena.paths import REPO_ROOT

GAMES_DIR = REPO_ROOT / "data" / "games"

_ROUND_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def round_games_dir(round_name: str, *, games_root: Path | None = None) -> Path:
    """Return `data/games/<round_name>/` for a measurement or tournament round."""
    name = round_name.strip()
    if not name or not _ROUND_NAME_RE.match(name):
        raise ValueError(
            f"invalid round name {round_name!r}; use letters, digits, . _ -"
        )
    return (games_root or GAMES_DIR) / name

Winner = Literal["a", "b", "draw"]

# Schema version stamped on records this code writes. v4 is the first version
# the rating layer will read: it makes both content hashes required (rating
# identity must never silently degrade) and adds `engine_version` and `round`.
# There is no v3 reader — every pre-v4 record was deleted, not migrated.
# See docs/arena/game-record-schema.md.
CURRENT_SCHEMA_VERSION = 4
MIN_SCHEMA_VERSION = 4

REQUIRED_FIELDS = (
    "game_id",
    "seed",
    "mode",
    "round",
    "bot_a",
    "bot_b",
    "bot_a_content_hash",
    "bot_b_content_hash",
    "engine_version",
    "winner",
    "turns",
    "terminated",
    "truncated",
    "started_at",
    "finished_at",
)

OPTIONAL_FIELDS = (
    "schema_version",
    "duration_seconds",
    "castles_built_a",
    "castles_built_b",
    "final_land_a",
    "final_land_b",
    "final_army_a",
    "final_army_b",
    "metrics",
)

# A hash that names no program. `fingerprint` used to return this on any
# OSError; under hash-keyed identity it would pool unrelated programs into one
# rated entity, so it is rejected at both the write and the read boundary.
UNKNOWN_HASH = "unknown"


@dataclass
class GameRecord:
    """One stored competition match (see docs/arena/game-record-schema.md)."""

    game_id: str
    seed: int
    mode: str
    # Round name, stored rather than inferred: eligibility must not depend on
    # parsing the path a record happens to sit at.
    round: str
    bot_a: str
    bot_b: str
    # Hash of each bot's source closure (arena/records/fingerprint.py). This is
    # the rating identity — not the repo-wide commit pin, which moves whenever
    # anything is committed and stays put when a bot is edited.
    bot_a_content_hash: str
    bot_b_content_hash: str
    # `competition-module` submodule SHA. A rules or engine change moves win
    # probabilities, so records from two eras must never pool.
    engine_version: str
    winner: Winner
    turns: int
    terminated: bool
    truncated: bool
    started_at: str
    finished_at: str
    schema_version: int = CURRENT_SCHEMA_VERSION
    duration_seconds: float | None = None
    castles_built_a: int | None = None
    castles_built_b: int | None = None
    final_land_a: int | None = None
    final_land_b: int | None = None
    final_army_a: int | None = None
    final_army_b: int | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Validate at construction, not only at load: the identity fields are
        # what the fit keys on, so there must be no way to build a record that
        # cannot be rated — including through the dataclass constructor.
        self.bot_a_content_hash = require_content_hash(self.bot_a_content_hash, "bot_a")
        self.bot_b_content_hash = require_content_hash(self.bot_b_content_hash, "bot_b")
        self.engine_version = require_non_empty(self.engine_version, "engine_version")
        self.round = require_non_empty(self.round, "round")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if not data.get("metrics"):
            data["metrics"] = {}
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GameRecord:
        version = int(data.get("schema_version", 1))
        if version < MIN_SCHEMA_VERSION:
            raise ValueError(
                f"game record schema v{version} is not readable; "
                f"v{MIN_SCHEMA_VERSION} is the minimum (no pre-v4 record survives)"
            )
        missing = [f for f in REQUIRED_FIELDS if f not in data]
        if missing:
            raise ValueError(f"game record missing fields: {missing}")
        winner = data["winner"]
        if winner not in ("a", "b", "draw"):
            raise ValueError(f"invalid winner: {winner!r}")
        metrics = coerce_metrics(data.get("metrics"))
        return cls(
            game_id=str(data["game_id"]),
            seed=int(data["seed"]),
            mode=str(data["mode"]),
            round=str(data["round"]),
            bot_a=str(data["bot_a"]),
            bot_b=str(data["bot_b"]),
            bot_a_content_hash=str(data["bot_a_content_hash"]),
            bot_b_content_hash=str(data["bot_b_content_hash"]),
            engine_version=str(data["engine_version"]),
            winner=winner,
            turns=int(data["turns"]),
            terminated=bool(data["terminated"]),
            truncated=bool(data["truncated"]),
            started_at=str(data["started_at"]),
            finished_at=str(data["finished_at"]),
            schema_version=version,
            duration_seconds=optional_float(data.get("duration_seconds")),
            castles_built_a=optional_int(data.get("castles_built_a")),
            castles_built_b=optional_int(data.get("castles_built_b")),
            final_land_a=optional_int(data.get("final_land_a")),
            final_land_b=optional_int(data.get("final_land_b")),
            final_army_a=optional_int(data.get("final_army_a")),
            final_army_b=optional_int(data.get("final_army_b")),
            metrics=metrics,
        )


def optional_int(value: Any) -> int | None:
    """Coerce an optional JSON field to int, keeping None as None."""
    if value is None:
        return None
    return int(value)


def optional_float(value: Any) -> float | None:
    """Coerce an optional JSON field to float, keeping None as None."""
    if value is None:
        return None
    return float(value)


def optional_str(value: Any) -> str | None:
    """Coerce an optional JSON field to str, keeping None as None."""
    if value is None:
        return None
    return str(value)


def require_non_empty(value: Any, field: str) -> str:
    """Coerce a required string field, rejecting None and empty."""
    if value is None or not str(value).strip():
        raise ValueError(f"{field} must be a non-empty string")
    return str(value)


def require_content_hash(value: Any, side: str) -> str:
    """
    Coerce a required content hash, rejecting the `"unknown"` sentinel.

    An `"unknown"` hash would pool every unreadable closure into one rated
    entity — a silent identity collision between unrelated programs.
    """
    digest = require_non_empty(value, f"{side}_content_hash")
    if digest == UNKNOWN_HASH:
        raise ValueError(
            f"{side}_content_hash is {UNKNOWN_HASH!r}; a record with no readable "
            f"bot closure has no rating identity and must not be stored"
        )
    return digest


def coerce_metrics(value: Any) -> dict[str, Any]:
    """Validate a record's optional `metrics` object, defaulting to empty."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("metrics must be an object when present")
    return dict(value)


def record_path(game_id: str, directory: Path) -> Path:
    return directory / f"{game_id}.json"


def write_record_json(payload: dict[str, Any], path: Path) -> Path:
    """Write one record as sorted, indented JSON, creating parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def read_record_json(path: Path, *, label: str = "record") -> dict[str, Any]:
    """Read one record JSON file, requiring a top-level object."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{label} must be an object: {path}")
    return data


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def duration_seconds_between(started_at: str, finished_at: str) -> float:
    start = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
    end = datetime.fromisoformat(finished_at.replace("Z", "+00:00"))
    return round((end - start).total_seconds(), 3)


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
    return record_path(game_id, games_dir or GAMES_DIR)


def save_game(record: GameRecord, games_dir: Path | None = None) -> Path:
    """Write the game record. Call this before updating ratings."""
    return write_record_json(record.to_dict(), game_path(record.game_id, games_dir))


def load_game(path: Path) -> GameRecord:
    return GameRecord.from_dict(read_record_json(path, label="game record"))


def list_game_paths(games_dir: Path | None = None) -> list[Path]:
    """
    List game JSON files under `games_dir`.

    When `games_dir` is the arena root `data/games/` (or omitted), scan
    recursively so per-round subfolders are included. Legacy flat JSON at the
    root still loads. When `games_dir` is a round folder, scan that folder only
    (non-recursive), skipping `manifest.json`.
    """
    directory = games_dir or GAMES_DIR
    if not directory.exists():
        return []
    root = GAMES_DIR.resolve()
    try:
        is_games_root = directory.resolve() == root
    except OSError:
        is_games_root = False

    if is_games_root:
        paths = [
            p
            for p in directory.rglob("*.json")
            if p.is_file() and p.name != "manifest.json"
        ]
    else:
        paths = [
            p
            for p in directory.glob("*.json")
            if p.is_file() and p.name != "manifest.json"
        ]
    return sorted(paths)


def load_all_games(games_dir: Path | None = None) -> list[GameRecord]:
    records = [load_game(p) for p in list_game_paths(games_dir)]
    records.sort(key=lambda r: (r.finished_at, r.game_id))
    return records


def bot_id_from_run_sh(run_sh: Path) -> str:
    return run_sh.resolve().parent.name


def git_head_sha(*, short: bool = True, repo_root: Path | None = None) -> str | None:
    """HEAD SHA for the workspace, or None when git is unavailable."""
    root = repo_root or REPO_ROOT
    cmd = ["git", "-C", str(root), "rev-parse"]
    if short:
        cmd.append("--short")
    cmd.append("HEAD")
    try:
        import subprocess

        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def git_commit_or_tag(repo_root: Path | None = None) -> str:
    """Best-effort version pin for classic records: HEAD short SHA or 'unknown'."""
    return git_head_sha(short=True, repo_root=repo_root) or "unknown"


ENGINE_SUBMODULE = "competition-module"


@lru_cache(maxsize=None)
def engine_version(repo_root: str | None = None) -> str:
    """
    SHA of the checked-out `competition-module`, stamped on every game record.

    The submodule is the engine: a bump moves win probabilities, so ratings
    from either side of one must never pool (plan §9 q2). Read from the
    submodule's own HEAD rather than the gitlink in the superproject's tree,
    because it is the checkout that actually played the game.
    """
    root = Path(repo_root) if repo_root else REPO_ROOT
    sha = git_head_sha(short=False, repo_root=root / ENGINE_SUBMODULE)
    if not sha:
        raise ValueError(
            f"cannot read {ENGINE_SUBMODULE} HEAD under {root}; every game record "
            f"needs an engine_version to keep eras from pooling"
        )
    return sha
