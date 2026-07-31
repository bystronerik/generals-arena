"""N seeds × bot pairs under the classic harness; store under data/classic_games/."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from arena.matches.classic import CLASSIC_ENV_DEFAULTS, run_classic_match
from arena.matches.loop import winner_seat
from arena.records.store import (
    REPO_ROOT,
    Winner,
    bot_id_from_run_sh,
    coerce_metrics,
    duration_seconds_between,
    git_commit_or_tag,
    make_game_id,
    optional_float,
    optional_int,
    read_record_json,
    record_path,
    utc_now_iso,
    write_record_json,
)
from arena.tournaments.competition import bot_pairs, parse_seeds
from arena.tournaments.parallel import cap_jobs, run_pool

CLASSIC_GAMES_DIR = REPO_ROOT / "data" / "classic_games"


@dataclass
class ClassicMatchSpec:
    seed: int
    bot_a_run: Path
    bot_b_run: Path


@dataclass
class ClassicGameRecord:
    """One stored classic-approximate match (never enters data/games/ or the fit)."""

    game_id: str
    seed: int
    mode: Literal["classic"]
    bot_a: str
    bot_b: str
    bot_a_commit_or_tag: str
    bot_b_commit_or_tag: str
    winner: Winner
    winner_player_id: int
    turns: int
    terminated: bool
    truncated: bool
    started_at: str
    finished_at: str
    schema_version: int = 1
    duration_seconds: float | None = None
    grid_size: int | None = None
    truncation_limit: int | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if not data.get("metrics"):
            data["metrics"] = {}
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ClassicGameRecord:
        winner = data["winner"]
        if winner not in ("a", "b", "draw"):
            raise ValueError(f"invalid winner: {winner!r}")
        metrics = coerce_metrics(data.get("metrics"))
        return cls(
            game_id=str(data["game_id"]),
            seed=int(data["seed"]),
            mode="classic",
            bot_a=str(data["bot_a"]),
            bot_b=str(data["bot_b"]),
            bot_a_commit_or_tag=str(data["bot_a_commit_or_tag"]),
            bot_b_commit_or_tag=str(data["bot_b_commit_or_tag"]),
            winner=winner,
            winner_player_id=int(data["winner_player_id"]),
            turns=int(data["turns"]),
            terminated=bool(data["terminated"]),
            truncated=bool(data["truncated"]),
            started_at=str(data["started_at"]),
            finished_at=str(data["finished_at"]),
            schema_version=int(data.get("schema_version", 1)),
            duration_seconds=optional_float(data.get("duration_seconds")),
            grid_size=optional_int(data.get("grid_size")),
            truncation_limit=optional_int(data.get("truncation_limit")),
            metrics=metrics,
        )


def build_match_specs(
    run_scripts: list[Path],
    seeds: list[int],
    *,
    include_self: bool = False,
    swap_sides: bool = False,
) -> list[ClassicMatchSpec]:
    """Expand seeds × unordered bot pairs (optional mirror and self-play)."""
    pairs = bot_pairs(run_scripts, include_self=include_self)
    if swap_sides:
        mirrored = [(b, a) for a, b in pairs if a.resolve() != b.resolve()]
        pairs = pairs + mirrored
    return [
        ClassicMatchSpec(seed=seed, bot_a_run=a, bot_b_run=b)
        for seed in seeds
        for a, b in pairs
    ]


def classic_game_path(game_id: str, games_dir: Path | None = None) -> Path:
    return record_path(game_id, games_dir or CLASSIC_GAMES_DIR)


def save_classic_game(record: ClassicGameRecord, games_dir: Path | None = None) -> Path:
    return write_record_json(record.to_dict(), classic_game_path(record.game_id, games_dir))


def load_classic_game(path: Path) -> ClassicGameRecord:
    return ClassicGameRecord.from_dict(
        read_record_json(path, label="classic game record")
    )


def _env_overrides(grid_size: int | None, truncation: int | None) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if grid_size is not None:
        overrides["grid_dims"] = (grid_size, grid_size)
    if truncation is not None:
        overrides["truncation"] = truncation
    return overrides


def run_one_classic(
    spec: ClassicMatchSpec,
    *,
    games_dir: Path | None = None,
    grid_size: int | None = None,
    truncation: int | None = None,
    commit: str | None = None,
) -> ClassicGameRecord:
    """Run one classic match and store the JSON record."""
    a_path = spec.bot_a_run.resolve()
    b_path = spec.bot_b_run.resolve()
    bot_a = bot_id_from_run_sh(a_path)
    bot_b = bot_id_from_run_sh(b_path)
    pin = commit or git_commit_or_tag()

    started_at = utc_now_iso()
    overrides = _env_overrides(grid_size, truncation)
    winner_player_id, turns, truncated = run_classic_match(
        a_path,
        b_path,
        seed=spec.seed,
        env_overrides=overrides or None,
    )
    finished_at = utc_now_iso()
    winner = winner_seat(winner_player_id, truncated=truncated)
    terminated = winner_player_id >= 0

    record = ClassicGameRecord(
        game_id=make_game_id(bot_a, bot_b, spec.seed),
        seed=spec.seed,
        mode="classic",
        bot_a=bot_a,
        bot_b=bot_b,
        bot_a_commit_or_tag=pin,
        bot_b_commit_or_tag=pin,
        winner=winner,
        winner_player_id=winner_player_id,
        turns=turns,
        terminated=terminated,
        truncated=truncated,
        started_at=started_at,
        finished_at=finished_at,
        duration_seconds=duration_seconds_between(started_at, finished_at),
        grid_size=grid_size or CLASSIC_ENV_DEFAULTS["grid_dims"][0],
        truncation_limit=truncation or CLASSIC_ENV_DEFAULTS["truncation"],
    )
    save_classic_game(record, games_dir)
    return record


def _run_one_classic_worker(payload: dict[str, Any]) -> ClassicGameRecord:
    spec = ClassicMatchSpec(
        seed=int(payload["seed"]),
        bot_a_run=Path(payload["bot_a_run"]),
        bot_b_run=Path(payload["bot_b_run"]),
    )
    return run_one_classic(
        spec,
        games_dir=Path(payload["games_dir"]) if payload.get("games_dir") else None,
        grid_size=payload.get("grid_size"),
        truncation=payload.get("truncation"),
        commit=payload.get("commit"),
    )


def run_classic_tournament(
    run_scripts: list[Path],
    seeds: list[int],
    *,
    games_dir: Path | None = None,
    include_self: bool = False,
    swap_sides: bool = False,
    grid_size: int | None = None,
    truncation: int | None = None,
    jobs: int = 1,
) -> list[ClassicGameRecord]:
    """Run every seed × pair; store each game under data/classic_games/."""
    specs = build_match_specs(
        run_scripts,
        seeds,
        include_self=include_self,
        swap_sides=swap_sides,
    )
    directory = games_dir or CLASSIC_GAMES_DIR
    commit = git_commit_or_tag()
    worker_jobs = cap_jobs(jobs)
    total = len(specs)

    print(f"[classic_tournament] running {total} match(es) with jobs={worker_jobs}")
    payloads = [
        {
            "seed": spec.seed,
            "bot_a_run": str(spec.bot_a_run.resolve()),
            "bot_b_run": str(spec.bot_b_run.resolve()),
            "games_dir": str(directory),
            "grid_size": grid_size,
            "truncation": truncation,
            "commit": commit,
        }
        for spec in specs
    ]

    def _on_result(done: int, total: int, record: ClassicGameRecord) -> None:
        print(
            f"[classic_tournament] ({done}/{total}) {record.bot_a} vs {record.bot_b} "
            f"seed={record.seed} -> {record.winner} turns={record.turns} "
            f"terminated={record.terminated} truncated={record.truncated} "
            f"game_id={record.game_id}"
        )

    records = run_pool(
        payloads,
        _run_one_classic_worker,
        jobs=worker_jobs,
        on_result=_on_result,
    )
    records.sort(key=lambda r: (r.seed, r.bot_a, r.bot_b, r.game_id))
    print(f"[classic_tournament] finished {len(records)} game(s)")
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run N seeds × bot pairs under the classic harness; "
            "store JSON under data/classic_games/ (never rated)."
        )
    )
    parser.add_argument(
        "bots",
        nargs="+",
        type=Path,
        help="two or more bot run.sh paths",
    )
    parser.add_argument(
        "--seeds",
        default="0",
        help="comma list and/or ranges, e.g. 0-3,10 (default: 0)",
    )
    parser.add_argument(
        "--games-dir",
        type=Path,
        default=CLASSIC_GAMES_DIR,
        help=f"classic game JSON directory (default: {CLASSIC_GAMES_DIR})",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=None,
        help=f"square board side (default: {CLASSIC_ENV_DEFAULTS['grid_dims'][0]})",
    )
    parser.add_argument(
        "--truncation",
        type=int,
        default=None,
        help=f"max turns (default: {CLASSIC_ENV_DEFAULTS['truncation']})",
    )
    parser.add_argument(
        "--include-self",
        action="store_true",
        help="also play each bot against itself (and all ordered pairs)",
    )
    parser.add_argument(
        "--swap-sides",
        action="store_true",
        help="also play each pair with sides swapped",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="parallel match workers (default: 1)",
    )
    args = parser.parse_args(argv)

    if args.jobs < 1:
        parser.error("--jobs must be >= 1")

    seeds = parse_seeds(args.seeds)
    run_classic_tournament(
        args.bots,
        seeds,
        games_dir=args.games_dir,
        include_self=args.include_self,
        swap_sides=args.swap_sides,
        grid_size=args.grid_size,
        truncation=args.truncation,
        jobs=args.jobs,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
