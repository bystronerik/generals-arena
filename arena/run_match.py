"""Run one in-process competition match and store the game record."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from arena.competition_match import run_competition_match
from arena.store import (
    GAMES_DIR,
    GameRecord,
    bot_id_from_run_sh,
    git_commit_or_tag,
    save_game,
    utc_now_iso,
)
from arena.telemetry import record_from_match_result


def run_and_store(
    bot_a_run: Path,
    bot_b_run: Path,
    *,
    seed: int = 0,
    mode: str = "competition",
    games_dir: Path | None = None,
    bot_a_id: str | None = None,
    bot_b_id: str | None = None,
    bot_a_commit: str | None = None,
    bot_b_commit: str | None = None,
    timeout: float | None = None,
    update_ratings: bool = False,
) -> GameRecord:
    """Run one competition match, store JSON, optionally update ratings."""
    a_path = bot_a_run.resolve()
    b_path = bot_b_run.resolve()
    bot_a = bot_a_id or bot_id_from_run_sh(a_path)
    bot_b = bot_b_id or bot_id_from_run_sh(b_path)
    commit_a = bot_a_commit if bot_a_commit is not None else git_commit_or_tag()
    commit_b = bot_b_commit if bot_b_commit is not None else commit_a

    started_at = utc_now_iso()
    result = run_competition_match(
        a_path, b_path, seed=seed, mode=mode, timeout=timeout
    )
    finished_at = utc_now_iso()

    record = record_from_match_result(
        result,
        bot_a=bot_a,
        bot_b=bot_b,
        seed=seed,
        mode=mode,
        bot_a_commit=commit_a,
        bot_b_commit=commit_b,
        started_at=started_at,
        finished_at=finished_at,
    )
    path = save_game(record, games_dir or GAMES_DIR)
    print(f"[run_match] stored {path}")
    print(
        f"[run_match] winner={record.winner} turns={record.turns} "
        f"terminated={record.terminated} truncated={record.truncated}"
    )

    if update_ratings:
        from arena.ratings import rate_stored_game

        rate_stored_game(record)
        print("[run_match] ratings updated")

    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a competition match and store data/games/<game_id>.json."
    )
    parser.add_argument("bot_a", type=Path, help="path to bot A run.sh")
    parser.add_argument("bot_b", type=Path, help="path to bot B run.sh")
    parser.add_argument(
        "--mode",
        default="competition",
        help="must be competition (default: competition)",
    )
    parser.add_argument("--seed", type=int, default=0, help="match seed (default: 0)")
    parser.add_argument(
        "--games-dir",
        type=Path,
        default=GAMES_DIR,
        help=f"output directory (default: {GAMES_DIR})",
    )
    parser.add_argument("--bot-a-id", default=None, help="override bot A id label")
    parser.add_argument("--bot-b-id", default=None, help="override bot B id label")
    parser.add_argument(
        "--bot-a-commit",
        default=None,
        help="override bot A commit/tag pin",
    )
    parser.add_argument(
        "--bot-b-commit",
        default=None,
        help="override bot B commit/tag pin",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="optional wall-clock match timeout in seconds",
    )
    parser.add_argument(
        "--update-ratings",
        action="store_true",
        help="after storing the game, apply Elo and rewrite leaderboard",
    )
    args = parser.parse_args(argv)

    run_and_store(
        args.bot_a,
        args.bot_b,
        seed=args.seed,
        mode=args.mode,
        games_dir=args.games_dir,
        bot_a_id=args.bot_a_id,
        bot_b_id=args.bot_b_id,
        bot_a_commit=args.bot_a_commit,
        bot_b_commit=args.bot_b_commit,
        timeout=args.timeout,
        update_ratings=args.update_ratings,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
