"""Run one in-process competition match and store the game record."""

from __future__ import annotations

import argparse
from pathlib import Path

from arena.matches.competition import run_competition_match
from arena.records.fingerprint import bot_content_hash
from arena.records.store import (
    GAMES_DIR,
    GameRecord,
    bot_id_from_run_sh,
    engine_version,
    save_game,
    utc_now_iso,
)
from arena.records.telemetry import record_from_match_result

# Round name for one-off matches that belong to no measurement round. Stored
# explicitly so the eligibility filter never has to guess from a path.
ADHOC_ROUND = "adhoc"


def run_and_store(
    bot_a_run: Path,
    bot_b_run: Path,
    *,
    seed: int = 0,
    mode: str = "competition",
    round_name: str = ADHOC_ROUND,
    games_dir: Path | None = None,
    bot_a_id: str | None = None,
    bot_b_id: str | None = None,
    bot_a_content_hash: str | None = None,
    bot_b_content_hash: str | None = None,
    timeout: float | None = None,
    update_ratings: bool = False,
) -> GameRecord:
    """Run one competition match, store JSON, optionally refit ratings."""
    a_path = bot_a_run.resolve()
    b_path = bot_b_run.resolve()
    bot_a = bot_a_id or bot_id_from_run_sh(a_path)
    bot_b = bot_b_id or bot_id_from_run_sh(b_path)
    hash_a = bot_a_content_hash if bot_a_content_hash is not None else bot_content_hash(a_path)
    hash_b = bot_b_content_hash if bot_b_content_hash is not None else bot_content_hash(b_path)

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
        round_name=round_name,
        bot_a_content_hash=hash_a,
        bot_b_content_hash=hash_b,
        engine_version=engine_version(),
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
        from arena.records.ratings import rate_stored_game

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
        "--round",
        default=ADHOC_ROUND,
        help=f"round name stored on the record (default: {ADHOC_ROUND})",
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
        help="after storing the game, refit ratings and rewrite the leaderboard",
    )
    args = parser.parse_args(argv)

    run_and_store(
        args.bot_a,
        args.bot_b,
        seed=args.seed,
        mode=args.mode,
        round_name=args.round,
        games_dir=args.games_dir,
        bot_a_id=args.bot_a_id,
        bot_b_id=args.bot_b_id,
        timeout=args.timeout,
        update_ratings=args.update_ratings,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
