"""Run one in-process competition match and store the game record."""

from __future__ import annotations

import argparse
from pathlib import Path

from arena.matches.competition import run_competition_match
from arena.records.fingerprint import bot_content_hash
from arena.records.registry import Registry, is_registerable
from arena.records.store import (
    GAMES_DIR,
    GameRecord,
    bot_id_from_run_sh,
    engine_version,
    save_game,
)
from arena.records.telemetry import record_from_match_result

# Round name for one-off matches that belong to no measurement round. Stored
# explicitly so the eligibility filter never has to guess from a path.
ADHOC_ROUND = "adhoc"


def _register(run_sh: Path) -> str:
    """
    Hash the bot and record the version, in this (parent) process.

    A bot outside `bots/` — only `competition-module`'s own `expander_python`
    — still gets a hash and still plays, but no registry entry, so its games
    are counted under `excluded.unregistered_hash` rather than silently pooled.
    """
    bot_dir = run_sh.parent
    if not is_registerable(bot_dir):
        digest = bot_content_hash(run_sh)
        print(
            f"[run_match] {bot_dir.name} is outside bots/; not registered, so this "
            f"game will be excluded from ratings"
        )
        return digest
    version, is_new = Registry().register(bot_dir)
    if is_new:
        print(f"[run_match] registered {bot_dir.name}@{version.content_hash}")
    return version.content_hash


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
    hash_a = bot_a_content_hash or _register(a_path)
    hash_b = bot_b_content_hash or _register(b_path)

    result = run_competition_match(
        a_path, b_path, seed=seed, mode=mode, timeout=timeout
    )

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
    )
    path = save_game(record, games_dir or GAMES_DIR)
    print(f"[run_match] stored {path}")
    print(
        f"[run_match] winner={record.winner} turns={record.turns} "
        f"terminated={record.terminated} truncated={record.truncated}"
    )

    if update_ratings:
        # Refit, not "apply one update". There is no incremental path: a
        # path-dependent estimator has no single right answer, and the two
        # paths it used to have silently disagreed.
        from arena.records.ratings.cli import refit

        fit = refit()
        print(f"[run_match] refitted ratings over {fit.counts.games} game(s)")

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
