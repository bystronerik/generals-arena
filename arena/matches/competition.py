"""
In-process competition match runner for arena batch workers.

Configures the shared loop in `arena.matches.loop` with a competition-mode
GeneralsEnv, without spawning matchup.py. Results may enter data/games/ and the rating fit.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from arena.paths import REPO_ROOT

_COMP = REPO_ROOT / "competition-module"
if str(_COMP) not in sys.path:
    sys.path.insert(0, str(_COMP))

from arena.matches.loop import run_stdio_match  # noqa: E402
from arena.records.store import Winner, bot_id_from_run_sh  # noqa: E402
from arena.records.telemetry import series_metrics  # noqa: E402
from arena.records.trajectories import TrajectoryRecorder  # noqa: E402
from generals import GeneralsEnv  # noqa: E402

LOG_TAG = "competition_match"


@dataclass
class CompetitionMatchResult:
    """Structured result of one in-process competition match."""

    winner: Winner
    turns: int
    terminated: bool
    truncated: bool
    castles_built_a: int | None
    castles_built_b: int | None
    # Terminal-state truth from the engine (loop.MatchLoopResult).
    final_land_a: int | None
    final_land_b: int | None
    final_army_a: int | None
    final_army_b: int | None
    stderr: str
    # Reducer output over the recorded per-turn series. Empty on an unrecorded
    # match — missing means not measured, never zero.
    series_metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RecordRequest:
    """
    Everything a trajectory needs that is decided before the match starts.

    Constructing this is what turns recording on. It carries `game_id` because
    a trajectory is keyed by the game it belongs to, which means the id has to
    exist before the first turn rather than being minted with the record
    afterwards.
    """

    game_id: str
    round_name: str
    engine_version: str
    directory: Path


def run_competition_match(
    bot_a_run: Path,
    bot_b_run: Path,
    *,
    seed: int = 0,
    mode: str = "competition",
    timeout: float | None = None,
    record: RecordRequest | None = None,
) -> CompetitionMatchResult:
    """
    Run one competition-mode stdio match in-process.

    Returns a structured result (winner seat, turns, castles, engine finals,
    bot stderr). With `record` set, the match also writes a per-turn trajectory
    (and probe traces for seats whose bot has one) under `record.directory`.
    This is the **only** function in the repo that builds a recorder: classic
    and remote paths cannot record, by construction.
    """
    if mode != "competition":
        raise ValueError(
            f"arena matches require mode='competition' (got {mode!r})"
        )

    recorder = None
    if record is not None:
        recorder = TrajectoryRecorder(
            game_id=record.game_id,
            seed=seed,
            mode=mode,
            round_name=record.round_name,
            engine_version=record.engine_version,
            bot_a=bot_id_from_run_sh(bot_a_run),
            bot_b=bot_id_from_run_sh(bot_b_run),
            directory=record.directory,
        )

    result = run_stdio_match(
        GeneralsEnv(mode=mode),
        bot_a_run,
        bot_b_run,
        seed=seed,
        log_tag=LOG_TAG,
        timeout=timeout,
        recorder=recorder,
    )
    return CompetitionMatchResult(
        winner=result.winner,
        turns=result.turns,
        terminated=result.terminated,
        truncated=result.truncated,
        castles_built_a=result.castles_built_a,
        castles_built_b=result.castles_built_b,
        final_land_a=result.final_land_a,
        final_land_b=result.final_land_b,
        final_army_a=result.final_army_a,
        final_army_b=result.final_army_b,
        stderr=result.stderr,
        series_metrics=(
            series_metrics(record.directory, record.game_id) if record is not None else {}
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run one in-process competition match (stdio bots)."
    )
    parser.add_argument("bot_a", type=Path, help="path to bot A run.sh")
    parser.add_argument("bot_b", type=Path, help="path to bot B run.sh")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (default: 0)")
    parser.add_argument(
        "--mode",
        default="competition",
        help="must be competition (default: competition)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="optional wall-clock match timeout in seconds",
    )
    args = parser.parse_args(argv)

    result = run_competition_match(
        args.bot_a,
        args.bot_b,
        seed=args.seed,
        mode=args.mode,
        timeout=args.timeout,
    )
    labels = [
        args.bot_a.resolve().parent.name,
        args.bot_b.resolve().parent.name,
    ]
    if result.terminated:
        player = 0 if result.winner == "a" else 1
        print(
            f"[{LOG_TAG}] turn {result.turns}: player {player} "
            f"({labels[player]}) captured the enemy general"
        )
    else:
        print(f"[{LOG_TAG}] turn {result.turns}: truncated (draw)")
    if result.castles_built_a is not None and result.castles_built_b is not None:
        print(
            f"[{LOG_TAG}] castles built: {result.castles_built_a} "
            f"({labels[0]}) vs {result.castles_built_b} ({labels[1]})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
