"""
In-process competition match runner for arena batch workers.

Configures the shared loop in `arena.matches.loop` with a competition-mode
GeneralsEnv, without spawning matchup.py. Results may enter data/games/ and the rating fit.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from arena.paths import REPO_ROOT

_COMP = REPO_ROOT / "competition-module"
if str(_COMP) not in sys.path:
    sys.path.insert(0, str(_COMP))

from arena.matches.loop import run_stdio_match  # noqa: E402
from arena.records.store import Winner  # noqa: E402
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
    stderr: str


def run_competition_match(
    bot_a_run: Path,
    bot_b_run: Path,
    *,
    seed: int = 0,
    mode: str = "competition",
    timeout: float | None = None,
) -> CompetitionMatchResult:
    """
    Run one competition-mode stdio match in-process.

    Returns a structured result (winner seat, turns, castles, bot stderr).
    """
    if mode != "competition":
        raise ValueError(
            f"arena matches require mode='competition' (got {mode!r})"
        )

    result = run_stdio_match(
        GeneralsEnv(mode=mode),
        bot_a_run,
        bot_b_run,
        seed=seed,
        log_tag=LOG_TAG,
        timeout=timeout,
    )
    return CompetitionMatchResult(
        winner=result.winner,
        turns=result.turns,
        terminated=result.terminated,
        truncated=result.truncated,
        castles_built_a=result.castles_built_a,
        castles_built_b=result.castles_built_b,
        stderr=result.stderr,
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
