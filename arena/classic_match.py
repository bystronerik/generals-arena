"""
Classic-approximate local harness for remote-only bots.

Configures the shared loop in `arena.match_loop` with a GeneralsEnv built for
classic rules: build_castles=False, deathtouch_turn=None, and neutral
pre-placed cities. This is NOT competition matchup — results do not enter
data/games/ or Elo.

See docs/engine/classic-matchup.md and docs/research/strategies/human-95-plan.md §4.2.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_COMP = _REPO_ROOT / "competition-module"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_COMP) not in sys.path:
    sys.path.insert(0, str(_COMP))

from arena.match_loop import run_stdio_match  # noqa: E402
from generals import GeneralsEnv  # noqa: E402

LOG_TAG = "classic_match"

# Classic-like defaults (see human-95-plan §4.2).
CLASSIC_ENV_DEFAULTS = dict(
    grid_dims=(24, 24),
    truncation=5000,
    build_castles=False,
    deathtouch_turn=None,
    num_castles_range=(3, 6),
    castle_val_range=(20, 40),
    mountain_density_range=(0.20, 0.26),
    perfect_info=False,
    min_generals_distance=10,
)


def make_classic_env(**overrides) -> GeneralsEnv:
    """Build a GeneralsEnv approximating classic generals.io 1v1."""
    params = {**CLASSIC_ENV_DEFAULTS, **overrides}
    return GeneralsEnv(**params)


def run_classic_match(
    bot_a_run: Path,
    bot_b_run: Path,
    *,
    seed: int = 0,
    env_overrides: dict | None = None,
    timeout: float | None = None,
) -> tuple[int, int, bool]:
    """
    Run one classic-approximate stdio match.

    Returns (winner_player_id, turns, truncated).
    winner_player_id is -1 when truncated without a capture.
    """
    result = run_stdio_match(
        make_classic_env(**(env_overrides or {})),
        bot_a_run,
        bot_b_run,
        seed=seed,
        log_tag=LOG_TAG,
        timeout=timeout,
    )
    return result.winner_player_id, result.turns, result.truncated


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a classic-approximate local match (neutral cities, no build, "
            "no deathtouch). Not a competition result."
        )
    )
    parser.add_argument("bot_a", type=Path, help="path to bot A run.sh")
    parser.add_argument("bot_b", type=Path, help="path to bot B run.sh")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (default: 0)")
    parser.add_argument(
        "--grid-size",
        type=int,
        default=CLASSIC_ENV_DEFAULTS["grid_dims"][0],
        help="square board side (default: 24)",
    )
    parser.add_argument(
        "--truncation",
        type=int,
        default=CLASSIC_ENV_DEFAULTS["truncation"],
        help="max turns before stop (default: 5000)",
    )
    args = parser.parse_args(argv)

    winner, turns, truncated = run_classic_match(
        args.bot_a,
        args.bot_b,
        seed=args.seed,
        env_overrides={
            "grid_dims": (args.grid_size, args.grid_size),
            "truncation": args.truncation,
        },
    )

    labels = [args.bot_a.resolve().parent.name, args.bot_b.resolve().parent.name]
    if winner >= 0:
        print(f"[{LOG_TAG}] turn {turns}: player {winner} ({labels[winner]}) won")
    elif truncated:
        print(f"[{LOG_TAG}] turn {turns}: truncated at {args.truncation} turns (draw)")
    else:
        print(f"[{LOG_TAG}] turn {turns}: no winner recorded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
