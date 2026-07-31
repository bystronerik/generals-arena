"""
Classic-approximate local harness for remote-only bots.

Uses GeneralsEnv with build_castles=False, deathtouch_turn=None, and neutral
pre-placed cities. This is NOT competition matchup — results do not enter
data/games/ or Elo.

See docs/engine/classic-matchup.md and docs/research/strategies/human-95-plan.md §4.2.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Literal

Winner = Literal["a", "b", "draw"]

import jax.numpy as jnp

_REPO_ROOT = Path(__file__).resolve().parent.parent
_COMP = _REPO_ROOT / "competition-module"
if str(_COMP) not in sys.path:
    sys.path.insert(0, str(_COMP))
if str(_COMP / "competition") not in sys.path:
    sys.path.insert(0, str(_COMP / "competition"))

from generals import GeneralsEnv  # noqa: E402
from generals.core import game  # noqa: E402
from matchup import (  # noqa: E402
    ask_agent,
    build_agent,
    close_agent,
    make_board,
    make_transition,
)
from protocol import encode_handshake  # noqa: E402

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


def _venv_path_env() -> dict[str, str]:
    env = os.environ.copy()
    venv_bin = str(Path(sys.executable).parent)
    env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _spawn_agent(
    run_sh: Path,
    player_id: int,
    H: int,
    W: int,
    label: str,
    env: dict[str, str],
) -> subprocess.Popen:
    proc = subprocess.Popen(
        ["bash", str(run_sh)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
        bufsize=1,
        text=True,
        cwd=str(run_sh.parent),
        env=env,
    )
    proc.stdin.write(encode_handshake(player_id, H, W))
    proc.stdin.flush()
    print(
        f"[classic_match] spawned {label} as player {player_id} "
        f"(pid={proc.pid}, {run_sh})",
        file=sys.stderr,
    )
    return proc


def classic_winner_seat(winner_player_id: int, *, truncated: bool) -> Winner:
    """Map run_classic_match player id to seat winner label."""
    if winner_player_id == 0:
        return "a"
    if winner_player_id == 1:
        return "b"
    if truncated or winner_player_id < 0:
        return "draw"
    raise ValueError(f"invalid winner_player_id: {winner_player_id}")


def run_classic_match(
    bot_a_run: Path,
    bot_b_run: Path,
    *,
    seed: int = 0,
    env_overrides: dict | None = None,
) -> tuple[int, int, bool]:
    """
    Run one classic-approximate stdio match.

    Returns (winner_player_id, turns, truncated).
    winner_player_id is -1 when truncated without a capture.
    """
    a0_path = bot_a_run.resolve()
    a1_path = bot_b_run.resolve()
    for p in (a0_path, a1_path):
        if not p.exists():
            raise FileNotFoundError(f"agent script not found: {p}")

    build_agent(a0_path)
    if a1_path != a0_path:
        build_agent(a1_path)

    venv_env = _venv_path_env()

    env = make_classic_env(**(env_overrides or {}))
    get_obs = game.get_full_observation if env.perfect_info else game.get_observation

    state = make_board(env, seed)
    H, W = (int(d) for d in state.armies.shape)

    labels = [a0_path.parent.name, a1_path.parent.name]
    agents = [
        _spawn_agent(a0_path, 0, H, W, labels[0], venv_env),
        _spawn_agent(a1_path, 1, H, W, labels[1], venv_env),
    ]

    transition = make_transition(env)
    winner = -1
    turn = 0
    truncated = False
    try:
        while turn < env.truncation:
            obs_0 = get_obs(state, 0)
            obs_1 = get_obs(state, 1)

            a_0 = ask_agent(agents[0], obs_0)
            a_1 = ask_agent(agents[1], obs_1)

            actions = jnp.stack([a_0, a_1])
            state, info = transition(state, actions)
            turn += 1

            if bool(info.is_done):
                winner = int(info.winner)
                break
        else:
            truncated = True
    finally:
        for proc in agents:
            close_agent(proc)

    return winner, turn, truncated


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
        print(f"[classic_match] turn {turns}: player {winner} ({labels[winner]}) won")
    elif truncated:
        print(f"[classic_match] turn {turns}: truncated at {args.truncation} turns (draw)")
    else:
        print(f"[classic_match] turn {turns}: no winner recorded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
