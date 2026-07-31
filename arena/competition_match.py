"""
In-process competition match runner for arena batch workers.

Uses competition-module matchup helpers (board, transition, stdio protocol)
without spawning matchup.py. Results may enter data/games/ and Elo.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO

import jax.numpy as jnp

_REPO_ROOT = Path(__file__).resolve().parent.parent
_COMP = _REPO_ROOT / "competition-module"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_COMP) not in sys.path:
    sys.path.insert(0, str(_COMP))
if str(_COMP / "competition") not in sys.path:
    sys.path.insert(0, str(_COMP / "competition"))

from arena.store import Winner  # noqa: E402
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


def _venv_path_env() -> dict[str, str]:
    env = os.environ.copy()
    venv_bin = str(Path(sys.executable).parent)
    env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")
    env["PYTHON"] = sys.executable
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _spawn_agent(
    run_sh: Path,
    player_id: int,
    H: int,
    W: int,
    label: str,
    env: dict[str, str],
    stderr_file: IO[str],
) -> subprocess.Popen:
    """Spawn a stdio bot; stderr goes to a temp file (avoids PIPE deadlock)."""
    proc = subprocess.Popen(
        ["bash", str(run_sh)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=stderr_file,
        bufsize=1,
        text=True,
        cwd=str(run_sh.parent),
        env=env,
    )
    assert proc.stdin is not None
    proc.stdin.write(encode_handshake(player_id, H, W))
    proc.stdin.flush()
    print(
        f"[competition_match] spawned {label} as player {player_id} "
        f"(pid={proc.pid}, {run_sh})",
        file=sys.stderr,
    )
    return proc


def _read_temp_stderr(stderr_file: IO[str]) -> str:
    try:
        stderr_file.seek(0)
        return stderr_file.read() or ""
    except (OSError, ValueError):
        return ""


def competition_winner_seat(winner_player_id: int, *, truncated: bool) -> Winner:
    if winner_player_id == 0:
        return "a"
    if winner_player_id == 1:
        return "b"
    if truncated or winner_player_id < 0:
        return "draw"
    raise ValueError(f"invalid winner_player_id: {winner_player_id}")


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

    a0_path = bot_a_run.resolve()
    a1_path = bot_b_run.resolve()
    for p in (a0_path, a1_path):
        if not p.exists():
            raise FileNotFoundError(f"agent script not found: {p}")

    build_agent(a0_path)
    if a1_path != a0_path:
        build_agent(a1_path)

    venv_env = _venv_path_env()
    env = GeneralsEnv(mode=mode)
    get_obs = game.get_full_observation if env.perfect_info else game.get_observation

    state = make_board(env, seed)
    H, W = (int(d) for d in state.armies.shape)

    labels = [a0_path.parent.name, a1_path.parent.name]
    stderr_files = [
        tempfile.TemporaryFile(mode="w+", encoding="utf-8"),
        tempfile.TemporaryFile(mode="w+", encoding="utf-8"),
    ]
    agents = [
        _spawn_agent(a0_path, 0, H, W, labels[0], venv_env, stderr_files[0]),
        _spawn_agent(a1_path, 1, H, W, labels[1], venv_env, stderr_files[1]),
    ]

    transition = make_transition(env)
    built = [0, 0]
    prev_castles = state.castles
    winner_player = -1
    turn = 0
    truncated = False
    started = time.monotonic()
    stderr_parts: list[str] = []

    try:
        while turn < env.truncation:
            if timeout is not None and (time.monotonic() - started) > timeout:
                raise TimeoutError(
                    f"competition match exceeded timeout={timeout}s "
                    f"at turn={turn} seed={seed}"
                )

            obs_0 = get_obs(state, 0)
            obs_1 = get_obs(state, 1)

            a_0 = ask_agent(agents[0], obs_0)
            a_1 = ask_agent(agents[1], obs_1)

            actions = jnp.stack([a_0, a_1])
            state, info = transition(state, actions)
            turn += 1

            if env.build_castles:
                born = state.castles & ~prev_castles
                if bool(born.any()):
                    for pid in (0, 1):
                        for _r, _c in zip(*jnp.where(born & state.ownership[pid])):
                            built[pid] += 1
                    prev_castles = state.castles

            if bool(info.is_done):
                winner_player = int(info.winner)
                break
        else:
            truncated = True
    finally:
        for proc in agents:
            close_agent(proc)
        for err in stderr_files:
            stderr_parts.append(_read_temp_stderr(err))
            err.close()

    seat = competition_winner_seat(winner_player, truncated=truncated)
    castles_a: int | None = built[0] if env.build_castles else None
    castles_b: int | None = built[1] if env.build_castles else None
    return CompetitionMatchResult(
        winner=seat,
        turns=turn,
        terminated=winner_player >= 0,
        truncated=truncated,
        castles_built_a=castles_a,
        castles_built_b=castles_b,
        stderr="\n".join(stderr_parts),
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
            f"[competition_match] turn {result.turns}: player {player} "
            f"({labels[player]}) captured the enemy general"
        )
    else:
        print(
            f"[competition_match] turn {result.turns}: truncated "
            f"(draw)"
        )
    if result.castles_built_a is not None and result.castles_built_b is not None:
        print(
            f"[competition_match] castles built: {result.castles_built_a} "
            f"({labels[0]}) vs {result.castles_built_b} ({labels[1]})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
