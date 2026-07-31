"""
Shared stdio match loop for the competition and classic harnesses.

Both harnesses drive the same competition-module machinery: build a board from
a ``GeneralsEnv``, spawn two stdio bots over the matchup protocol, and step the
transition until a capture or truncation. They differ only in how the env is
constructed, so each one configures this loop instead of reimplementing it.

``arena.competition_match`` and ``arena.classic_match`` are the entry points;
neither result shape changes, and only competition results may enter
``data/games/`` and Elo.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO

import jax.numpy as jnp

from arena.paths import REPO_ROOT

_COMP = REPO_ROOT / "competition-module"
for _path in (_COMP, _COMP / "competition"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from arena.store import Winner  # noqa: E402
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
class MatchLoopResult:
    """One finished stdio match, before harness-specific record mapping."""

    winner: Winner
    winner_player_id: int
    turns: int
    terminated: bool
    truncated: bool
    castles_built_a: int | None
    castles_built_b: int | None
    stderr: str


def venv_path_env() -> dict[str, str]:
    """Child environment that resolves this venv's interpreter for run.sh."""
    env = os.environ.copy()
    # Keep .venv/bin on PATH; resolve() follows symlinks to the system framework.
    venv_bin = str(Path(sys.executable).parent)
    env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")
    env["PYTHON"] = sys.executable
    env["PYTHONUNBUFFERED"] = "1"
    return env


def spawn_agent(
    run_sh: Path,
    player_id: int,
    H: int,
    W: int,
    label: str,
    env: dict[str, str],
    stderr_file: IO[str],
    *,
    log_tag: str,
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
        f"[{log_tag}] spawned {label} as player {player_id} "
        f"(pid={proc.pid}, {run_sh})",
        file=sys.stderr,
    )
    return proc


def winner_seat(winner_player_id: int, *, truncated: bool) -> Winner:
    """Map a matchup winner player id to the stored seat label."""
    if winner_player_id == 0:
        return "a"
    if winner_player_id == 1:
        return "b"
    if truncated or winner_player_id < 0:
        return "draw"
    raise ValueError(f"invalid winner_player_id: {winner_player_id}")


def _read_temp_stderr(stderr_file: IO[str]) -> str:
    try:
        stderr_file.seek(0)
        return stderr_file.read() or ""
    except (OSError, ValueError):
        return ""


def run_stdio_match(
    env,
    bot_a_run: Path,
    bot_b_run: Path,
    *,
    seed: int,
    log_tag: str,
    timeout: float | None = None,
) -> MatchLoopResult:
    """
    Play one stdio match on `env` and return its outcome.

    Castles are tallied only when `env.build_castles` is set, so classic runs
    report None for both counts. Bot stderr is captured for the caller (the
    competition path parses `[telemetry]` lines out of it) rather than
    streamed to the terminal.
    """
    a0_path = bot_a_run.resolve()
    a1_path = bot_b_run.resolve()
    for p in (a0_path, a1_path):
        if not p.exists():
            raise FileNotFoundError(f"agent script not found: {p}")

    build_agent(a0_path)
    if a1_path != a0_path:
        build_agent(a1_path)

    venv_env = venv_path_env()
    get_obs = game.get_full_observation if env.perfect_info else game.get_observation

    state = make_board(env, seed)
    H, W = (int(d) for d in state.armies.shape)

    labels = [a0_path.parent.name, a1_path.parent.name]
    stderr_files = [
        tempfile.TemporaryFile(mode="w+", encoding="utf-8"),
        tempfile.TemporaryFile(mode="w+", encoding="utf-8"),
    ]
    agents = [
        spawn_agent(a0_path, 0, H, W, labels[0], venv_env, stderr_files[0], log_tag=log_tag),
        spawn_agent(a1_path, 1, H, W, labels[1], venv_env, stderr_files[1], log_tag=log_tag),
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
                    f"{log_tag} match exceeded timeout={timeout}s "
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

    return MatchLoopResult(
        winner=winner_seat(winner_player, truncated=truncated),
        winner_player_id=winner_player,
        turns=turn,
        terminated=winner_player >= 0,
        truncated=truncated,
        castles_built_a=built[0] if env.build_castles else None,
        castles_built_b=built[1] if env.build_castles else None,
        stderr="\n".join(stderr_parts),
    )
