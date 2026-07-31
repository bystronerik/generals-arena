"""
Shared stdio match loop for the competition and classic harnesses.

Both harnesses drive the same competition-module machinery: build a board from
a ``GeneralsEnv``, spawn two stdio bots over the matchup protocol, and step the
transition until a capture or truncation. They differ only in how the env is
constructed, so each one configures this loop instead of reimplementing it.

``arena.matches.competition`` and ``arena.matches.classic`` are the entry points;
neither result shape changes, and only competition results may enter
``data/games/`` and the rating fit.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO

import jax.numpy as jnp
import numpy as np

from arena.paths import REPO_ROOT
from arena.records.fingerprint import PROBE_FILENAME
from arena.records.trajectories import (
    SEATS,
    TrajectoryRecorder,
    gzip_into_place,
)

_COMP = REPO_ROOT / "competition-module"
for _path in (_COMP, _COMP / "competition"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from arena.records.store import Winner  # noqa: E402
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
    # Terminal-state truth, straight off the engine's own `GameInfo`. The bots
    # are not asked and cannot lie: a seat's view of its opponent is
    # fog-limited, and a seat that crashed reports nothing at all. `None` only
    # when no turn was ever stepped.
    final_land_a: int | None
    final_land_b: int | None
    final_army_a: int | None
    final_army_b: int | None
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


def has_probe(run_sh: Path) -> bool:
    """Whether this bot can be traced at all. `cm_*` wrappers never can."""
    return (run_sh.parent / PROBE_FILENAME).is_file()


def agent_command(run_sh: Path, trace: Path | None) -> tuple[list[str], Path]:
    """
    The command and working directory for one seat.

    Without a trace this is `bash run.sh`, byte for byte what every unrecorded
    match has always run. With one it is the arena's instrumented runner, which
    drives the identical protocol from the identical parsing code and
    additionally samples the bot's probe. The runner needs `arena` importable,
    so it runs from the repo root.
    """
    if trace is None:
        return ["bash", str(run_sh)], run_sh.parent
    command = [
        sys.executable,
        "-m",
        "arena.instrument.runner",
        str(run_sh.parent),
        "--trace",
        str(trace),
    ]
    return command, REPO_ROOT


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
    trace: Path | None = None,
) -> subprocess.Popen:
    """Spawn a stdio bot; stderr goes to a temp file (avoids PIPE deadlock)."""
    command, cwd = agent_command(run_sh, trace)
    proc = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=stderr_file,
        bufsize=1,
        text=True,
        cwd=str(cwd),
        env=env,
    )
    assert proc.stdin is not None
    proc.stdin.write(encode_handshake(player_id, H, W))
    proc.stdin.flush()
    how = "instrumented" if trace is not None else str(run_sh)
    print(
        f"[{log_tag}] spawned {label} as player {player_id} "
        f"(pid={proc.pid}, {how})",
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


def _ints(array) -> list[int]:
    """
    A small JAX array as Python ints, in one device-to-host transfer.

    `[int(x) for x in array]` would pay a transfer per element; on the recorded
    path that is 14 per turn instead of 4, which is the difference between
    fitting the wall-clock budget and not.
    """
    return np.asarray(array).tolist()


def _final_scalars(
    info,
) -> tuple[tuple[int | None, int | None], tuple[int | None, int | None]]:
    """
    `((land_a, land_b), (army_a, army_b))` off the last `GameInfo`.

    Read once at the end rather than per turn: each access pulls a JAX device
    array back to the host. On a capture the engine has already transferred the
    loser's cells to the winner, so these are the post-capture totals — the
    engine's own final answer, not the last thing either bot saw.
    """
    if info is None:
        return (None, None), (None, None)
    return (int(info.land[0]), int(info.land[1])), (int(info.army[0]), int(info.army[1]))


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
    recorder: TrajectoryRecorder | None = None,
) -> MatchLoopResult:
    """
    Play one stdio match on `env` and return its outcome.

    Castles are tallied only when `env.build_castles` is set, so classic runs
    report None for both counts. Final land and army come off the engine's last
    `GameInfo`, so a record never depends on a bot reporting its own score.
    Bot stderr is captured for the caller — it is debug output now, nothing
    parses it — rather than streamed to the terminal.

    With `recorder=None` (the default) the loop body is exactly what it has
    always been and every seat is spawned via `run.sh`. With a recorder, each
    turn's applied actions and engine scalars are logged, and seats whose bot
    carries a `probe.py` are spawned through `arena.instrument.runner` instead
    so their internals are traced too. Only the competition path ever passes
    one — see docs/arena/trajectories.md.
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
    # Scratch trace files, one per traceable seat. The runner writes plain
    # jsonl here; the harness gzips them into the trajectory directory once the
    # process is closed.
    trace_scratch: list[Path | None] = [None, None]
    if recorder is not None:
        recorder.set_dims(H, W)
        scratch_root = Path(tempfile.mkdtemp(prefix="arena-trace-"))
        for index, path in enumerate((a0_path, a1_path)):
            if has_probe(path):
                trace_scratch[index] = scratch_root / f"{SEATS[index]}.jsonl"

    agents = [
        spawn_agent(
            a0_path, 0, H, W, labels[0], venv_env, stderr_files[0],
            log_tag=log_tag, trace=trace_scratch[0],
        ),
        spawn_agent(
            a1_path, 1, H, W, labels[1], venv_env, stderr_files[1],
            log_tag=log_tag, trace=trace_scratch[1],
        ),
    ]

    transition = make_transition(env)
    built = [0, 0]
    prev_castles = state.castles
    winner_player = -1
    turn = 0
    truncated = False
    started = time.monotonic()
    stderr_parts: list[str] = []
    last_info = None

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
            last_info = info
            turn += 1

            if recorder is not None:
                recorder.record_turn(
                    turn,
                    _ints(a_0),
                    _ints(a_1),
                    _ints(info.land),
                    _ints(info.army),
                    state=state,
                )

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

    land, army = _final_scalars(last_info)
    winner = winner_seat(winner_player, truncated=truncated)

    if recorder is not None:
        # After close_agent: the runner writes its trace when its stdin reaches
        # EOF, so the file does not exist until the process has exited.
        recorder.finish(
            winner=winner,
            turns=turn,
            terminated=winner_player >= 0,
            truncated=truncated,
            state=state,
        )
        recorder.write()
        for index, scratch in enumerate(trace_scratch):
            if scratch is not None:
                gzip_into_place(scratch, recorder.trace_destination(SEATS[index]))
        shutil.rmtree(scratch_root, ignore_errors=True)

    return MatchLoopResult(
        winner=winner,
        winner_player_id=winner_player,
        turns=turn,
        terminated=winner_player >= 0,
        truncated=truncated,
        castles_built_a=built[0] if env.build_castles else None,
        castles_built_b=built[1] if env.build_castles else None,
        final_land_a=land[0],
        final_land_b=land[1],
        final_army_a=army[0],
        final_army_b=army[1],
        stderr="\n".join(stderr_parts),
    )
