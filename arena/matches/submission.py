"""
Submission-shaped harness: judge process contract for an extracted bot.

Enforces the competition match constraints from RULES.md §08 against one
judged seat. Results never enter ``data/games/`` or the rating fit — use
``arena.matches.run_match`` for stored arena games.

CLI::

    python -m arena.matches.submission <bundle.zip> \\
      --opponent bots/smoke/run.sh --mode competition --seed 0
"""

from __future__ import annotations

import argparse
import fcntl
import os
import select
import signal
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import IO

import jax.numpy as jnp

from arena.paths import REPO_ROOT

_COMP = REPO_ROOT / "competition-module"
for _path in (_COMP, _COMP / "competition"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from arena.bundle import _path_with_python  # noqa: E402
from arena.matches.loop import (  # noqa: E402
    _read_temp_stderr,
    spawn_agent,
    venv_path_env,
    winner_seat,
)
from arena.records.store import Winner  # noqa: E402
from generals import GeneralsEnv  # noqa: E402
from generals.core import game  # noqa: E402
from matchup import (  # noqa: E402
    ask_agent,
    build_agent,
    close_agent,
    make_board,
    make_transition,
)
from protocol import encode_handshake, encode_observation  # noqa: E402

LOG_TAG = "submission"

# Judge limits (RULES.md §08). EOF grace is unpublished; reuse the local
# runner's 3 s wait until judge docs publish a value.
FIRST_REPLY_SECONDS = 10.0
NORMAL_REPLY_SECONDS = 0.150
FAULT_BUDGET = 50
MEMORY_CAP_BYTES = 2 * 1024**3
EOF_GRACE_SECONDS = 3.0

PASS_ACTION = (1, 0, 0, 0, 0)
_PASS_JNP = jnp.array(PASS_ACTION, dtype=jnp.int32)


class RejectReason(str, Enum):
    """Why a judged run was rejected, or ACCEPTED when every check passed."""

    ACCEPTED = "accepted"
    FIRST_REPLY_TIMEOUT = "first_reply_timeout"
    NORMAL_REPLY_TIMEOUT = "normal_reply_timeout"
    MISSING_REPLY = "missing_reply"
    MALFORMED_REPLY = "malformed_reply"
    FAULT_BUDGET = "fault_budget"
    MEMORY_CAP = "memory_cap"
    CRASH = "crash"
    EOF_HANG = "eof_hang"
    EOF_NONZERO = "eof_nonzero"


@dataclass(frozen=True)
class SubmissionResult:
    """Outcome of one judged competition match. Never a game-store record."""

    accepted: bool
    reason: RejectReason
    faults: int
    peak_rss_bytes: int
    eof_voluntary: bool
    eof_exit_code: int | None
    eof_forced: bool
    winner: Winner
    turns: int
    terminated: bool
    truncated: bool
    stderr: str


class _MemoryGuard:
    """Sample process-group RSS and kill the group when it crosses the cap."""

    def __init__(self, pgid: int, cap_bytes: int, interval: float = 0.02) -> None:
        self.pgid = pgid
        self.cap_bytes = cap_bytes
        self.interval = interval
        self.peak_rss_bytes = 0
        self.tripped = False
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="submission-rss", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                rss = process_group_rss_bytes(self.pgid)
            except OSError:
                continue
            if rss > self.peak_rss_bytes:
                self.peak_rss_bytes = rss
            if rss > self.cap_bytes:
                self.tripped = True
                _kill_process_group(self.pgid)
                return


def process_group_rss_bytes(pgid: int) -> int:
    """
    Sum RSS for every live process in ``pgid``.

    Prefer Linux ``/proc`` (promotion path). Fall back to ``ps``, which reports
    RSS in KiB on both Linux and macOS. Peak RSS and hard address-space limits
    still differ by platform; promotion uses Linux.
    """
    if Path("/proc").is_dir():
        return _process_group_rss_procfs(pgid)
    return _process_group_rss_ps(pgid)


def _process_group_rss_procfs(pgid: int) -> int:
    total = 0
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text().split()
            # Field 5 (1-based) is pgrp; list index 4 after the comm token can
            # contain spaces inside parentheses, so parse from the right of ').'
            close = " ".join(stat).rfind(")")
            fields = " ".join(stat)[close + 2 :].split()
            if int(fields[2]) != pgid:  # pgrp is the 3rd field after comm
                continue
            # smaps_rollup Rss is kB; fall back to statm pages.
            rollup = entry / "smaps_rollup"
            if rollup.is_file():
                for line in rollup.read_text().splitlines():
                    if line.startswith("Rss:"):
                        total += int(line.split()[1]) * 1024
                        break
            else:
                pages = int((entry / "statm").read_text().split()[1])
                total += pages * os.sysconf("SC_PAGE_SIZE")
        except (OSError, ValueError, IndexError):
            continue
    return total


def _process_group_rss_ps(pgid: int) -> int:
    out = subprocess.check_output(
        ["ps", "-eo", "pid=,rss=,pgid="],
        text=True,
    )
    total_kib = 0
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            rss_kib = int(parts[1])
            group = int(parts[2])
        except ValueError:
            continue
        if group == pgid:
            total_kib += rss_kib
    return total_kib * 1024


def _kill_process_group(pgid: int) -> None:
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def parse_action_line(line: str) -> tuple[int, int, int, int, int] | None:
    """Return five ints when the line is exactly five integers; else None."""
    parts = line.split()
    if len(parts) != 5:
        return None
    try:
        return tuple(int(p) for p in parts)  # type: ignore[return-value]
    except ValueError:
        return None


def _set_nonblocking(fd: int) -> None:
    flags = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)


class _JudgedSeat:
    """One judged stdio bot with per-reply clocks and fault accounting."""

    def __init__(
        self,
        proc: subprocess.Popen[bytes],
        *,
        first_reply_s: float,
        normal_reply_s: float,
        fault_budget: int,
    ) -> None:
        self.proc = proc
        self.first_reply_s = first_reply_s
        self.normal_reply_s = normal_reply_s
        self.fault_budget = fault_budget
        self.faults = 0
        self.replies = 0
        self.first_reject: RejectReason | None = None
        self._buf = bytearray()
        assert proc.stdout is not None
        _set_nonblocking(proc.stdout.fileno())

    @property
    def pgid(self) -> int:
        return os.getpgid(self.proc.pid)

    def _alive(self) -> bool:
        return self.proc.poll() is None

    def _record_fault(self, reason: RejectReason) -> None:
        self.faults += 1
        if self.first_reject is None:
            self.first_reject = reason

    def ask(
        self,
        obs,
        *,
        memory_tripped: Callable[[], bool],
    ) -> tuple[jnp.ndarray, RejectReason | None]:
        """
        Write one observation and read one reply under the judge clock.

        Returns ``(action, forfeit_reason)``. ``forfeit_reason`` is set when
        the seat must leave the match immediately (crash, fault budget).
        """
        assert self.proc.stdin is not None
        assert self.proc.stdout is not None

        if memory_tripped():
            return _PASS_JNP, RejectReason.MEMORY_CAP

        if not self._alive():
            self._record_fault(RejectReason.CRASH)
            return _PASS_JNP, RejectReason.CRASH

        frame = encode_observation(obs).encode("ascii")
        try:
            self.proc.stdin.write(frame)
            self.proc.stdin.flush()
        except BrokenPipeError:
            self._record_fault(RejectReason.CRASH)
            return _PASS_JNP, RejectReason.CRASH

        # Reply clock starts after the complete observation frame is flushed.
        limit = self.first_reply_s if self.replies == 0 else self.normal_reply_s
        deadline = time.monotonic() + limit
        timeout_reason = (
            RejectReason.FIRST_REPLY_TIMEOUT
            if self.replies == 0
            else RejectReason.NORMAL_REPLY_TIMEOUT
        )

        line = self._read_line(deadline, memory_tripped)
        self.replies += 1

        if memory_tripped():
            return _PASS_JNP, RejectReason.MEMORY_CAP

        if line is None:
            # Closed stdout / process exit before a complete line.
            if not self._alive():
                self._record_fault(RejectReason.CRASH)
                return _PASS_JNP, RejectReason.CRASH
            self._record_fault(RejectReason.MISSING_REPLY)
            return self._after_fault()

        if line is False:
            self._record_fault(timeout_reason)
            return self._after_fault()

        if line == "":
            self._record_fault(RejectReason.MISSING_REPLY)
            return self._after_fault()

        action = parse_action_line(line)
        if action is None:
            self._record_fault(RejectReason.MALFORMED_REPLY)
            return self._after_fault()

        return jnp.array(action, dtype=jnp.int32), None

    def _after_fault(self) -> tuple[jnp.ndarray, RejectReason | None]:
        if self.faults >= self.fault_budget:
            if self.first_reject is None:
                self.first_reject = RejectReason.FAULT_BUDGET
            return _PASS_JNP, RejectReason.FAULT_BUDGET
        return _PASS_JNP, None

    def _read_line(
        self,
        deadline: float,
        memory_tripped: Callable[[], bool],
    ) -> str | None | bool:
        """
        Read one complete line without blocking ``readline``.

        Returns the decoded line (no trailing newline), ``False`` on timeout,
        or ``None`` when the pipe closes before a newline.
        """
        fd = self.proc.stdout.fileno()
        while True:
            nl = self._buf.find(b"\n")
            if nl >= 0:
                raw = bytes(self._buf[:nl])
                del self._buf[: nl + 1]
                return raw.decode("ascii", errors="replace").rstrip("\r")

            if memory_tripped():
                return False

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False

            if not self._alive() and not self._buf:
                return None

            ready, _, _ = select.select([fd], [], [], remaining)
            if not ready:
                return False
            try:
                chunk = os.read(fd, 4096)
            except BlockingIOError:
                continue
            if not chunk:
                return None
            self._buf.extend(chunk)


def _spawn_judged(
    run_sh: Path,
    player_id: int,
    H: int,
    W: int,
    label: str,
    env: dict[str, str],
    stderr_file: IO[str],
) -> subprocess.Popen[bytes]:
    """Spawn the judged bot in its own process group (binary stdio)."""
    proc = subprocess.Popen(
        ["bash", str(run_sh)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=stderr_file,
        bufsize=0,
        cwd=str(run_sh.parent),
        env=env,
        start_new_session=True,
    )
    assert proc.stdin is not None
    proc.stdin.write(encode_handshake(player_id, H, W).encode("ascii"))
    proc.stdin.flush()
    print(
        f"[{LOG_TAG}] spawned judged {label} as player {player_id} "
        f"(pid={proc.pid}, pgid={os.getpgid(proc.pid)})",
        file=sys.stderr,
    )
    return proc


def _close_judged(
    proc: subprocess.Popen[bytes],
    *,
    grace: float,
) -> tuple[bool, int | None, bool]:
    """
    Close stdin and wait for a voluntary exit.

    Returns ``(voluntary, exit_code, forced)``. ``forced`` is True when the
    harness had to signal or kill the process group.
    """
    try:
        if proc.stdin is not None:
            proc.stdin.close()
    except (BrokenPipeError, ValueError):
        pass

    try:
        code = proc.wait(timeout=grace)
        return code == 0, code, False
    except subprocess.TimeoutExpired:
        pgid = os.getpgid(proc.pid) if proc.poll() is None else None
        if pgid is not None:
            try:
                os.killpg(pgid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                _kill_process_group(pgid)
                proc.wait()
        else:
            proc.kill()
            proc.wait()
        return False, proc.returncode, True


def extract_bundle(zip_path: Path, extract_dir: Path) -> Path:
    """Unpack a submission zip; return the path to its root ``run.sh``."""
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            target = Path(zf.extract(info, extract_dir))
            mode = (info.external_attr >> 16) & 0o777
            if mode:
                target.chmod(mode)
    run_sh = extract_dir / "run.sh"
    if not run_sh.is_file():
        raise FileNotFoundError(f"bundle has no run.sh: {zip_path}")
    return run_sh


def run_submission_match(
    judged_run: Path,
    opponent_run: Path,
    *,
    seed: int = 0,
    mode: str = "competition",
    judged_seat: int = 0,
    memory_cap_bytes: int = MEMORY_CAP_BYTES,
    first_reply_s: float = FIRST_REPLY_SECONDS,
    normal_reply_s: float = NORMAL_REPLY_SECONDS,
    fault_budget: int = FAULT_BUDGET,
    eof_grace_s: float = EOF_GRACE_SECONDS,
    max_turns: int | None = None,
) -> SubmissionResult:
    """
    Play one competition match with judge enforcement on ``judged_run``.

    Does not write ``data/games/`` or touch ratings. ``max_turns`` shortens the
    loop for fixture tests; production callers leave it ``None``.
    """
    if mode != "competition":
        raise ValueError(f"submission harness requires mode='competition' (got {mode!r})")
    if judged_seat not in (0, 1):
        raise ValueError(f"judged_seat must be 0 or 1 (got {judged_seat})")

    judged_path = judged_run.resolve()
    opponent_path = opponent_run.resolve()
    for path in (judged_path, opponent_path):
        if not path.exists():
            raise FileNotFoundError(f"agent script not found: {path}")

    build_agent(judged_path)
    if opponent_path != judged_path:
        build_agent(opponent_path)

    env = GeneralsEnv(mode=mode)
    get_obs = game.get_full_observation if env.perfect_info else game.get_observation
    state = make_board(env, seed)
    H, W = (int(d) for d in state.armies.shape)
    turn_limit = env.truncation if max_turns is None else min(max_turns, env.truncation)

    venv_env = venv_path_env()
    # Bundles expect bare `python` on PATH (see arena.bundle smoke).
    venv_env["PATH"] = _path_with_python(venv_env.get("PATH", os.defpath), judged_path.parent)

    stderr_files = [
        tempfile.TemporaryFile(mode="w+", encoding="utf-8"),
        tempfile.TemporaryFile(mode="w+", encoding="utf-8"),
    ]
    labels = [judged_path.parent.name, opponent_path.parent.name]
    if judged_seat == 1:
        labels = [opponent_path.parent.name, judged_path.parent.name]

    judged_proc: subprocess.Popen[bytes] | None = None
    opponent_proc = None
    guard: _MemoryGuard | None = None
    seat: _JudgedSeat | None = None
    forfeit: RejectReason | None = None
    winner_player = -1
    turn = 0
    truncated = False
    eof_voluntary = False
    eof_exit_code: int | None = None
    eof_forced = False
    peak_rss = 0
    faults = 0
    first_reject: RejectReason | None = None

    try:
        if judged_seat == 0:
            judged_proc = _spawn_judged(
                judged_path, 0, H, W, labels[0], venv_env, stderr_files[0]
            )
            opponent_proc = spawn_agent(
                opponent_path, 1, H, W, labels[1], venv_env, stderr_files[1],
                log_tag=LOG_TAG,
            )
        else:
            opponent_proc = spawn_agent(
                opponent_path, 0, H, W, labels[0], venv_env, stderr_files[0],
                log_tag=LOG_TAG,
            )
            judged_proc = _spawn_judged(
                judged_path, 1, H, W, labels[1], venv_env, stderr_files[1]
            )

        seat = _JudgedSeat(
            judged_proc,
            first_reply_s=first_reply_s,
            normal_reply_s=normal_reply_s,
            fault_budget=fault_budget,
        )
        guard = _MemoryGuard(seat.pgid, memory_cap_bytes)
        guard.start()

        transition = make_transition(env)

        while turn < turn_limit:
            if guard.tripped:
                forfeit = RejectReason.MEMORY_CAP
                break

            obs_0 = get_obs(state, 0)
            obs_1 = get_obs(state, 1)

            if judged_seat == 0:
                a_0, forfeit = seat.ask(obs_0, memory_tripped=lambda: guard.tripped)
                if forfeit is not None:
                    break
                a_1 = ask_agent(opponent_proc, obs_1)
            else:
                a_0 = ask_agent(opponent_proc, obs_0)
                a_1, forfeit = seat.ask(obs_1, memory_tripped=lambda: guard.tripped)
                if forfeit is not None:
                    break

            state, info = transition(state, jnp.stack([a_0, a_1]))
            turn += 1

            if bool(info.is_done):
                winner_player = int(info.winner)
                break
        else:
            truncated = True

        if forfeit == RejectReason.MEMORY_CAP or (guard is not None and guard.tripped):
            forfeit = RejectReason.MEMORY_CAP
            # Opponent wins when the judged seat is killed for memory.
            winner_player = 1 - judged_seat
        elif forfeit in (
            RejectReason.CRASH,
            RejectReason.FAULT_BUDGET,
        ):
            winner_player = 1 - judged_seat
        elif forfeit is not None:
            winner_player = 1 - judged_seat

    finally:
        if guard is not None:
            guard.stop()
            peak_rss = max(peak_rss, guard.peak_rss_bytes)
            if guard.tripped and forfeit is None:
                forfeit = RejectReason.MEMORY_CAP
                winner_player = 1 - judged_seat
        if seat is not None:
            faults = seat.faults
            first_reject = seat.first_reject
        if judged_proc is not None:
            eof_voluntary, eof_exit_code, eof_forced = _close_judged(
                judged_proc, grace=eof_grace_s
            )
        if opponent_proc is not None:
            close_agent(opponent_proc)
        stderr_parts = [_read_temp_stderr(f) for f in stderr_files]
        for f in stderr_files:
            f.close()

    # Soft protocol faults do not stop the match; acceptance still requires
    # zero faults, voluntary status-0 EOF, and no memory trip.
    reason = RejectReason.ACCEPTED
    if forfeit is not None:
        reason = forfeit
    elif peak_rss > memory_cap_bytes:
        reason = RejectReason.MEMORY_CAP
        winner_player = 1 - judged_seat
    elif first_reject is not None:
        reason = first_reject
    elif eof_forced:
        reason = RejectReason.EOF_HANG
    elif eof_exit_code not in (0, None) or not eof_voluntary:
        reason = RejectReason.EOF_NONZERO

    accepted = reason == RejectReason.ACCEPTED and faults == 0
    if accepted and (eof_forced or not eof_voluntary or eof_exit_code != 0):
        accepted = False
        reason = RejectReason.EOF_HANG if eof_forced else RejectReason.EOF_NONZERO
    if accepted and peak_rss > memory_cap_bytes:
        accepted = False
        reason = RejectReason.MEMORY_CAP

    winner = winner_seat(winner_player, truncated=truncated and forfeit is None)
    return SubmissionResult(
        accepted=accepted,
        reason=reason,
        faults=faults,
        peak_rss_bytes=peak_rss,
        eof_voluntary=eof_voluntary and not eof_forced,
        eof_exit_code=eof_exit_code,
        eof_forced=eof_forced,
        winner=winner,
        turns=turn,
        terminated=winner_player >= 0,
        truncated=truncated and forfeit is None,
        stderr="\n".join(stderr_parts),
    )


def run_submission_bundle(
    zip_path: Path,
    opponent_run: Path,
    *,
    seed: int = 0,
    mode: str = "competition",
    memory_cap_bytes: int = MEMORY_CAP_BYTES,
    extract_dir: Path | None = None,
    **kwargs,
) -> SubmissionResult:
    """Extract ``zip_path`` and run the judged harness against ``opponent_run``."""
    if extract_dir is None:
        tmp = tempfile.TemporaryDirectory(prefix="submission-")
        extract_dir = Path(tmp.name)
        try:
            run_sh = extract_bundle(zip_path, extract_dir)
            return run_submission_match(
                run_sh,
                opponent_run,
                seed=seed,
                mode=mode,
                memory_cap_bytes=memory_cap_bytes,
                **kwargs,
            )
        finally:
            tmp.cleanup()
    run_sh = extract_bundle(zip_path, extract_dir)
    return run_submission_match(
        run_sh,
        opponent_run,
        seed=seed,
        mode=mode,
        memory_cap_bytes=memory_cap_bytes,
        **kwargs,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run an extracted submission zip under the judge process contract. "
            "Does not store games or update ratings."
        )
    )
    parser.add_argument("bundle", type=Path, help="path to submission zip")
    parser.add_argument(
        "--opponent",
        type=Path,
        required=True,
        help="path to opponent run.sh",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--mode",
        default="competition",
        help="must be competition (default: competition)",
    )
    parser.add_argument(
        "--memory-cap-bytes",
        type=int,
        default=MEMORY_CAP_BYTES,
        help=f"process-tree RSS cap (default: {MEMORY_CAP_BYTES})",
    )
    parser.add_argument(
        "--eof-grace",
        type=float,
        default=EOF_GRACE_SECONDS,
        help=f"seconds to wait for voluntary EOF exit (default: {EOF_GRACE_SECONDS})",
    )
    parser.add_argument(
        "--judged-seat",
        type=int,
        choices=(0, 1),
        default=0,
        help="seat index for the judged bundle (default: 0)",
    )
    args = parser.parse_args(argv)

    result = run_submission_bundle(
        args.bundle,
        args.opponent,
        seed=args.seed,
        mode=args.mode,
        memory_cap_bytes=args.memory_cap_bytes,
        eof_grace_s=args.eof_grace,
        judged_seat=args.judged_seat,
    )
    status = "ACCEPT" if result.accepted else "REJECT"
    print(
        f"[{LOG_TAG}] {status} reason={result.reason.value} "
        f"faults={result.faults} peak_rss={result.peak_rss_bytes} "
        f"eof_voluntary={result.eof_voluntary} eof_code={result.eof_exit_code} "
        f"eof_forced={result.eof_forced} winner={result.winner} turns={result.turns}"
    )
    return 0 if result.accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
