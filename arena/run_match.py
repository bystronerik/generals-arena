"""Wrap competition-module matchup.py and store the game record."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from arena.store import (
    GAMES_DIR,
    REPO_ROOT,
    GameRecord,
    bot_id_from_run_sh,
    git_commit_or_tag,
    make_game_id,
    save_game,
    utc_now_iso,
)

MATCHUP_PY = REPO_ROOT / "competition-module" / "competition" / "matchup.py"

_WIN_RE = re.compile(
    r"\[matchup\] turn (?P<turns>\d+): player (?P<player>\d+) captured"
)
_DRAW_RE = re.compile(
    r"\[matchup\] turn (?P<turns>\d+): truncated"
)


@dataclass
class MatchResult:
    winner: str  # a | b | draw
    turns: int
    terminated: bool
    truncated: bool
    stdout: str
    stderr: str
    returncode: int


def _venv_path_env() -> dict[str, str]:
    env = os.environ.copy()
    venv_bin = str(Path(sys.executable).resolve().parent)
    env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    return env


def parse_matchup_output(combined: str) -> tuple[str, int, bool, bool]:
    """Parse matchup stdout/stderr into (winner, turns, terminated, truncated)."""
    win = None
    for match in _WIN_RE.finditer(combined):
        win = match
    draw = None
    for match in _DRAW_RE.finditer(combined):
        draw = match

    if win is not None:
        player = int(win.group("player"))
        if player not in (0, 1):
            raise ValueError(f"unexpected winner player id: {player}")
        winner = "a" if player == 0 else "b"
        return winner, int(win.group("turns")), True, False
    if draw is not None:
        return "draw", int(draw.group("turns")), False, True
    raise ValueError(
        "could not parse matchup result; expected a capture or truncation line"
    )


def run_matchup(
    bot_a_run: Path,
    bot_b_run: Path,
    *,
    seed: int = 0,
    mode: str = "competition",
    timeout: float | None = None,
) -> MatchResult:
    """Run matchup.py --mode competition (or the given mode)."""
    if mode != "competition":
        raise ValueError(
            f"arena matches require mode='competition' (got {mode!r})"
        )
    if not MATCHUP_PY.exists():
        raise FileNotFoundError(f"matchup.py not found: {MATCHUP_PY}")
    a = bot_a_run.resolve()
    b = bot_b_run.resolve()
    if not a.exists():
        raise FileNotFoundError(f"bot A run.sh not found: {a}")
    if not b.exists():
        raise FileNotFoundError(f"bot B run.sh not found: {b}")

    cmd = [
        sys.executable,
        str(MATCHUP_PY),
        str(a),
        str(b),
        "--mode",
        mode,
        "--seed",
        str(seed),
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(MATCHUP_PY.parent),
        capture_output=True,
        text=True,
        env=_venv_path_env(),
        timeout=timeout,
        check=False,
    )
    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if proc.returncode != 0:
        raise RuntimeError(
            f"matchup.py exited {proc.returncode}\n"
            f"--- stdout ---\n{proc.stdout}\n"
            f"--- stderr ---\n{proc.stderr}"
        )
    winner, turns, terminated, truncated = parse_matchup_output(combined)
    return MatchResult(
        winner=winner,
        turns=turns,
        terminated=terminated,
        truncated=truncated,
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
        returncode=proc.returncode,
    )


def run_and_store(
    bot_a_run: Path,
    bot_b_run: Path,
    *,
    seed: int = 0,
    mode: str = "competition",
    games_dir: Path | None = None,
    bot_a_id: str | None = None,
    bot_b_id: str | None = None,
    bot_a_commit: str | None = None,
    bot_b_commit: str | None = None,
    timeout: float | None = None,
    update_ratings: bool = False,
) -> GameRecord:
    """Run one competition match, store JSON, optionally update ratings."""
    a_path = bot_a_run.resolve()
    b_path = bot_b_run.resolve()
    bot_a = bot_a_id or bot_id_from_run_sh(a_path)
    bot_b = bot_b_id or bot_id_from_run_sh(b_path)
    commit_a = bot_a_commit if bot_a_commit is not None else git_commit_or_tag()
    commit_b = bot_b_commit if bot_b_commit is not None else commit_a

    started_at = utc_now_iso()
    result = run_matchup(a_path, b_path, seed=seed, mode=mode, timeout=timeout)
    finished_at = utc_now_iso()

    record = GameRecord(
        game_id=make_game_id(bot_a, bot_b, seed),
        seed=seed,
        mode=mode,
        bot_a=bot_a,
        bot_b=bot_b,
        bot_a_commit_or_tag=commit_a,
        bot_b_commit_or_tag=commit_b,
        winner=result.winner,  # type: ignore[arg-type]
        turns=result.turns,
        terminated=result.terminated,
        truncated=result.truncated,
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
        from arena.ratings import rate_stored_game

        rate_stored_game(record)
        print("[run_match] ratings updated")

    # Relay matchup logs after the store summary (stdout only, keep order stable).
    if result.stderr:
        print(result.stderr, end="" if result.stderr.endswith("\n") else "\n")
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a competition matchup and store data/games/<game_id>.json."
    )
    parser.add_argument("bot_a", type=Path, help="path to bot A run.sh")
    parser.add_argument("bot_b", type=Path, help="path to bot B run.sh")
    parser.add_argument(
        "--mode",
        default="competition",
        help="must be competition (default: competition)",
    )
    parser.add_argument("--seed", type=int, default=0, help="matchup seed (default: 0)")
    parser.add_argument(
        "--games-dir",
        type=Path,
        default=GAMES_DIR,
        help=f"output directory (default: {GAMES_DIR})",
    )
    parser.add_argument("--bot-a-id", default=None, help="override bot A id label")
    parser.add_argument("--bot-b-id", default=None, help="override bot B id label")
    parser.add_argument(
        "--bot-a-commit",
        default=None,
        help="override bot A commit/tag pin",
    )
    parser.add_argument(
        "--bot-b-commit",
        default=None,
        help="override bot B commit/tag pin",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="optional subprocess timeout in seconds",
    )
    parser.add_argument(
        "--update-ratings",
        action="store_true",
        help="after storing the game, apply Elo and rewrite leaderboard",
    )
    args = parser.parse_args(argv)

    run_and_store(
        args.bot_a,
        args.bot_b,
        seed=args.seed,
        mode=args.mode,
        games_dir=args.games_dir,
        bot_a_id=args.bot_a_id,
        bot_b_id=args.bot_b_id,
        bot_a_commit=args.bot_a_commit,
        bot_b_commit=args.bot_b_commit,
        timeout=args.timeout,
        update_ratings=args.update_ratings,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
