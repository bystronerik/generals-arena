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
    duration_seconds_between,
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
_CASTLE_RE = re.compile(
    r"\[matchup\] castles built: (?P<a>\d+) \([^)]+\) vs (?P<b>\d+)"
)
_TELEMETRY_PREFIX_RE = re.compile(
    r"\[telemetry\] player=(?P<player>[01]) turn=(?P<turn>\d+) "
    r"my_land=(?P<my_land>\d+) my_army=(?P<my_army>\d+) "
    r"opp_land=(?P<opp_land>\d+) opp_army=(?P<opp_army>\d+)"
)
_EXTRA_KV_RE = re.compile(r"(\w+)=(\S+)")


@dataclass
class BotTelemetry:
    my_land: int
    my_army: int
    opp_land: int
    opp_army: int
    enemy_general_sighted: int | None = None
    first_sighting_turn: int | None = None
    extras: dict[str, str] | None = None

    def all_extras(self) -> dict[str, str]:
        merged: dict[str, str] = dict(self.extras or {})
        if self.enemy_general_sighted is not None:
            merged.setdefault("enemy_general_sighted", str(self.enemy_general_sighted))
        if self.first_sighting_turn is not None:
            merged.setdefault("first_sighting_turn", str(self.first_sighting_turn))
        return merged


def parse_castles_built(combined: str) -> tuple[int | None, int | None]:
    match = None
    for found in _CASTLE_RE.finditer(combined):
        match = found
    if match is None:
        return None, None
    return int(match.group("a")), int(match.group("b"))


def _parse_telemetry_extras(tail: str) -> dict[str, str]:
    return {match.group(1): match.group(2) for match in _EXTRA_KV_RE.finditer(tail)}


def _coerce_metric_value(key: str, raw: str) -> bool | int | str:
    if key == "enemy_general_sighted":
        return bool(int(raw))
    try:
        return int(raw)
    except ValueError:
        return raw


def parse_bot_telemetry(combined: str) -> dict[int, BotTelemetry]:
    """Return the last telemetry line per player id (0 or 1)."""
    last: dict[int, BotTelemetry] = {}
    for line in combined.splitlines():
        match = _TELEMETRY_PREFIX_RE.search(line)
        if match is None:
            continue
        player = int(match.group("player"))
        extras = _parse_telemetry_extras(line[match.end() :])
        sighted = extras.get("enemy_general_sighted")
        sighting_turn = extras.get("first_sighting_turn")
        last[player] = BotTelemetry(
            my_land=int(match.group("my_land")),
            my_army=int(match.group("my_army")),
            opp_land=int(match.group("opp_land")),
            opp_army=int(match.group("opp_army")),
            enemy_general_sighted=int(sighted) if sighted is not None else None,
            first_sighting_turn=int(sighting_turn) if sighting_turn is not None else None,
            extras=extras,
        )
    return last


def apply_telemetry_to_record(
    record: GameRecord,
    telemetry_by_player: dict[int, BotTelemetry],
) -> None:
    """Fill optional land/army and sighting metrics from bot stderr telemetry."""
    if not telemetry_by_player:
        return

    t0 = telemetry_by_player.get(0)
    t1 = telemetry_by_player.get(1)

    if t0 is not None:
        record.final_land_a = t0.my_land
        record.final_army_a = t0.my_army
    if t1 is not None:
        record.final_land_b = t1.my_land
        record.final_army_b = t1.my_army
    if t0 is not None and t1 is None:
        record.final_land_b = t0.opp_land
        record.final_army_b = t0.opp_army
    elif t1 is not None and t0 is None:
        record.final_land_a = t1.opp_land
        record.final_army_a = t1.opp_army

    metrics = dict(record.metrics)
    for player_id, suffix in ((0, "_a"), (1, "_b")):
        telemetry = telemetry_by_player.get(player_id)
        if telemetry is None:
            continue
        for key, raw in telemetry.all_extras().items():
            metrics[f"{key}{suffix}"] = _coerce_metric_value(key, raw)

    if (
        record.truncated
        and record.winner == "draw"
        and record.final_land_a is not None
        and record.final_land_b is not None
    ):
        metrics["land_margin_a"] = record.final_land_a - record.final_land_b
        metrics["land_margin_b"] = record.final_land_b - record.final_land_a

    record.metrics = metrics


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
    combined = (result.stdout or "") + "\n" + (result.stderr or "")

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
        schema_version=2,
        duration_seconds=duration_seconds_between(started_at, finished_at),
    )

    castles_a, castles_b = parse_castles_built(combined)
    record.castles_built_a = castles_a
    record.castles_built_b = castles_b
    apply_telemetry_to_record(record, parse_bot_telemetry(combined))
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
