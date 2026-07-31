"""Picklable ProcessPool worker for competition tournament matches.

Keep this module free of top-level JAX imports so `worker_initializer` can pin
CPU threads before the first match imports competition_match.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from arena.store import (
    GameRecord,
    bot_id_from_run_sh,
    duration_seconds_between,
    git_commit_or_tag,
    make_game_id,
    save_game,
    utc_now_iso,
)


def run_one_worker(payload: dict[str, Any]) -> GameRecord:
    """One in-process competition match + store (ProcessPool entry point)."""
    from arena.competition_match import run_competition_match
    from arena.run_match import apply_telemetry_to_record, parse_bot_telemetry

    a = Path(payload["bot_a_run"])
    b = Path(payload["bot_b_run"])
    seed = int(payload["seed"])
    games_dir = Path(payload["games_dir"])
    mode = str(payload.get("mode", "competition"))
    timeout = payload.get("timeout")
    commit = str(payload.get("commit") or git_commit_or_tag())
    bot_a = bot_id_from_run_sh(a)
    bot_b = bot_id_from_run_sh(b)

    started_at = utc_now_iso()
    result = run_competition_match(
        a,
        b,
        seed=seed,
        mode=mode,
        timeout=float(timeout) if timeout is not None else None,
    )
    finished_at = utc_now_iso()

    record = GameRecord(
        game_id=make_game_id(bot_a, bot_b, seed),
        seed=seed,
        mode=mode,
        bot_a=bot_a,
        bot_b=bot_b,
        bot_a_commit_or_tag=commit,
        bot_b_commit_or_tag=commit,
        winner=result.winner,
        turns=result.turns,
        terminated=result.terminated,
        truncated=result.truncated,
        started_at=started_at,
        finished_at=finished_at,
        schema_version=2,
        duration_seconds=duration_seconds_between(started_at, finished_at),
        castles_built_a=result.castles_built_a,
        castles_built_b=result.castles_built_b,
    )
    apply_telemetry_to_record(record, parse_bot_telemetry(result.stderr or ""))
    save_game(record, games_dir)
    return record
