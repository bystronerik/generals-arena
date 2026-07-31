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
    git_commit_or_tag,
    save_game,
    utc_now_iso,
)


def run_one_worker(payload: dict[str, Any]) -> GameRecord:
    """One in-process competition match + store (ProcessPool entry point)."""
    from arena.competition_match import run_competition_match
    from arena.fingerprint import bot_content_hash
    from arena.telemetry import record_from_match_result

    a = Path(payload["bot_a_run"])
    b = Path(payload["bot_b_run"])
    seed = int(payload["seed"])
    games_dir = Path(payload["games_dir"])
    mode = str(payload.get("mode", "competition"))
    timeout = payload.get("timeout")
    commit = str(payload.get("commit") or git_commit_or_tag())
    bot_a = bot_id_from_run_sh(a)
    bot_b = bot_id_from_run_sh(b)

    hash_a = str(payload.get("bot_a_content_hash") or bot_content_hash(a))
    hash_b = str(payload.get("bot_b_content_hash") or bot_content_hash(b))

    started_at = utc_now_iso()
    result = run_competition_match(
        a,
        b,
        seed=seed,
        mode=mode,
        timeout=float(timeout) if timeout is not None else None,
    )
    finished_at = utc_now_iso()

    record = record_from_match_result(
        result,
        bot_a=bot_a,
        bot_b=bot_b,
        seed=seed,
        mode=mode,
        bot_a_commit=commit,
        bot_b_commit=commit,
        started_at=started_at,
        finished_at=finished_at,
        bot_a_content_hash=hash_a,
        bot_b_content_hash=hash_b,
    )
    save_game(record, games_dir)
    return record
