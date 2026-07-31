"""Picklable ProcessPool worker for competition tournament matches.

Keep this module free of top-level JAX imports so `worker_initializer` can pin
CPU threads before the first match imports competition_match.

Workers never hash a roster and never write the version registry: the parent
does both once, before the pool starts, and passes the results down. A worker
that hashed for itself would race an edit made mid-round, and a worker that
wrote the registry would race its siblings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from arena.records.store import (
    GameRecord,
    bot_id_from_run_sh,
    save_game,
)


def _required(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if value is None or not str(value).strip():
        raise ValueError(f"worker payload is missing {key!r}; the parent must supply it")
    return str(value)


def run_one_worker(payload: dict[str, Any]) -> GameRecord:
    """One in-process competition match + store (ProcessPool entry point)."""
    from arena.matches.competition import run_competition_match
    from arena.records.registry import Registry
    from arena.records.telemetry import record_from_match_result

    a = Path(payload["bot_a_run"])
    b = Path(payload["bot_b_run"])
    seed = int(payload["seed"])
    games_dir = Path(payload["games_dir"])
    mode = str(payload.get("mode", "competition"))
    timeout = payload.get("timeout")
    bot_a = bot_id_from_run_sh(a)
    bot_b = bot_id_from_run_sh(b)

    round_name = _required(payload, "round")
    hash_a = _required(payload, "bot_a_content_hash")
    hash_b = _required(payload, "bot_b_content_hash")
    engine = _required(payload, "engine_version")

    # Assert-only: fail the match rather than store a game whose bots have no
    # reviewable version. The parent registered them before the pool started.
    registry = Registry()
    registry.require_registered(bot_a, hash_a)
    registry.require_registered(bot_b, hash_b)

    result = run_competition_match(
        a,
        b,
        seed=seed,
        mode=mode,
        timeout=float(timeout) if timeout is not None else None,
    )

    record = record_from_match_result(
        result,
        bot_a=bot_a,
        bot_b=bot_b,
        seed=seed,
        mode=mode,
        round_name=round_name,
        bot_a_content_hash=hash_a,
        bot_b_content_hash=hash_b,
        engine_version=engine,
    )
    save_game(record, games_dir)
    return record
