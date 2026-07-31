"""
Build a stored `GameRecord` from a finished competition match.

Nothing here reads bot stderr. Bots emit no telemetry at all any more: the
`[telemetry]` line and every `telemetry_extras()` method were deleted so that
the submitted bundle and the rated closure are pure game logic
(docs/arena/recorder-plan.md constraint 2). The observations a record carries
now come from two places, both outside `bots/`:

- **the engine**, for finals and castle tallies — ground truth for both seats,
  available on every match whether or not it was recorded;
- **recorded trajectories**, for per-turn series and probe output — present
  only when the match ran with recording on.

Absent means not measured, never zero: an unrecorded match simply carries no
probe metrics. See docs/arena/game-record-schema.md.
"""

from __future__ import annotations

from typing import Any

from arena.records.store import (
    CURRENT_SCHEMA_VERSION,
    GameRecord,
)


def engine_metrics(result) -> dict[str, Any]:
    """
    The observations the engine can state on its own, for `metrics`.

    Every key is omitted when the engine did not measure it — a classic env
    builds no castles, and a match that stepped no turn has no finals.
    """
    metrics: dict[str, Any] = {}
    for key, value in (
        ("castles_built_a", result.castles_built_a),
        ("castles_built_b", result.castles_built_b),
        ("final_land_a", result.final_land_a),
        ("final_land_b", result.final_land_b),
        ("final_army_a", result.final_army_a),
        ("final_army_b", result.final_army_b),
    ):
        if value is not None:
            metrics[key] = value

    # Land margin is what separates "stalled level" from "stalled far ahead" on
    # a truncated draw, which is the only outcome `winner` cannot rank.
    land_a = metrics.get("final_land_a")
    land_b = metrics.get("final_land_b")
    if result.truncated and result.winner == "draw" and land_a is not None and land_b is not None:
        metrics["land_margin_a"] = land_a - land_b
        metrics["land_margin_b"] = land_b - land_a

    return metrics


def record_from_match_result(
    result,
    *,
    game_id: str,
    bot_a: str,
    bot_b: str,
    seed: int,
    mode: str,
    round_name: str,
    bot_a_content_hash: str,
    bot_b_content_hash: str,
    engine_version: str,
) -> GameRecord:
    """
    Build a stored GameRecord from a CompetitionMatchResult.

    Callers supply only the identity fields the match itself does not carry.
    Those are required — `GameRecord` rejects a record whose bot hashes or
    engine era are unknown, because it could never be rated.

    `game_id` is passed in rather than minted here: a trajectory is keyed by
    the game it belongs to, so the id has to exist before the first turn.
    """
    return GameRecord(
        game_id=game_id,
        seed=seed,
        mode=mode,
        round=round_name,
        bot_a=bot_a,
        bot_b=bot_b,
        bot_a_content_hash=bot_a_content_hash,
        bot_b_content_hash=bot_b_content_hash,
        engine_version=engine_version,
        winner=result.winner,
        turns=result.turns,
        truncated=result.truncated,
        schema_version=CURRENT_SCHEMA_VERSION,
        metrics=engine_metrics(result),
    )
