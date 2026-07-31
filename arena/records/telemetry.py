"""
Parse `[telemetry]` lines from bot stderr and merge them into a GameRecord.

Wire format producer: `bots/_common/wire.py:_telemetry_line`. Both sides must
change together — the fixed prefix here (`player`, `turn`, `my_land`,
`my_army`, `opp_land`, `opp_army`) mirrors that function's format string, and
any trailing `key=value` pairs a bot emits land in `BotTelemetry.extras` and
then in `GameRecord.metrics` as `<key>_a` / `<key>_b`.

See docs/arena/game-record-schema.md for the stored field names.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from arena.records.store import (
    CURRENT_SCHEMA_VERSION,
    GameRecord,
    duration_seconds_between,
    make_game_id,
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


def record_from_match_result(
    result,
    *,
    bot_a: str,
    bot_b: str,
    seed: int,
    mode: str,
    round_name: str,
    bot_a_content_hash: str,
    bot_b_content_hash: str,
    engine_version: str,
    started_at: str,
    finished_at: str,
) -> GameRecord:
    """
    Build a stored GameRecord from a CompetitionMatchResult.

    Applies bot telemetry from `result.stderr`, so callers only supply the
    identity and timing fields the match itself does not carry. The identity
    fields are required — `GameRecord` rejects a record whose bot hashes or
    engine era are unknown, because it could never be rated (schema v4).
    """
    record = GameRecord(
        game_id=make_game_id(bot_a, bot_b, seed),
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
        terminated=result.terminated,
        truncated=result.truncated,
        started_at=started_at,
        finished_at=finished_at,
        schema_version=CURRENT_SCHEMA_VERSION,
        duration_seconds=duration_seconds_between(started_at, finished_at),
        castles_built_a=result.castles_built_a,
        castles_built_b=result.castles_built_b,
    )
    apply_telemetry_to_record(record, parse_bot_telemetry(result.stderr or ""))
    return record
