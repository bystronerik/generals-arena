"""
Per-turn probe for `castle_builder`: whether the enemy general has been sighted.

Loaded only by `arena.instrument.runner` on recorded matches. Passive: it reads
attributes and returns them, never touching the agent's state. **Not** part of
the bot's source closure, hash, or submission bundle, and never imported from
`agent.py` — `arena.records.fingerprint` raises if it ever is.

Keys are declared in `arena.records.telemetry_schema`; an undeclared one fails
the recorded match rather than landing untyped.
"""
from __future__ import annotations


def extras(agent) -> dict:
    return {
        # The first-sighting *turn* is no longer reported here: it is the
        # `first_turn_true` reducer over this series (telemetry_schema).
        "enemy_general_sighted": 0 if agent.belief.first_sighting_turn is None else 1,
    }
