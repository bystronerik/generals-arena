"""
Per-turn probe for `blitz`: phase, strike count, and whether the enemy general is believed found.

Loaded only by `arena.instrument.runner` on recorded matches. Passive: it reads
attributes and returns them, never touching the agent's state. **Not** part of
the bot's source closure, hash, or submission bundle, and never imported from
`agent.py` — `arena.records.fingerprint` raises if it ever is.

Keys are declared in `arena.records.telemetry_schema`; an undeclared one fails
the recorded match rather than landing untyped.
"""
from __future__ import annotations


def extras(agent) -> dict:
    mem = agent._core.memory
    return {
        "phase": mem.phase,
        "strikes": mem.strikes,
        "enemy_general_sighted": 1 if mem.belief.enemy_general else 0,
    }
