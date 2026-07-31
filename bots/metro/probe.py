"""
Per-turn probe for `metro`: its own castle count and whether it is pushing.

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
        # Deliberately not `castles_built`: that name is the engine's tally of
        # castle births. This is metro's own belief about the same quantity,
        # and the two disagree often enough to matter.
        "castles_built_probe": mem.castles_built,
        "pushing": 1 if mem.pushing else 0,
    }
