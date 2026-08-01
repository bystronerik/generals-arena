"""
Per-turn probe for `macaria`: the core's phase, and what the search did.

Loaded only by `arena.instrument.runner` on recorded matches. Passive: it reads
attributes and returns them, never touching the agent's state. **Not** part of
the bot's source closure, hash, or submission bundle, and never imported from
`agent.py` — `arena.records.fingerprint` raises if it ever is.

`(searched, overrode)` is reported as a pair on purpose: it separates "the
search never ran" from "it ran and agreed with the core", which are different
defects presenting the same symptom (macaria plays exactly like blitz).

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
        "searched": 1 if agent.searched else 0,
        "overrode": 1 if agent.overrode else 0,
        "search_iters": agent.search_iters,
        "move_ms": agent.move_ms,
    }
