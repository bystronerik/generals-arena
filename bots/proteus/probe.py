"""
Per-turn probe for `proteus`: which sub-strategy is driving, and how often it has switched.

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
        "active": agent.switcher.current,
        "label": agent.switcher.label,
        "switches": len(agent.switcher.history),
    }
