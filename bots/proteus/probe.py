"""
Per-turn probe for `proteus`: which sub-strategy is driving, why, and on what
evidence.

Loaded only by `arena.instrument.runner` on recorded matches. Passive: it reads
attributes and returns them, never touching the agent's state. **Not** part of
the bot's source closure, hash, or submission bundle, and never imported from
`agent.py` — `arena.records.fingerprint` raises if it ever is.

Keys are declared in `arena.records.telemetry_schema`; an undeclared one fails
the recorded match rather than landing untyped.

The evidence keys are the point. A wrong *label* and a wrong *counter* are
different defects with the same symptom (a lost game), and telling them apart
after the fact needs the inputs the label was computed from, not just the
label: `stack_near` and `duel_turn` are the whole aggressor case, and `structures`
is the castle programme the classifier deliberately does not act on —
recorded so that decision stays falsifiable.
"""
from __future__ import annotations

from proteus.classifier import structure_estimate


def extras(agent) -> dict:
    pressure = agent.pressure
    structures = structure_estimate(agent.model)
    return {
        "active": agent.switcher.current,
        "label": agent.switcher.label,
        "switches": len(agent.switcher.history),
        # Evidence behind the aggressor branch.
        "stack_near": pressure.max_stack_near,
        "turns_near": pressure.turns_near,
        # -1 rather than absent: the schema types these INT, and "no fist yet"
        # is a measurement, not a gap.
        "duel_turn": -1 if pressure.duel_turn is None else pressure.duel_turn,
        # Reported, not acted on.
        "structures": -1 if structures is None else structures,
    }
