"""
Per-turn probe for `yankee`: which core is driving, why, and what the search did.

Loaded only by `arena.instrument.runner` on recorded matches. Passive: it reads
attributes and returns them, never touching the agent's state. **Not** part of
the bot's source closure, hash, or submission bundle, and never imported from
`agent.py` — `arena.records.fingerprint` raises if it ever is.

Keys are declared in `arena.records.telemetry_schema`; an undeclared one fails
the recorded match rather than landing untyped.

The evidence keys are the point. A wrong *label* and a wrong *counter* are
different defects with the same symptom (a lost game), and telling them apart
after the fact needs the inputs the label was computed from, not just the
label: `stack_near` and `duel_turn` are the whole aggressor case, and
`structures` is the castle programme the classifier deliberately does not act
on — recorded so that decision stays falsifiable.

yankee extends the same discipline to the search. `searched` and `overrode` are
separate keys because "the search never ran" and "it ran and agreed with the
core" are different facts that both look like "the search did nothing" from
outside — and the first version of `search.py` was diagnosed on exactly that
distinction. `search_iters` records what 40 ms actually bought, which is a
property of the machine rather than of the bot and so has to be measured.

The agent's counters are cumulative, so each sample is a difference against the
previous one. That is the only state this probe keeps, and it never writes any
to the agent. It is held in a positional list rather than a dict because
`tests/test_telemetry_schema.py` reads the emitted key set straight out of this
file's source, and a dict literal's keys would read as undeclared telemetry.
"""
from __future__ import annotations

from yankee.classifier import structure_estimate

_LAST_SEARCHED, _LAST_OVERRIDES, _LAST_ITERS = 0, 1, 2
_last = [0, 0, 0]


def extras(agent) -> dict:
    pressure = agent.pressure
    structures = structure_estimate(agent.model)
    stats = agent.search.stats

    searched = stats.turns_searched - _last[_LAST_SEARCHED]
    overrode = stats.overrides - _last[_LAST_OVERRIDES]
    iters = stats.iterations - _last[_LAST_ITERS]
    _last[_LAST_SEARCHED] = stats.turns_searched
    _last[_LAST_OVERRIDES] = stats.overrides
    _last[_LAST_ITERS] = stats.iterations

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
        # The search, separated into ran / disagreed / how hard it worked.
        "searched": 1 if searched > 0 else 0,
        "overrode": 1 if overrode > 0 else 0,
        "search_iters": max(0, iters),
        "move_ms": int(round(agent.last_move_ms)),
    }
