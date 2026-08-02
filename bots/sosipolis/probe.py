"""
Per-turn probe for sosipolis.

Loaded only by arena.instrument.runner on recorded matches. Never imported by
bot code. Keys must exist in arena.records.telemetry_schema.
"""
from __future__ import annotations


def extras(agent) -> dict:
    head = getattr(agent, "chain_head", None)
    if head is None:
        chain_tok = "none"
    else:
        chain_tok = f"{head[0]},{head[1]}"
    land50 = getattr(agent, "land_at_50", None)
    return {
        "phase": agent.phase,
        "enemy_general_sighted": agent.enemy_general_sighted,
        "searched": 1 if agent.searched else 0,
        "search_iters": agent.search_iters,
        "move_ms": agent.move_ms,
        "candidate_count": agent.candidate_count,
        "top_section": agent.top_section,
        "pocket_skips": agent.pocket_skips,
        "castles_built_probe": agent.castles_built_probe,
        "clock_phase": getattr(agent, "clock_phase", "wave"),
        "chain_head": chain_tok,
        "recall_fired": int(getattr(agent, "recall_fired", 0)),
        "land_at_50": -1 if land50 is None else int(land50),
    }
