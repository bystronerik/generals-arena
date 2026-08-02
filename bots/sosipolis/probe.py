"""
Per-turn probe for sosipolis.

Loaded only by arena.instrument.runner on recorded matches. Never imported by
bot code. Keys must exist in arena.records.telemetry_schema.
"""
from __future__ import annotations


def extras(agent) -> dict:
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
    }
