"""
Per-turn probe for Morpheus.

Loaded only by arena.instrument.runner on recorded matches. Never imported by
bot code. Keys must exist in arena.records.telemetry_schema.
"""
from __future__ import annotations


def extras(agent) -> dict:
    return {
        "move_ms": int(getattr(agent, "move_ms", 0)),
        "search_iters": int(getattr(agent, "search_iters", 0)),
        "completed_simulations": int(getattr(agent, "completed_simulations", 0)),
        "forward_equivalents": int(getattr(agent, "forward_equivalents", 0)),
        "belief_ess": int(getattr(agent, "belief_ess", 0)),
        "recovery": int(getattr(agent, "recovery", 0)),
        "tree_size": int(getattr(agent, "tree_size", 0)),
        "fallback_level": str(getattr(agent, "fallback_level", "pass")),
        "cost_belief_ms": int(getattr(agent, "cost_belief_ms", 0)),
        "cost_root_ms": int(getattr(agent, "cost_root_ms", 0)),
        "cost_search_ms": int(getattr(agent, "cost_search_ms", 0)),
        "cost_reply_ms": int(getattr(agent, "cost_reply_ms", 0)),
        "belief_plus_root_ok": int(getattr(agent, "belief_plus_root_ok", 0)),
    }
