"""
Per-turn probe for sosipolis.

Loaded only by arena.instrument.runner on recorded matches. Never imported by
bot code. Keys must exist in arena.records.telemetry_schema.
"""
from __future__ import annotations


def _tok(cell) -> str:
    if cell is None:
        return "none"
    if isinstance(cell, str):
        return cell
    return f"{cell[0]},{cell[1]}"


def extras(agent) -> dict:
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
        "chain_head": _tok(getattr(agent, "chain_head", None)),
        "recall_fired": int(getattr(agent, "recall_fired", 0)),
        "land_at_50": -1 if land50 is None else int(land50),
        "first_castle_turn": int(getattr(agent, "first_castle_turn", -1)),
        "enemy_gen": getattr(agent, "enemy_gen", "none"),
        "tip": _tok(getattr(agent, "tip", None)),
        "tip_army": int(getattr(agent, "tip_army", 0)),
        "tip_dist_goal": int(getattr(agent, "tip_dist_goal", -1)),
        "branch": getattr(agent, "branch", "mcts"),
        "toward": int(getattr(agent, "toward", 0)),
        "objective": _tok(getattr(agent, "objective", None)),
        "chain_continued": int(getattr(agent, "chain_continued", 0)),
        "hunt": _tok(getattr(agent, "hunt", None)),
        "muster": _tok(getattr(agent, "muster", None)),
        "tip_ready": int(getattr(agent, "tip_ready", 0)),
        "root_n": int(getattr(agent, "root_n", 0)),
        "overrode": int(getattr(agent, "overrode", 0)),
        "prior_rank": int(getattr(agent, "prior_rank", -1)),
        "best_visits": int(getattr(agent, "best_visits", 0)),
        "prior0_visits": int(getattr(agent, "prior0_visits", 0)),
        "contact_waypoint": _tok(getattr(agent, "contact_waypoint", None)),
        "contact_macro": getattr(agent, "contact_macro", "none"),
        "contact_commit_turn": int(getattr(agent, "contact_commit_turn", -1)),
        "contact_switches": int(getattr(agent, "contact_switches", 0)),
        "contact_macro_score": int(
            round(float(getattr(agent, "contact_macro_score", 0.0)) * 1000)
        ),
        "contact_candidate_mass": int(
            round(float(getattr(agent, "contact_candidate_mass", 0.0)) * 1000)
        ),
    }
