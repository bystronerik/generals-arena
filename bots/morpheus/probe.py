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
        "proposal_n_unique_info_keys": int(
            getattr(agent, "proposal_n_unique_info_keys", 0)
        ),
        "proposal_n_unique_policy_inputs": int(
            getattr(agent, "proposal_n_unique_policy_inputs", 0)
        ),
        "proposal_n_singleton_particles": int(
            getattr(agent, "proposal_n_singleton_particles", 0)
        ),
        "proposal_n_policy_batches": int(
            getattr(agent, "proposal_n_policy_batches", 0)
        ),
        "search_selection_calls": int(
            (getattr(agent, "component_calls", {}) or {}).get("selection", 0)
        ),
        "search_leaf_batch_calls": int(
            (getattr(agent, "component_calls", {}) or {}).get("leaf_batch", 0)
        ),
        "search_enemy_prior_calls": int(
            (getattr(agent, "component_calls", {}) or {}).get("enemy_prior_batch", 0)
        ),
        "root_pass_prior_milli": int(getattr(agent, "root_pass_prior_milli", 0)),
        "root_top_action": int(getattr(agent, "root_top_action", -1)),
        "root_top_prior_milli": int(getattr(agent, "root_top_prior_milli", 0)),
        "chosen_action": int(getattr(agent, "chosen_action", -1)),
        "chosen_is_pass": int(getattr(agent, "chosen_is_pass", 0)),
        "policy_fallback_is_pass": int(getattr(agent, "policy_fallback_is_pass", 0)),
        "root_legal_nonpass": int(getattr(agent, "root_legal_nonpass", 0)),
        "has_root_result": int(getattr(agent, "has_root_result", 0)),
        "nn_top_action": int(getattr(agent, "nn_top_action", -1)),
        "nn_top_prior_milli": int(getattr(agent, "nn_top_prior_milli", 0)),
        "chosen_matches_nn_top": int(getattr(agent, "chosen_matches_nn_top", 0)),
        "chosen_in_nn_top3": int(getattr(agent, "chosen_in_nn_top3", 0)),
        "enemy_visible": int(getattr(agent, "enemy_visible", 0)),
    }
