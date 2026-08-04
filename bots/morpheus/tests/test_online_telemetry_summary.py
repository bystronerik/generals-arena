"""Part 09a — proposal / per-turn search telemetry in online measurement rows."""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.morpheus

from runtime import COST_COMPONENTS
from training.morpheus.measure_online import _summarize_scenario


def _turn(
    *,
    turn: int,
    first: bool,
    move_ms: float,
    belief_ms: float,
    leaf_ms: float,
    selection_calls: int,
    leaf_calls: int,
    unique_info: int,
    unique_policy: int,
    singleton: int,
) -> dict:
    component_ms = {name: 0.0 for name in COST_COMPONENTS}
    component_calls = {name: 0 for name in COST_COMPONENTS}
    component_ms["belief_proposal"] = belief_ms
    component_ms["leaf_batch"] = leaf_ms
    component_calls["selection"] = selection_calls
    component_calls["leaf_batch"] = leaf_calls
    return {
        "turn": turn,
        "first": first,
        "move_ms": move_ms,
        "completed_simulations": 8 if not first else 0,
        "belief_plus_root_ok": 1,
        "component_ms": component_ms,
        "component_calls": component_calls,
        "proposal_n_particles": 32,
        "proposal_n_singleton_particles": singleton,
        "proposal_n_unique_info_keys": unique_info,
        "proposal_n_unique_policy_inputs": unique_policy,
        "proposal_n_policy_batches": 0 if unique_policy == 0 else 2,
        "forward_by_consumer": {},
        "cost_search_ms": leaf_ms + float(selection_calls),
    }


def test_summarize_scenario_exposes_turn_and_board_telemetry():
    rows = [
        {
            "side": 18,
            "turns_played": 3,
            "first_move_ms": [80.0],
            "normal_move_ms": [120.0, 130.0],
            "normal_sims": [8, 8],
            "components": {name: [1.0] for name in COST_COMPONENTS},
            "turns": [
                _turn(
                    turn=0,
                    first=True,
                    move_ms=80.0,
                    belief_ms=0.0,
                    leaf_ms=0.0,
                    selection_calls=0,
                    leaf_calls=0,
                    unique_info=0,
                    unique_policy=0,
                    singleton=0,
                ),
                _turn(
                    turn=1,
                    first=False,
                    move_ms=120.0,
                    belief_ms=10.0,
                    leaf_ms=20.0,
                    selection_calls=2,
                    leaf_calls=1,
                    unique_info=30,
                    unique_policy=0,
                    singleton=32,
                ),
                _turn(
                    turn=2,
                    first=False,
                    move_ms=130.0,
                    belief_ms=40.0,
                    leaf_ms=25.0,
                    selection_calls=4,
                    leaf_calls=2,
                    unique_info=28,
                    unique_policy=12,
                    singleton=0,
                ),
            ],
            "belief_plus_root_ok_rate": 1.0,
            "recovery_rate": 0.0,
            "collapsed_rate": 0.0,
            "deadline_faults": 0,
            "min_sims_misses": 0,
            "peak_rss_bytes": 1000,
        },
        {
            "side": 21,
            "turns_played": 2,
            "first_move_ms": [90.0],
            "normal_move_ms": [140.0],
            "normal_sims": [6],
            "components": {name: [2.0] for name in COST_COMPONENTS},
            "turns": [
                _turn(
                    turn=0,
                    first=True,
                    move_ms=90.0,
                    belief_ms=0.0,
                    leaf_ms=0.0,
                    selection_calls=0,
                    leaf_calls=0,
                    unique_info=0,
                    unique_policy=0,
                    singleton=0,
                ),
                _turn(
                    turn=1,
                    first=False,
                    move_ms=140.0,
                    belief_ms=50.0,
                    leaf_ms=30.0,
                    selection_calls=3,
                    leaf_calls=1,
                    unique_info=31,
                    unique_policy=0,
                    singleton=32,
                ),
            ],
            "belief_plus_root_ok_rate": 1.0,
            "recovery_rate": 0.0,
            "collapsed_rate": 0.0,
            "deadline_faults": 0,
            "min_sims_misses": 0,
            "peak_rss_bytes": 2000,
        },
    ]

    summary = _summarize_scenario(rows)
    assert summary["proposal_unique_info_keys_max"] == 31
    assert summary["proposal_unique_policy_inputs_max"] == 12
    assert summary["turn_component_call_mean"]["selection"] == pytest.approx(
        (0 + 2 + 4 + 0 + 3) / 5
    )
    assert summary["turn_component_total_mean_ms"]["belief_proposal"] == pytest.approx(
        (0 + 10 + 40 + 0 + 50) / 5
    )
    assert "18" in summary["by_board"]
    assert "21" in summary["by_board"]
    assert summary["by_board"]["18"]["proposal_unique_policy_inputs_max"] == 12
    assert summary["by_board"]["21"]["proposal_unique_policy_inputs_max"] == 0
    assert summary["component_total_over_move_mean"] > 0.0
