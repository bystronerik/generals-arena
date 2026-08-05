"""Part 13 compute accounting — formulas, layout selection, cadence rule."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.morpheus

from training.morpheus.compute.accounting import (
    A100_BUDGET_HOURS,
    A100ChargeSheet,
    aggregate_a100_hours,
    games_per_checkpoint,
    learner_starved,
    select_layout,
    steady_state_rates,
)
from training.morpheus.compute.cadence import (
    evaluate_cadence_candidate,
    select_smallest_useful_cadence,
)
from training.morpheus.compute.qualify import decide_pass, run_qualification

REPO = Path(__file__).resolve().parents[3]


def test_steady_state_formulas_match_spec():
    rates = steady_state_rates(
        completed_games=10,
        worker_hours=2.0,
        workers=4,
        positions_per_game=20.0,
        self_play_wall_hours=3.0,
        checkpoint_count=5,
    )
    assert rates.games_per_worker_hour == 5.0
    assert rates.aggregate_games_per_hour == 20.0
    assert rates.positions_supply_per_hour == 400.0
    assert rates.total_games == 60.0
    assert rates.games_per_checkpoint == 12


def test_games_per_checkpoint_floors():
    assert games_per_checkpoint(99.9, 10) == 9
    with pytest.raises(ValueError):
        games_per_checkpoint(10, 0)


def test_a100_charge_sheet_sums_and_budget():
    sheet = A100ChargeSheet(
        jax_preflight=1.0,
        throughput_qualification=2.0,
        learning_curve_pilot=3.0,
        objective_ablations=4.0,
        gpu_self_play=5.0,
        main_training=20.0,
        deployment_calibration=6.0,
    )
    assert sheet.total() == 41.0
    assert sheet.fits_budget()
    over = A100ChargeSheet(main_training=A100_BUDGET_HOURS + 1.0)
    assert not over.fits_budget()
    acct = aggregate_a100_hours(sheet)
    assert acct["total_a100_hours"] == 41.0
    assert acct["fits_budget"] is True
    assert set(acct["lines"]) == {
        "jax_preflight",
        "throughput_qualification",
        "learning_curve_pilot",
        "objective_ablations",
        "gpu_self_play",
        "main_training",
        "deployment_calibration",
    }


def test_select_layout_prefers_games_per_cpu_hour_then_latency():
    rows = [
        {
            "name": "slow",
            "completed_games": 4,
            "cpu_hours": 2.0,
            "mean_game_latency_s": 10.0,
        },
        {
            "name": "fast",
            "completed_games": 4,
            "cpu_hours": 1.0,
            "mean_game_latency_s": 20.0,
        },
        {
            "name": "tie-low-latency",
            "completed_games": 4,
            "cpu_hours": 1.0,
            "mean_game_latency_s": 5.0,
        },
    ]
    pick = select_layout(rows)
    assert pick["selected_name"] == "tie-low-latency"


def test_learner_starved_when_supply_below_consume():
    assert learner_starved(
        positions_supply_per_hour=100.0, positions_consume_per_hour=200.0
    )
    assert not learner_starved(
        positions_supply_per_hour=200.0, positions_consume_per_hour=100.0
    )
    assert not learner_starved(
        positions_supply_per_hour=0.0, positions_consume_per_hour=0.0
    )


def test_cadence_requires_all_three_conditions():
    # Strong WDL for two classes (32 wins + 32 losses each).
    class_wdl = {
        3: {"wins": 32, "losses": 32, "draws": 0},
        4: {"wins": 40, "losses": 40, "draws": 0},
    }
    fail_missing = evaluate_cadence_candidate(
        name="a",
        games_per_checkpoint=100,
        checkpoint_count=4,
    )
    assert fail_missing["useful"] is False

    fail_cal = evaluate_cadence_candidate(
        name="b",
        games_per_checkpoint=100,
        checkpoint_count=4,
        class_wdl=class_wdl,
        active_classes=[3, 4],
        belief_calibration_regressed=True,
        pairwise_verdict="improvement",
    )
    assert fail_cal["useful"] is False
    assert fail_cal["curriculum"]["ok"] is True

    fail_pair = evaluate_cadence_candidate(
        name="c",
        games_per_checkpoint=100,
        checkpoint_count=4,
        class_wdl=class_wdl,
        active_classes=[3, 4],
        belief_calibration_regressed=False,
        pairwise_verdict="unproven",
    )
    assert fail_pair["useful"] is False

    ok = evaluate_cadence_candidate(
        name="d",
        games_per_checkpoint=50,
        checkpoint_count=8,
        class_wdl=class_wdl,
        active_classes=[3, 4],
        belief_calibration_regressed=False,
        pairwise_verdict="improvement",
    )
    assert ok["useful"] is True

    pick = select_smallest_useful_cadence([fail_pair, ok, fail_cal])
    assert pick["useful_found"] is True
    assert pick["selected_name"] == "d"


def test_decide_pass_requires_useful_cadence():
    body = {
        "selected_layout": {"backend": "cpu", "seat_search": "sequential"},
        "workers_per_a100": 8,
        "learner_starved": False,
        "rates": {"games_per_checkpoint": 10, "checkpoint_count": 4},
        "a100_accounting": {"fits_budget": True},
        "cadence": {"useful_found": False},
        "deployment_calibration": {"fits": True},
        "fallback_plan": {"selected": "narrower_non_promotable_research_scope"},
    }
    d = decide_pass(body)
    assert d["verdict"] == "no"
    assert d["checks"]["useful_cadence_found"] is False


def test_run_qualification_smoke_writes_no_report(tmp_path: Path):
    cfg = REPO / "training/morpheus/configs/modal-qualification.json"
    # Use a tiny inline layout override via pre-supplied measurements.
    fake_layouts = [
        {
            "name": "cpu1-seq",
            "backend": "cpu",
            "physical_cores_per_game": 1,
            "seat_search": "sequential",
            "concurrent_games": 1,
            "completed_games": 2,
            "wall_s": 10.0,
            "cpu_hours": 10.0 / 3600.0,
            "worker_hours": 10.0 / 3600.0,
            "mean_game_latency_s": 5.0,
            "positions_per_game": 16.0,
            "games_per_worker_hour": 2 / (10.0 / 3600.0),
        },
        {
            "name": "cpu2-par",
            "backend": "cpu",
            "physical_cores_per_game": 2,
            "seat_search": "parallel",
            "concurrent_games": 1,
            "completed_games": 2,
            "wall_s": 4.0,
            "cpu_hours": 4.0 / 3600.0,
            "worker_hours": 4.0 / 3600.0,
            "mean_game_latency_s": 2.0,
            "positions_per_game": 16.0,
            "games_per_worker_hour": 2 / (4.0 / 3600.0),
        },
    ]
    json_path = tmp_path / "morpheus-modal-qualification.json"
    md_path = tmp_path / "morpheus-modal-qualification.md"
    report = run_qualification(
        cfg,
        output_dir=tmp_path / "out",
        json_path=json_path,
        md_path=md_path,
        layout_results=fake_layouts,
    )
    assert report["decision"]["verdict"] == "no"
    # cpu2-par: 2 games / (4/3600) cpu-h = 1800/h vs cpu1-seq 720/h.
    assert report["selected_layout"]["name"] == "cpu2-par"
    assert report["workers_per_a100"] == 16
    assert report["rates"]["games_per_checkpoint"] > 0
    assert "main_training" in report["a100_accounting"]["lines"]
    assert json_path.is_file()
    assert md_path.is_file()
    assert "Verdict: **no**" in md_path.read_text(encoding="utf-8")
