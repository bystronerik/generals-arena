"""Part 13 qualification orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from training.morpheus.compute.accounting import (
    A100_BUDGET_HOURS,
    A100ChargeSheet,
    aggregate_a100_hours,
    learner_starved,
    select_layout,
    steady_state_rates,
)
from training.morpheus.compute.cadence import (
    evaluate_cadence_candidate,
    select_smallest_useful_cadence,
)
from training.morpheus.compute.measure import measure_layout
from training.morpheus.compute.report import stamp_generated, write_report
from training.morpheus.self_play.driver import DriverConfig

REPO = Path(__file__).resolve().parents[3]


def _load_config(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _driver_from_config(cfg: Mapping[str, Any]) -> DriverConfig:
    sp = dict(cfg.get("self_play") or {})
    return DriverConfig.from_dict(sp)


def _load_prior_charges(cfg: Mapping[str, Any]) -> A100ChargeSheet:
    prior = dict(cfg.get("prior_a100_charges") or {})
    # Prefer live jax-preflight report when present.
    preflight = REPO / "docs/research/measurements/morpheus-jax-preflight.json"
    if preflight.is_file() and "jax_preflight" not in prior:
        try:
            data = json.loads(preflight.read_text(encoding="utf-8"))
            prior["jax_preflight"] = float(
                (data.get("accounting") or {}).get("a100_hours") or 0.0
            )
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
    ablation = REPO / "docs/research/measurements/morpheus-objective-ablation.json"
    if ablation.is_file() and "objective_ablations" not in prior:
        try:
            data = json.loads(ablation.read_text(encoding="utf-8"))
            prior["objective_ablations"] = float(
                (data.get("accounting") or {}).get("a100_hours") or 0.0
            )
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
    return A100ChargeSheet.from_mapping(prior)


def decide_pass(report_body: Mapping[str, Any]) -> dict[str, Any]:
    """Part 13 exit criterion."""
    acct = report_body.get("a100_accounting") or {}
    cadence = report_body.get("cadence") or {}
    layout = report_body.get("selected_layout") or {}
    rates = report_body.get("rates") or {}
    starved = bool(report_body.get("learner_starved"))
    deployment = report_body.get("deployment_calibration") or {}

    checks = {
        "exact_layout_stated": bool(layout.get("backend") and layout.get("seat_search")),
        "learner_not_starved": not starved,
        "useful_cadence_found": bool(cadence.get("useful_found")),
        "games_per_checkpoint_positive": int(rates.get("games_per_checkpoint") or 0) > 0,
        "checkpoint_count_positive": int(rates.get("checkpoint_count") or 0) > 0,
        "a100_fits_budget": bool(acct.get("fits_budget")),
        "deployment_calibration_fits": bool(deployment.get("fits", False)),
        "workers_per_a100_named": report_body.get("workers_per_a100") is not None,
    }
    passed = all(checks.values())
    fallback = None
    if not passed:
        fallback = dict(report_body.get("fallback_plan") or {})
        if not fallback:
            fallback = {
                "order": [
                    "reduce_training_particles_sims_or_depth",
                    "fewer_checkpoints_more_games_per_checkpoint",
                    "narrower_non_promotable_research_scope",
                ],
                "selected": "narrower_non_promotable_research_scope",
                "note": (
                    "No tested cadence passed curriculum, belief calibration, "
                    "and pairwise improvement together. Keep the deployment-matched "
                    "final phase. Do not start the main trainer."
                ),
            }
    return {
        "pass": passed,
        "verdict": "yes" if passed else "no",
        "checks": checks,
        "fallback": fallback,
        "note": (
            "Exact CPU/GPU layout, useful cadence, and 48h A100 budget all clear."
            if passed
            else (
                "Compute gate closed. "
                + str((fallback or {}).get("note") or "See fallback.")
            )
        ),
    }


def run_qualification(
    config_path: Path | str,
    *,
    output_dir: Path | str | None = None,
    json_path: Path | str | None = None,
    md_path: Path | str | None = None,
    layout_results: list[dict[str, Any]] | None = None,
    throughput_a100_hours: float = 0.0,
    learning_curve_a100_hours: float = 0.0,
    gpu_self_play_a100_hours: float = 0.0,
) -> dict[str, Any]:
    """Measure layouts (unless pre-supplied), score cadence, write the report."""
    cfg_path = Path(config_path)
    if not cfg_path.is_file():
        cfg_path = REPO / config_path
    cfg = _load_config(cfg_path)
    root = REPO
    out_root = Path(output_dir) if output_dir else root / "data/morpheus/compute_gate"
    out_root.mkdir(parents=True, exist_ok=True)

    driver = _driver_from_config(cfg)
    layouts_cfg = list(cfg.get("layouts") or [])
    if not layouts_cfg:
        raise ValueError("modal-qualification config requires layouts[]")

    measured: list[dict[str, Any]]
    if layout_results is not None:
        measured = list(layout_results)
    else:
        measured = []
        for layout in layouts_cfg:
            name = str(layout.get("name") or "layout")
            row = measure_layout(
                layout,
                base_config=driver,
                output=out_root / name,
                repo_root=root,
            )
            measured.append(row)

    selection = select_layout(measured)
    selected = dict(selection["selected"])

    schedule = dict(cfg.get("schedule") or {})
    workers = int(schedule.get("workers_per_a100") or cfg.get("workers_per_a100") or 1)
    self_play_wall_hours = float(schedule.get("self_play_wall_hours") or 24.0)
    checkpoint_candidates = list(
        schedule.get("checkpoint_count_candidates") or [4, 8, 16]
    )
    positions_consume = float(schedule.get("positions_consume_per_hour") or 0.0)
    main_train_h = float(schedule.get("main_training_a100_hours") or 0.0)
    deploy_h = float(schedule.get("deployment_calibration_a100_hours") or 0.0)
    deploy_fits = bool(schedule.get("deployment_calibration_fits", deploy_h > 0))

    # Prefer the first candidate count for rate table; evaluate all for cadence.
    primary_ckpt = int(checkpoint_candidates[0])
    rates = steady_state_rates(
        completed_games=int(selected["completed_games"]),
        worker_hours=float(selected["worker_hours"]),
        workers=workers,
        positions_per_game=float(selected["positions_per_game"]),
        self_play_wall_hours=self_play_wall_hours,
        checkpoint_count=primary_ckpt,
    )

    starved = learner_starved(
        positions_supply_per_hour=rates.positions_supply_per_hour,
        positions_consume_per_hour=positions_consume,
    )

    prior = _load_prior_charges(cfg)
    charges = A100ChargeSheet(
        jax_preflight=prior.jax_preflight,
        throughput_qualification=float(throughput_a100_hours),
        learning_curve_pilot=float(learning_curve_a100_hours),
        objective_ablations=prior.objective_ablations,
        gpu_self_play=float(gpu_self_play_a100_hours),
        main_training=main_train_h,
        deployment_calibration=deploy_h,
    )
    # Fill remaining budget into main_training only when schedule left it at 0
    # and caller asked to allocate residual — keep explicit schedule values.
    acct = aggregate_a100_hours(charges)

    cadence_cfg = dict(cfg.get("cadence_pilot") or {})
    pilot_evidence = dict(cadence_cfg.get("evidence") or {})
    candidates_out: list[dict[str, Any]] = []
    for ckpt_n in checkpoint_candidates:
        gpc = steady_state_rates(
            completed_games=int(selected["completed_games"]),
            worker_hours=float(selected["worker_hours"]),
            workers=workers,
            positions_per_game=float(selected["positions_per_game"]),
            self_play_wall_hours=self_play_wall_hours,
            checkpoint_count=int(ckpt_n),
        ).games_per_checkpoint
        ev = dict(pilot_evidence.get(str(ckpt_n)) or pilot_evidence.get(ckpt_n) or {})
        candidates_out.append(
            evaluate_cadence_candidate(
                name=f"ckpt-{ckpt_n}",
                games_per_checkpoint=gpc,
                checkpoint_count=int(ckpt_n),
                class_wdl=ev.get("class_wdl"),
                active_classes=ev.get("active_classes"),
                belief_calibration_regressed=ev.get("belief_calibration_regressed"),
                pairwise_verdict=ev.get("pairwise_verdict"),
            )
        )
    cadence_pick = select_smallest_useful_cadence(candidates_out)
    # If a useful cadence exists, refresh rates to that checkpoint count.
    if cadence_pick.get("useful_found") and cadence_pick.get("selected"):
        sel_n = int(cadence_pick["selected"]["checkpoint_count"])
        rates = steady_state_rates(
            completed_games=int(selected["completed_games"]),
            worker_hours=float(selected["worker_hours"]),
            workers=workers,
            positions_per_game=float(selected["positions_per_game"]),
            self_play_wall_hours=self_play_wall_hours,
            checkpoint_count=sel_n,
        )

    cpu_hours_total = sum(float(r.get("cpu_hours") or 0.0) for r in measured)

    # Fallback selection: if throughput is too low for any positive gpc, prefer
    # reduced search; else prefer non-promotable scope when cadence evidence is absent.
    if int(rates.games_per_checkpoint) <= 0:
        fallback_selected = "reduce_training_particles_sims_or_depth"
    elif not cadence_pick.get("useful_found"):
        fallback_selected = "narrower_non_promotable_research_scope"
    else:
        fallback_selected = "fewer_checkpoints_more_games_per_checkpoint"

    fallback_plan = {
        "order": [
            "reduce_training_particles_sims_or_depth",
            "fewer_checkpoints_more_games_per_checkpoint",
            "narrower_non_promotable_research_scope",
        ],
        "selected": fallback_selected,
        "note": (
            "No tested cadence passed curriculum confidence, held-out belief "
            "calibration, and arena pairwise improvement. Deployment-matched "
            "final phase remains required. Main training must not start."
            if not cadence_pick.get("useful_found")
            else "Useful cadence exists but another gate failed."
        ),
        "preserve_deployment_matched_final_phase": True,
    }

    body: dict[str, Any] = {
        "config": str(cfg_path),
        "selected_layout": selected,
        "layout_measurements": measured,
        "layout_ranking": selection["ranking"],
        "workers_per_a100": workers,
        "rates": rates.to_dict(),
        "learner_starved": starved,
        "positions_consume_per_hour": positions_consume,
        "a100_accounting": acct,
        "cpu_hours_total": cpu_hours_total,
        "cadence": {
            **cadence_pick,
            "candidates": candidates_out,
        },
        "deployment_calibration": {
            "a100_hours": deploy_h,
            "fits": deploy_fits and bool(acct.get("fits_budget")),
            "note": (
                "Deployment-matched self-play and calibration phase is scheduled "
                "inside the A100 budget; Part 09 has no accepted online survivor, "
                "so settings stay training-semantics-preserving with a final "
                "deployment-matched phase (Part 09b defers the hard latency gate)."
            ),
        },
        "fallback_plan": fallback_plan,
        "self_play_settings": {
            "max_turns": driver.max_turns,
            "n_particles": driver.n_particles,
            "target_simulations": driver.target_simulations,
            "min_simulations": driver.min_simulations,
            "search_depth": driver.search_depth,
            "note": (
                "Throughput uses this qualification search budget. Full turn-1200 "
                "games rates will be lower; remeasure before locking a promotable run."
            ),
        },
        "prerequisites": {
            "jax_preflight": "yes",
            "measurement_corpus": "yes",
            "online_qualification": "no_deferred_by_09b",
            "curriculum_rule": "resolved",
            "self_play_league": "implemented",
            "training_objective": "fixtures_yes_weights_unset",
        },
    }
    decision = decide_pass(body)
    report = stamp_generated({**body, "decision": decision})

    jpath = Path(json_path) if json_path else None
    mpath = Path(md_path) if md_path else None
    write_kwargs: dict[str, Any] = {}
    if jpath is not None:
        write_kwargs["json_path"] = jpath
    if mpath is not None:
        write_kwargs["md_path"] = mpath
    write_report(report, **write_kwargs)
    return report
