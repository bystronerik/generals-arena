#!/usr/bin/env python3
"""Morpheus measurement entry points.

Usage:
    python scripts/morpheus_measure.py army-normalization \\
      --trajectories data/trajectories/morpheus-bootstrap \\
      --output docs/research/measurements/morpheus-army-normalization.json

    python scripts/morpheus_measure.py belief-recovery \\
      --trajectories data/trajectories/morpheus-bootstrap \\
      --force-proposal-mismatch \\
      --output docs/research/measurements/morpheus-belief-recovery.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
_BOT = REPO / "bots" / "morpheus"
for entry in (REPO / "bots", _BOT):
    s = str(entry)
    if s not in sys.path:
        sys.path.insert(0, s)

DEFAULT_TRAJ = REPO / "data" / "trajectories" / "morpheus-bootstrap"
DEFAULT_OUT = (
    REPO / "docs" / "research" / "measurements" / "morpheus-army-normalization.json"
)


def _write_report(out: Path, report: dict) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def cmd_army_normalization(args: argparse.Namespace) -> int:
    from training.morpheus.measure_army import measure_army_normalization

    report = measure_army_normalization(
        trajectories_dir=Path(args.trajectories),
        army_scale=args.army_scale,
        max_games=args.max_games,
    )
    out = Path(args.output)
    _write_report(out, report)
    print(
        json.dumps(
            {
                "ok": True,
                "output": str(out),
                **{
                    k: report[k]
                    for k in (
                        "game_count",
                        "sample_count",
                        "selected_scale",
                        "recommendation",
                    )
                    if k in report
                },
            },
            indent=2,
        )
    )
    return 0


def cmd_belief_recovery(args: argparse.Namespace) -> int:
    from training.morpheus.measure_belief import measure_belief_recovery

    report = measure_belief_recovery(
        trajectories_dir=Path(args.trajectories),
        max_games=args.max_games,
        n_particles=args.n_particles,
        force_proposal_mismatch=bool(args.force_proposal_mismatch),
        max_turns=args.max_turns,
    )
    out = Path(args.output)
    _write_report(out, report)
    print(
        json.dumps(
            {
                "ok": True,
                "output": str(out),
                "recovery_rate": report["recovery_rate"],
                "p99_cost_ms": report["p99_cost_ms"],
            },
            indent=2,
        )
    )
    return 0


def cmd_opponent_belief(args: argparse.Namespace) -> int:
    from training.morpheus.measure_belief import measure_opponent_belief

    report = measure_opponent_belief(
        trajectories_dir=Path(args.trajectories),
        max_games=args.max_games,
        n_particles=args.n_particles,
        max_turns=args.max_turns,
    )
    out = Path(args.output)
    _write_report(out, report)
    print(
        json.dumps(
            {
                "ok": True,
                "output": str(out),
                "mean_enemy_action_log_loss": report["mean_enemy_action_log_loss"],
                "mean_particle_survival": report["mean_particle_survival"],
            },
            indent=2,
        )
    )
    return 0


def cmd_ess_threshold(args: argparse.Namespace) -> int:
    from training.morpheus.measure_belief import measure_ess_threshold

    report = measure_ess_threshold(
        trajectories_dir=Path(args.trajectories),
        max_games=args.max_games,
        n_particles=args.n_particles,
        max_turns=args.max_turns,
    )
    out = Path(args.output)
    _write_report(out, report)
    print(
        json.dumps(
            {
                "ok": True,
                "output": str(out),
                "selected_ess_threshold_fraction": report[
                    "selected_ess_threshold_fraction"
                ],
            },
            indent=2,
        )
    )
    return 0


def cmd_search_resources(args: argparse.Namespace) -> int:
    from training.morpheus.measure_search import measure_search_resources

    report = measure_search_resources(
        suite=Path(args.suite) if args.suite else None,
        live_sims=args.live_sims,
    )
    out = Path(args.output)
    _write_report(out, report)
    print(
        json.dumps(
            {
                "ok": True,
                "output": str(out),
                "suite_pass": report["suite_pass"],
                "table_hit_rate": report["table_hit_rate"],
                "eviction_loss_rate": report["eviction_loss_rate"],
                "completed_simulations": report["completed_simulations"],
                "approx_tree_bytes": report["approx_tree_bytes"],
            },
            indent=2,
        )
    )
    return 0


def cmd_online_runtime(args: argparse.Namespace) -> int:
    from training.morpheus.measure_online import measure_online_runtime

    report = measure_online_runtime(
        Path(args.config),
        single_core=bool(args.single_core),
        core=int(args.core),
        write_runtime=not bool(args.dry_run),
        runtime_out=Path(args.runtime_out) if args.runtime_out else None,
        bot_deployment_out=Path(args.bot_deployment_out)
        if args.bot_deployment_out
        else None,
    )
    out = Path(args.output)
    _write_report(out, report)
    print(
        json.dumps(
            {
                "ok": True,
                "output": str(out),
                "verdict": report["verdict"],
                "survivor_count": report["survivor_count"],
                "selected": report.get("selected"),
                "reasons_no": report.get("reasons_no"),
            },
            indent=2,
        )
    )
    return 0 if report["verdict"] == "yes" else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    army = sub.add_parser(
        "army-normalization",
        help="Army quantiles and quantization error for the log1p scale",
    )
    army.add_argument("--trajectories", type=Path, default=DEFAULT_TRAJ)
    army.add_argument("--output", type=Path, default=DEFAULT_OUT)
    army.add_argument("--army-scale", type=float, default=4096.0)
    army.add_argument("--max-games", type=int, default=0, help="0 = all games")
    army.set_defaults(func=cmd_army_normalization)

    meas_dir = REPO / "docs" / "research" / "measurements"

    recovery = sub.add_parser(
        "belief-recovery",
        help="Exact recovery rate and p99 cost after forced proposal mismatch",
    )
    recovery.add_argument("--trajectories", type=Path, default=DEFAULT_TRAJ)
    recovery.add_argument(
        "--output",
        type=Path,
        default=meas_dir / "morpheus-belief-recovery.json",
    )
    recovery.add_argument(
        "--force-proposal-mismatch",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    recovery.add_argument("--max-games", type=int, default=4)
    recovery.add_argument("--n-particles", type=int, default=16)
    recovery.add_argument("--max-turns", type=int, default=40)
    recovery.set_defaults(func=cmd_belief_recovery)

    opponent = sub.add_parser(
        "opponent-belief",
        help="Enemy-action log loss and real-observation particle survival",
    )
    opponent.add_argument("--trajectories", type=Path, default=DEFAULT_TRAJ)
    opponent.add_argument(
        "--output",
        type=Path,
        default=meas_dir / "morpheus-opponent-belief.json",
    )
    opponent.add_argument("--max-games", type=int, default=4)
    opponent.add_argument("--n-particles", type=int, default=16)
    opponent.add_argument("--max-turns", type=int, default=40)
    opponent.set_defaults(func=cmd_opponent_belief)

    ess = sub.add_parser(
        "ess-threshold",
        help="ESS resample threshold sweep with uniqueness and survival",
    )
    ess.add_argument("--trajectories", type=Path, default=DEFAULT_TRAJ)
    ess.add_argument(
        "--output",
        type=Path,
        default=meas_dir / "morpheus-ess-threshold.json",
    )
    ess.add_argument("--max-games", type=int, default=2)
    ess.add_argument("--n-particles", type=int, default=16)
    ess.add_argument("--max-turns", type=int, default=30)
    ess.set_defaults(func=cmd_ess_threshold)

    search = sub.add_parser(
        "search-resources",
        help="Tactical suite plus table hit rate, eviction loss, and memory",
    )
    search.add_argument(
        "--suite",
        type=Path,
        default=REPO / "bots" / "morpheus" / "tests" / "fixtures" / "tactical-suite.json",
    )
    search.add_argument(
        "--output",
        type=Path,
        default=meas_dir / "morpheus-search-resources.json",
    )
    search.add_argument("--live-sims", type=int, default=16)
    search.set_defaults(func=cmd_search_resources)

    online = sub.add_parser(
        "online-runtime",
        help="Coupled one-core complete-turn p50/p99 deployment selection",
    )
    online.add_argument(
        "--config",
        type=Path,
        default=REPO / "scripts" / "configs" / "morpheus" / "online-sweep.json",
    )
    online.add_argument(
        "--output",
        type=Path,
        default=meas_dir / "morpheus-online-runtime.json",
    )
    online.add_argument(
        "--single-core",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    online.add_argument("--core", type=int, default=0)
    online.add_argument(
        "--runtime-out",
        type=Path,
        default=REPO / "scripts" / "configs" / "morpheus" / "online-runtime.json",
    )
    online.add_argument(
        "--bot-deployment-out",
        type=Path,
        default=REPO / "bots" / "morpheus" / "deployment.json",
    )
    online.add_argument(
        "--dry-run",
        action="store_true",
        help="Measure and report without writing online-runtime.json",
    )
    online.set_defaults(func=cmd_online_runtime)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
