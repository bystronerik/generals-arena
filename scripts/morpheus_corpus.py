#!/usr/bin/env python3
"""Part 00c: Morpheus measurement corpus — select, record, verify.

Usage:
    python scripts/morpheus_corpus.py select \\
      --panel scripts/configs/morpheus/bootstrap-panel.json

    python scripts/morpheus_corpus.py record \\
      --panel scripts/configs/morpheus/bootstrap-panel.json \\
      --round morpheus-bootstrap \\
      --round-seed 7 \\
      --seat-policy alternate \\
      --output data/trajectories/morpheus-bootstrap

    python scripts/morpheus_corpus.py verify \\
      --trajectories data/trajectories/morpheus-bootstrap \\
      --report docs/research/measurements/morpheus-bootstrap-corpus.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DEFAULT_PANEL = REPO / "scripts" / "configs" / "morpheus" / "bootstrap-panel.json"
DEFAULT_REPORT = (
    REPO / "docs" / "research" / "measurements" / "morpheus-bootstrap-corpus.json"
)


def cmd_select(args: argparse.Namespace) -> int:
    from training.morpheus.corpus.panel import (
        load_panel,
        validate_panel_against_checkout,
    )

    panel = load_panel(args.panel)
    problems = validate_panel_against_checkout(panel)
    print(json.dumps(
        {
            "panel": panel["name"],
            "members": len(panel["members"]),
            "selection_date": panel["selection_date"],
            "rating_era": panel["rating_era"],
            "ok": not problems,
            "problems": problems,
        },
        indent=2,
    ))
    return 0 if not problems else 1


def cmd_record(args: argparse.Namespace) -> int:
    from training.morpheus.corpus.record import record_corpus

    result = record_corpus(
        panel_path=args.panel,
        round_name=args.round,
        round_seed=args.round_seed,
        seat_policy=args.seat_policy,
        games_per_pair=args.games_per_pair,
        output=args.output,
        jobs=args.jobs,
        update_ratings=not args.no_ratings,
        strict_versions=args.strict_versions,
    )
    print(json.dumps(result, indent=2))
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    from training.morpheus.corpus.verify import verify_corpus

    result = verify_corpus(
        trajectories_dir=args.trajectories,
        report_path=args.report,
        panel_path=args.panel,
        scan_events=not args.skip_events,
    )
    decision = result["decision"]
    print(f"verdict={decision['verdict']}")
    print(f"note={decision['note']}")
    print(f"trajectories={result['coverage']['trajectory_count']}")
    print(f"decisive={result['coverage']['events']['decisive']}")
    print(
        "forced_mismatch_eligible="
        f"{result['coverage']['events']['forced_mismatch_eligible']}"
    )
    missing = decision.get("missing_classes") or {}
    if missing:
        print(f"missing_classes={json.dumps(missing)}")
    print(f"wrote {args.report}")
    print(f"wrote {args.report.with_suffix('.md')}")
    return 0 if decision["pass"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    select_p = sub.add_parser(
        "select",
        help="validate the named panel against this checkout",
    )
    select_p.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    select_p.set_defaults(func=cmd_select)

    record_p = sub.add_parser(
        "record",
        help="run the panel tournament and write engine trajectories",
    )
    record_p.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    record_p.add_argument("--round", required=True)
    record_p.add_argument(
        "--round-seed",
        type=int,
        default=None,
        help="override panel round_seed",
    )
    record_p.add_argument(
        "--seat-policy",
        choices=("alternate",),
        default="alternate",
    )
    record_p.add_argument(
        "--games-per-pair",
        type=int,
        default=None,
        help="override panel games_per_pair",
    )
    record_p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="must equal data/trajectories/<round> when set",
    )
    record_p.add_argument("--jobs", type=int, default=None)
    record_p.add_argument(
        "--no-ratings",
        action="store_true",
        help="store games without refitting ratings",
    )
    record_p.add_argument("--strict-versions", action="store_true")
    record_p.set_defaults(func=cmd_record)

    verify_p = sub.add_parser(
        "verify",
        help="replay-verify trajectories and write the coverage report",
    )
    verify_p.add_argument(
        "--trajectories",
        type=Path,
        required=True,
    )
    verify_p.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    verify_p.add_argument(
        "--panel",
        type=Path,
        default=DEFAULT_PANEL,
        help="panel used for source labels and decision context",
    )
    verify_p.add_argument(
        "--skip-events",
        action="store_true",
        help="skip contact/sight/castle replay scan (header coverage only)",
    )
    verify_p.set_defaults(func=cmd_verify)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
