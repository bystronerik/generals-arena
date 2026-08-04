#!/usr/bin/env python3
"""Local Part 12 objective ablation entry point.

Usage:
    python scripts/morpheus_objective.py ablate \\
      --config training/morpheus/configs/objective-ablation.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    ablate = sub.add_parser("ablate", help="Run objective ablation fixtures")
    ablate.add_argument(
        "--config",
        default="training/morpheus/configs/objective-ablation.json",
    )
    ablate.add_argument("--json-out", default="")
    ablate.add_argument("--md-out", default="")
    ablate.add_argument(
        "--a100-hours",
        type=float,
        default=0.0,
        help="A100 hours to record (charged to Part 13)",
    )

    args = parser.parse_args(argv)
    if args.cmd == "ablate":
        from training.morpheus.objective.ablate import run_ablation

        cfg = Path(args.config)
        if not cfg.is_file():
            cfg = REPO / args.config
        report = run_ablation(
            cfg,
            json_path=Path(args.json_out) if args.json_out else None,
            md_path=Path(args.md_out) if args.md_out else None,
            a100_hours=float(args.a100_hours),
        )
        print(json.dumps(report["decision"], indent=2))
        return 0 if report["decision"]["pass"] else 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
