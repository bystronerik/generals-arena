#!/usr/bin/env python3
"""Part 00b: sandbox static 8-bit export preflight.

Usage (local host):
    python scripts/morpheus_export_preflight.py \\
      --requirements requirements-sandbox.txt \\
      --output docs/research/measurements/morpheus-export-preflight.json

Usage (Linux sandbox-shaped seat on Modal CPU):
    modal run scripts/morpheus_modal_export_preflight.py

Probes only packages pinned for the competition sandbox. Does not install or
import ONNX Runtime or other unavailable dependencies.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--requirements",
        type=Path,
        default=REPO / "requirements-sandbox.txt",
        help="Sandbox requirements file (default: requirements-sandbox.txt)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "docs/research/measurements/morpheus-export-preflight.json",
        help="JSON report path (Markdown is written beside it)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--n-blocks",
        type=int,
        default=12,
        help="Inverted residual blocks in the probe (default: 12)",
    )
    args = parser.parse_args()

    from training.morpheus.export_preflight.measure import run_export_preflight
    from training.morpheus.export_preflight.report import build_report, write_report

    result = run_export_preflight(
        requirements_path=args.requirements.resolve(),
        seed=args.seed,
        n_blocks=args.n_blocks,
    )
    report = build_report(result)
    json_path, md_path = write_report(report, json_path=args.output.resolve())
    verdict = report["decision"]["verdict"]
    viable = report["decision"]["viable_candidates"]
    print(f"verdict={verdict}")
    print(f"viable={viable}")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
