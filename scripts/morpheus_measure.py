#!/usr/bin/env python3
"""Morpheus measurement entry points.

Usage:
    python scripts/morpheus_measure.py army-normalization \\
      --trajectories data/trajectories/morpheus-bootstrap \\
      --output docs/research/measurements/morpheus-army-normalization.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DEFAULT_TRAJ = REPO / "data" / "trajectories" / "morpheus-bootstrap"
DEFAULT_OUT = (
    REPO / "docs" / "research" / "measurements" / "morpheus-army-normalization.json"
)


def cmd_army_normalization(args: argparse.Namespace) -> int:
    from training.morpheus.measure_army import measure_army_normalization

    report = measure_army_normalization(
        trajectories_dir=Path(args.trajectories),
        army_scale=args.army_scale,
        max_games=args.max_games,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "output": str(out), **{
        k: report[k] for k in ("game_count", "sample_count", "selected_scale", "recommendation")
        if k in report
    }}, indent=2))
    return 0


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

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
