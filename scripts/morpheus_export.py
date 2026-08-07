#!/usr/bin/env python3
"""Export a Morpheus float checkpoint to a TorchScript artifact + manifest.

Default format is float32 (bit-faithful, no calibration). ``--format int8``
keeps the static PTQ research path.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BOTS = REPO / "bots"
MORPHEUS = BOTS / "morpheus"
if str(BOTS) not in sys.path:
    sys.path.insert(0, str(BOTS))
if str(MORPHEUS) not in sys.path:
    sys.path.insert(0, str(MORPHEUS))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Checkpoint file or directory with state_dict.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for manifest.json and model.pt",
    )
    parser.add_argument(
        "--format",
        dest="fmt",
        choices=("float32", "int8"),
        default="float32",
        help="artifact format; float32 is the deployment default, int8 is research-only",
    )
    parser.add_argument("--qengine", type=str, default=None, help="torch quantized engine (int8 only)")
    parser.add_argument("--run-id", type=str, default="export-local")
    parser.add_argument("--checkpoint-id", type=str, default="unknown")
    parser.add_argument(
        "--allow-parity-fail",
        action="store_true",
        help="write the artifact even when online float/export MAE exceeds soft limits "
        "(research cadence only; MAE stays in the manifest)",
    )
    args = parser.parse_args()

    from morpheus.export import export_from_checkpoint

    result = export_from_checkpoint(
        args.checkpoint.resolve(),
        args.output.resolve(),
        fmt=args.fmt,
        qengine=args.qengine,
        training_run={
            "run_id": args.run_id,
            "checkpoint_id": args.checkpoint_id,
        },
        require_online_parity=not bool(args.allow_parity_fail),
    )
    print(f"exported to {result.output_dir}")
    print(f"qengine={result.qengine}")
    print("full float_to_export_mae:")
    for key, value in sorted(result.float_to_export_mae.items()):
        print(f"  {key}: {value:.6f}")
    print("online float_to_export_mae:")
    for entry, mae in sorted(result.online_float_to_export_mae.items()):
        print(f"  {entry}:")
        if isinstance(mae, dict):
            for key, value in sorted(mae.items()):
                print(f"    {key}: {value:.6f}")
        else:
            print(f"    {mae}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
