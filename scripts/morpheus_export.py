#!/usr/bin/env python3
"""Export a Morpheus float checkpoint to a static 8-bit TorchScript artifact + manifest."""
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
    parser.add_argument("--qengine", type=str, default=None, help="torch quantized engine")
    parser.add_argument("--run-id", type=str, default="export-local")
    parser.add_argument("--checkpoint-id", type=str, default="unknown")
    args = parser.parse_args()

    from morpheus.export import export_from_checkpoint

    result = export_from_checkpoint(
        args.checkpoint.resolve(),
        args.output.resolve(),
        qengine=args.qengine,
        training_run={
            "run_id": args.run_id,
            "checkpoint_id": args.checkpoint_id,
        },
    )
    print(f"exported to {result.output_dir}")
    print(f"qengine={result.qengine}")
    print("full float_to_export_mae:")
    for key, value in sorted(result.float_to_export_mae.items()):
        print(f"  {key}: {value:.6f}")
    print("online float_to_export_mae:")
    for entry, mae in sorted(result.online_float_to_export_mae.items()):
        print(f"  {entry}:")
        for key, value in sorted(mae.items()):
            print(f"    {key}: {value:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
