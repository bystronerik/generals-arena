#!/usr/bin/env python3
"""Benchmark float-to-export parity on fixed Morpheus tensors."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BOTS = REPO / "bots"
if str(BOTS) not in sys.path:
    sys.path.insert(0, str(BOTS))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Float checkpoint (default: tiny fixture)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="JSON report path (default: stdout only)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--work-dir", type=Path, help="temp export directory")
    args = parser.parse_args()

    import tempfile
    import torch

    from morpheus.export import (
        build_model_from_checkpoint,
        export_static_int8,
        load_checkpoint_state,
    )
    from training.morpheus.export_preflight.fixtures import all_batch_fixtures

    checkpoint_path = args.checkpoint or (
        REPO / "training/morpheus/tests/fixtures/tiny-checkpoint"
    )
    checkpoint = load_checkpoint_state(checkpoint_path.resolve())
    model = build_model_from_checkpoint(checkpoint)

    work_dir = args.work_dir
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="morpheus-bench-"))
    else:
        work_dir = work_dir.resolve()
        work_dir.mkdir(parents=True, exist_ok=True)

    export_result = export_static_int8(model, work_dir, seed=args.seed)
    session = torch.jit.load(str(export_result.artifact_path))
    session.eval()

    batch_reports: dict[str, dict] = {}
    for fixture in all_batch_fixtures(seed=args.seed):
        with torch.no_grad():
            float_out = model(fixture.tensor)
            export_policy, export_pass, export_wdl = session(fixture.tensor)
        float_policy = float_out.policy
        mae_policy = float((float_policy - export_policy).abs().mean().item())
        mae_pass = float((float_out.pass_logit - export_pass).abs().mean().item())
        mae_wdl = float((float_out.wdl_logits - export_wdl).abs().mean().item())
        batch_reports[fixture.name] = {
            "batch": fixture.batch,
            "policy_mae": mae_policy,
            "pass_logit_mae": mae_pass,
            "wdl_mae": mae_wdl,
        }

    report = {
        "seed": args.seed,
        "checkpoint": str(checkpoint_path.resolve()),
        "qengine": export_result.qengine,
        "example_mae": export_result.float_to_export_mae,
        "batches": batch_reports,
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text)
        print(f"wrote {args.output}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
