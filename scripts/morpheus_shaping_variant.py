#!/usr/bin/env python3
"""Materialize a Morpheus prior-shaping variant as its own rated bot closure.

Part 17 C1/C2 compare shaping settings. A setting only becomes an honest
comparison if it is part of the content hash, so each arm is a real bot
directory with its own ``deployment.json`` — never an environment variable read
by a shared closure.

The 3.2 MB export artifact is symlinked, not copied: ``fingerprint`` reads
through the link, so every arm hashes the same model bytes while costing no
extra disk. ``tests/`` and ``__pycache__`` are skipped (both are outside the
closure by rule).

Examples::

    # C1 net-value ablation: same shaping, no network.
    python scripts/morpheus_shaping_variant.py uniform --evaluator uniform

    # C2 lambda sweep.
    python scripts/morpheus_shaping_variant.py lam050 --lambda 0.5
    python scripts/morpheus_shaping_variant.py lam000 --lambda 0.0

    python scripts/morpheus_shaping_variant.py uniform --clean
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "bots" / "morpheus"
ARTIFACT_DIRNAME = "artifact"
SKIP_DIRS = {"__pycache__", "tests", ".pytest_cache", ".mypy_cache"}
SKIP_SUFFIXES = {".pyc", ".pyo"}


def variant_dir(name: str) -> Path:
    return REPO / "bots" / f"morpheus-{name}"


def _is_copyable(path: Path, root: Path) -> bool:
    rel = path.relative_to(root)
    if any(part in SKIP_DIRS for part in rel.parts):
        return False
    if rel.parts and rel.parts[0] == ARTIFACT_DIRNAME:
        return False
    return path.suffix not in SKIP_SUFFIXES


def materialize(name: str, overrides: dict) -> Path:
    dest = variant_dir(name)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    for src in sorted(SOURCE.rglob("*")):
        if not src.is_file() or not _is_copyable(src, SOURCE):
            continue
        target = dest / src.relative_to(SOURCE)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        if src.name == "run.sh":
            target.chmod(0o755)

    # Symlink the artifact so every arm hashes identical model bytes.
    (dest / ARTIFACT_DIRNAME).symlink_to(
        SOURCE / ARTIFACT_DIRNAME, target_is_directory=True
    )

    config = json.loads((SOURCE / "deployment.json").read_text(encoding="utf-8"))
    config.update(overrides)
    (dest / "deployment.json").write_text(
        json.dumps(config, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return dest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("name", help="variant suffix; bot becomes bots/morpheus-<name>")
    ap.add_argument(
        "--evaluator",
        choices=("network", "uniform"),
        help="swap the learned prior/value for a flat one (C1 ablation arm)",
    )
    ap.add_argument(
        "--lambda",
        dest="lam",
        type=float,
        help="set both shaping lambdas at once (C2 sweep)",
    )
    ap.add_argument("--lambda-pre", type=float, help="pre-contact lambda only")
    ap.add_argument("--lambda-post", type=float, help="post-contact lambda only")
    ap.add_argument("--log-clip", type=float, help="override shaping_log_clip")
    ap.add_argument("--floor-frac", type=float, help="override shaping_floor_frac")
    ap.add_argument("--clean", action="store_true", help="remove the variant and exit")
    args = ap.parse_args(argv)

    dest = variant_dir(args.name)
    if args.clean:
        if dest.exists():
            shutil.rmtree(dest)
            print(f"removed {dest.relative_to(REPO)}")
        else:
            print(f"nothing to remove at {dest.relative_to(REPO)}")
        return 0

    overrides: dict = {}
    if args.evaluator is not None:
        overrides["evaluator"] = args.evaluator
    if args.lam is not None:
        overrides["shaping_lambda_pre_contact"] = float(args.lam)
        overrides["shaping_lambda_post_contact"] = float(args.lam)
    if args.lambda_pre is not None:
        overrides["shaping_lambda_pre_contact"] = float(args.lambda_pre)
    if args.lambda_post is not None:
        overrides["shaping_lambda_post_contact"] = float(args.lambda_post)
    if args.log_clip is not None:
        overrides["shaping_log_clip"] = float(args.log_clip)
    if args.floor_frac is not None:
        overrides["shaping_floor_frac"] = float(args.floor_frac)
    if not overrides:
        ap.error("no override given; the variant would duplicate bots/morpheus")

    dest = materialize(args.name, overrides)
    # Validate through the real loader so a bad arm fails here, not mid-round.
    sys.path[:0] = [str(dest), str(REPO / "bots")]
    from deployment import load_deployment  # noqa: E402

    cfg = load_deployment(dest / "deployment.json")
    print(f"created {dest.relative_to(REPO)}")
    for key, value in sorted(overrides.items()):
        print(f"  {key} = {getattr(cfg, key)!r} (requested {value!r})")
    print(f"  run.sh: {(dest / 'run.sh').relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
