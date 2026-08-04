#!/usr/bin/env python3
"""Part 10: Morpheus curriculum — build, classify, verify.

Usage:
    python scripts/morpheus_curriculum.py build \\
      --panel scripts/configs/morpheus/bootstrap-panel.json \\
      --trajectories data/trajectories/morpheus-bootstrap \\
      --output training/morpheus/manifests/curriculum.json

    python scripts/morpheus_curriculum.py classify \\
      --trajectories data/trajectories/morpheus-bootstrap \\
      --panel scripts/configs/morpheus/bootstrap-panel.json

    python scripts/morpheus_curriculum.py verify \\
      --manifest training/morpheus/manifests/curriculum.json
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
DEFAULT_TRAJECTORIES = REPO / "data" / "trajectories" / "morpheus-bootstrap"
DEFAULT_OUTPUT = REPO / "training" / "morpheus" / "manifests" / "curriculum.json"


def cmd_build(args: argparse.Namespace) -> int:
    from training.morpheus.curriculum.build import build_curriculum

    result = build_curriculum(
        panel_path=args.panel,
        trajectories_dir=args.trajectories,
        output=args.output,
        full_start_count=args.full_start_count,
        max_games=args.max_games,
        skip_verify=args.skip_verify,
        repo_root=REPO,
    )
    print(json.dumps(result, indent=2))
    return 0 if result.get("ok", True) else 1


def cmd_classify(args: argparse.Namespace) -> int:
    from arena.records.trajectories import read_trajectory
    from training.morpheus.corpus.coverage import load_source_index
    from training.morpheus.corpus.panel import load_panel
    from training.morpheus.curriculum.build import iter_panel_trajectories
    from training.morpheus.curriculum.classify import (
        classify_summary,
        classify_trajectory_prefixes,
        sample_prefixes_for_build,
    )
    from training.morpheus.curriculum.definitions import is_banned_source

    panel = load_panel(args.panel) if args.panel else None
    default_label = str((panel or {}).get("source_label") or "fixed_panel")
    index = load_source_index(args.trajectories)
    paths = iter_panel_trajectories(args.trajectories)
    if args.max_games is not None:
        paths = paths[: args.max_games]

    totals: dict[str, int] = {}
    games = 0
    skipped = 0
    for path in paths:
        traj = read_trajectory(path)
        label = index.get(traj.game_id) or default_label
        if is_banned_source(label):
            skipped += 1
            continue
        games += 1
        classified = classify_trajectory_prefixes(traj)
        selected = sample_prefixes_for_build(classified)
        summary = classify_summary(selected if args.sampled else classified)
        for key, count in summary["class_counts"].items():
            totals[key] = totals.get(key, 0) + count

    print(
        json.dumps(
            {
                "games": games,
                "skipped_banned": skipped,
                "sampled": bool(args.sampled),
                "class_counts": totals,
            },
            indent=2,
        )
    )
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    from training.morpheus.curriculum.verify import verify_manifest

    result = verify_manifest(
        args.manifest,
        repo_root=REPO,
        max_items=args.max_items,
        check_belief=args.belief,
        n_particles=args.n_particles,
    )
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    build_p = sub.add_parser("build", help="classify prefixes and write the manifest")
    build_p.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    build_p.add_argument("--trajectories", type=Path, default=DEFAULT_TRAJECTORIES)
    build_p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    build_p.add_argument("--full-start-count", type=int, default=8)
    build_p.add_argument("--max-games", type=int, default=None)
    build_p.add_argument(
        "--skip-verify",
        action="store_true",
        help="skip per-trajectory digest verify while building (not for release)",
    )
    build_p.set_defaults(func=cmd_build)

    classify_p = sub.add_parser(
        "classify",
        help="report class counts over trajectories without writing a manifest",
    )
    classify_p.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    classify_p.add_argument("--trajectories", type=Path, default=DEFAULT_TRAJECTORIES)
    classify_p.add_argument("--max-games", type=int, default=None)
    classify_p.add_argument(
        "--sampled",
        action="store_true",
        help="apply the build sampler caps before counting",
    )
    classify_p.set_defaults(func=cmd_classify)

    verify_p = sub.add_parser(
        "verify",
        help="replay digests and both-seat fog observations for every item",
    )
    verify_p.add_argument("--manifest", type=Path, required=True)
    verify_p.add_argument("--max-items", type=int, default=None)
    verify_p.add_argument(
        "--belief",
        action="store_true",
        help="also reconstruct memory and belief for each item",
    )
    verify_p.add_argument("--n-particles", type=int, default=4)
    verify_p.set_defaults(func=cmd_verify)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
