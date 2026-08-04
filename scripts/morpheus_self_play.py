#!/usr/bin/env python3
"""Part 11: Morpheus self-play league producer.

Usage:
    python scripts/morpheus_self_play.py \\
      --config training/morpheus/configs/self-play-smoke.json \\
      --games 2 \\
      --output /tmp/morpheus-self-play

    python scripts/morpheus_self_play.py verify \\
      --shards /tmp/morpheus-self-play
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DEFAULT_CONFIG = REPO / "training" / "morpheus" / "configs" / "self-play-smoke.json"


def cmd_run(args: argparse.Namespace) -> int:
    from training.morpheus.self_play.driver import DriverConfig, run_batch

    config = DriverConfig.load(args.config)
    if args.games is not None:
        config = replace(config, games=int(args.games))
    if args.output is not None:
        config = replace(config, output=str(args.output))
    if args.seed is not None:
        config = replace(config, seed=int(args.seed))

    result = run_batch(config, output=Path(config.output), repo_root=REPO)
    print(
        json.dumps(
            {
                "ok": True,
                "output": str(result.output),
                "games": len(result.shards),
                "shards": [str(p) for p in result.shards],
                "proportions": result.proportions,
                "sources": [m.source for m in result.matchups],
            },
            indent=2,
        )
    )
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    from training.morpheus.self_play.verify import verify_directory

    report = verify_directory(Path(args.shards))
    print(json.dumps(report.to_dict(), indent=2))
    return 0 if report.ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="produce self-play shards (default)")
    run.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="self-play JSON config",
    )
    run.add_argument("--games", type=int, default=None, help="override games count")
    run.add_argument("--output", type=Path, default=None, help="shard output directory")
    run.add_argument("--seed", type=int, default=None, help="override batch seed")
    run.set_defaults(func=cmd_run)

    verify = sub.add_parser("verify", help="replay-verify shard digests and outcomes")
    verify.add_argument(
        "--shards",
        type=Path,
        required=True,
        help="directory of *.shard.jsonl.gz",
    )
    verify.set_defaults(func=cmd_verify)

    # Allow bare flags without the run subcommand for the documented invocation.
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--games", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    if not argv or argv[0] not in ("run", "verify"):
        # Documented form: scripts/morpheus_self_play.py --config ... --games 2
        args = parser.parse_args(["run", *argv])
    else:
        args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
