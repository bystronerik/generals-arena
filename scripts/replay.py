#!/usr/bin/env python3
"""Read a scraped leaderboard replay and say how the game actually went.

Answers the questions a win/loss column cannot: where the economies diverged,
what happened and when, whether our side ever learned where the enemy general
was, and whether it ever gathered an army and went at it. Text by default,
`--json` for a machine.

Strictly read-only over competition-replays/. These are observational games
played under rules we did not run, so nothing here may reach data/games/,
data/ratings/, or data/remote_games/. See docs/engine/leaderboard-replays.md.

    python scripts/replay.py full erik.bystron 15932
    python scripts/replay.py batch erik.bystron --outcome lose
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arena.instrument.replay.analysis import ForfeitReplay, analyze  # noqa: E402
from arena.instrument.replay.batch import (  # noqa: E402
    aggregate,
    iter_games,
    render_aggregate,
)
from arena.instrument.replay.loader import (  # noqa: E402
    OUTCOMES,
    REPLAYS_DIR,
    ReplayNotFound,
    open_replay,
)
from arena.instrument.replay.report import DEFAULT_EVERY, build_report, render  # noqa: E402

SECTIONS = {
    "summarize": ("timeline", "verdict"),
    "events": ("events",),
    "flow": ("flow", "verdict"),
    "full": ("timeline", "events", "flow", "verdict"),
}


def emit(payload, as_json: bool) -> None:
    print(json.dumps(payload, indent=2) if as_json else payload)


def run_single(args: argparse.Namespace) -> int:
    try:
        replay = open_replay(args.player, args.match_id, args.root)
    except ReplayNotFound as exc:
        print(exc, file=sys.stderr)
        return 2
    try:
        analysis = analyze(replay)
    except ForfeitReplay as exc:
        print(exc, file=sys.stderr)
        return 3

    sections = SECTIONS[args.command]
    report = build_report(analysis, every=getattr(args, "every", DEFAULT_EVERY))
    emit(report.as_json(sections) if args.json else render(report, sections), args.json)
    return 0


def run_batch(args: argparse.Namespace) -> int:
    lines = []
    forfeits = []
    stream = not args.json
    for line, folder, path in iter_games(args.player, args.outcome, args.root):
        if line is None:
            forfeits.append(path.stem)
            if stream:
                print(f"{path.stem:>7} {folder:<5}   forfeit (<= 1 tick), skipped", flush=True)
            continue
        lines.append(line)
        if stream:
            print(line.render(), flush=True)

    if not lines:
        print(f"no played replays for {args.player} under {args.root}", file=sys.stderr)
        return 2

    summary = aggregate(lines, len(forfeits))
    if args.json:
        emit(
            {
                "player": args.player,
                "outcome_filter": args.outcome,
                "summary": summary,
                "forfeits": forfeits,
                "games": [line.as_json() for line in lines],
            },
            True,
        )
    else:
        print()
        print(render_aggregate(summary, args.player))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=REPLAYS_DIR, help=argparse.SUPPRESS)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def game_parser(name: str, help_text: str) -> argparse.ArgumentParser:
        sub = subparsers.add_parser(name, help=help_text, description=help_text)
        sub.add_argument("player", help="leaderboard player name (the replay folder)")
        sub.add_argument("match_id", help="replay id, e.g. 15932")
        sub.add_argument("--json", action="store_true", help="emit JSON instead of text")
        sub.set_defaults(handler=run_single)
        return sub

    summarize = game_parser("summarize", "sampled economy timeline plus a prose verdict")
    summarize.add_argument(
        "--every",
        type=int,
        default=DEFAULT_EVERY,
        help=f"sample the timeline every N ticks (default: {DEFAULT_EVERY})",
    )
    game_parser("events", "chronological event log with phase segmentation")
    game_parser("flow", "information vs action: what was knowable, what was done about it")
    full = game_parser("full", "timeline, events, flow and verdict in one run")
    full.add_argument(
        "--every",
        type=int,
        default=DEFAULT_EVERY,
        help=f"sample the timeline every N ticks (default: {DEFAULT_EVERY})",
    )

    batch = subparsers.add_parser(
        "batch",
        help="one line per replay for a whole folder, plus flaw aggregates",
        description="one line per replay for a whole folder, plus flaw aggregates",
    )
    batch.add_argument("player", help="leaderboard player name (the replay folder)")
    batch.add_argument(
        "--outcome",
        choices=(*OUTCOMES, "all"),
        default="all",
        help=(
            "restrict to the queried player's own result, derived from the replay "
            "rather than from the folder the scraper used (default: all)"
        ),
    )
    batch.add_argument("--json", action="store_true", help="emit JSON instead of text")
    batch.set_defaults(handler=run_batch)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
