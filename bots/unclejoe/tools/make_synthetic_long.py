"""Build the `synthetic-long` corpus member (port-plan §6).

The natural corpus games end well before the observation state machine's
512-turn counter windows roll over, so no real game exercises a rollover.
`synthetic-long` is the longest natural game's frames played **twice**: the
state update never reads the bot's actions, so any frame stream is a valid
state-machine input, and a doubled 800-turn game reaches ~1,600 turns and
crosses both windows.

This tool used to be a hand-run snippet, which is why a corpus rebuild kept
losing the fixture (`docs/bots/joe-rs/parity.md` documents it; nothing
regenerated it). Run it between the two `capture_fixtures.py` phases:

    .venv/bin/python bots/unclejoe/tools/capture_fixtures.py --play
    .venv/bin/python bots/unclejoe/tools/make_synthetic_long.py
    .venv/bin/python bots/unclejoe/tools/capture_fixtures.py --capture

The capture phase globs `*.in.log`, so it picks this up with no extra flag.
There is deliberately no `.out.log`: the doubled stream is not a real game,
so no recorded reply exists to cross-check, and `capture_game` treats the
oracle's own outputs as the fixture in that case.

The source game is whichever natural game is longest, and it changes
between checkpoints — a stronger net ends games sooner. The name is
recorded in `synthetic-long.json` so a rebuild is auditable.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

CORPUS = Path(__file__).resolve().parent.parent.parent.parent / "data/joe/unclejoe-parity"
NAME = "synthetic-long"
MIN_TURNS = 1025  # must clear the second 512 window


def turn_count(path: Path) -> int:
    lines = path.read_text().splitlines()
    _, height, _ = (int(x) for x in lines[0].split())
    return (len(lines) - 1) // (1 + 3 * height)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=CORPUS / "games")
    parser.add_argument("--repeat", type=int, default=2)
    args = parser.parse_args()

    sources = [p for p in sorted(args.out.glob("*.in.log")) if p.name != f"{NAME}.in.log"]
    if not sources:
        parser.error(f"no natural game logs in {args.out} — run --play first")

    source = max(sources, key=turn_count)
    turns = turn_count(source)

    lines = source.read_text().splitlines()
    handshake, frames = lines[0], lines[1:]
    total = turns * args.repeat
    if total < MIN_TURNS:
        parser.error(
            f"longest game {source.name} is {turns} turns; x{args.repeat} = {total} "
            f"does not clear the second 512 window ({MIN_TURNS}) — raise --repeat")

    out_log = args.out / f"{NAME}.in.log"
    out_log.write_text("\n".join([handshake] + frames * args.repeat) + "\n")
    (args.out / f"{NAME}.json").write_text(json.dumps({
        "source": source.name.removesuffix(".in.log"),
        "source_turns": turns,
        "repeat": args.repeat,
        "turns": total,
    }, indent=2) + "\n")
    print(f"[synthetic-long] {source.name} ({turns} turns) x{args.repeat} "
          f"-> {out_log.name} ({total} turns)")


if __name__ == "__main__":
    main()
