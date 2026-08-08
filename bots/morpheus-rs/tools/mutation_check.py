#!/usr/bin/env python3
"""Does the parity harness actually notice when the port is wrong?

    python bots/morpheus-rs/tools/mutation_check.py
    python bots/morpheus-rs/tools/mutation_check.py --full   # against the corpus

A green parity run proves the two implementations agree on the cases it ran.
It says nothing about whether those cases *reach* the behaviour under test —
and on this port that gap was real twice over. Dropping the 50-tick army
growth passed 672 recorded transition cases, because no recorded state sat at
`time % 50 == 49`. Dropping the NumPy negative-index wrap passed too: the wrap
reads row `h-1`, and the competition preset pads smaller boards to 21×21 with
mountains, so on a padded board that row is border and can never be owned.

So this deliberately breaks one behaviour at a time, in the real source, and
checks the harness reports a mismatch. A mutation that **survives** is not a
bug in the port — it means nothing in the case set can tell the two behaviours
apart, which is a hole in the harness and gets recorded as one.

Sources are restored from an in-memory copy in a `finally`, and the binary is
rebuilt at the end, so an interrupted run leaves the tree clean.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

BOT_DIR = Path(__file__).resolve().parents[1]
REPO = BOT_DIR.parents[1]
SRC = BOT_DIR / "crates" / "core" / "src"


@dataclass(frozen=True)
class Mutation:
    """One behaviour, deleted. `note` explains what it costs if it survives."""

    name: str
    file: str
    before: str
    after: str
    note: str = ""


MUTATIONS: tuple[Mutation, ...] = (
    Mutation(
        "50-tick army growth",
        "transition.rs",
        "if state.time % 50 == 0 {",
        "if false && state.time % 50 == 0 {",
    ),
    Mutation(
        "even-tick structure growth",
        "transition.rs",
        "if state.time % 2 == 0 {",
        "if false && state.time % 2 == 0 {",
    ),
    Mutation(
        "numpy negative-index wrap",
        "state.rs",
        "let r = if row < 0 { row + self.h as i32 } else { row };",
        "let r = row;",
    ),
    Mutation(
        "deathtouch turn boundary",
        "transition.rs",
        "state.time >= turn",
        "state.time > turn",
    ),
    Mutation(
        "attacker wins ties",
        "transition.rs",
        "let attacker_wins = army > target_army;",
        "let attacker_wins = army >= target_army;",
    ),
    Mutation(
        "half-split rounds up",
        "transition.rs",
        "source_army.div_euclid(2)",
        "(source_army + 1).div_euclid(2)",
    ),
    Mutation(
        "a move may empty its source",
        "transition.rs",
        "raw.min(source_army - 1).max(0)",
        "raw.min(source_army).max(0)",
        note=(
            "Equivalent mutant, not a harness hole. `army_to_move` guards its "
            "invariant twice over — once in the `raw` formula and again in the "
            "clamp — so either alone is sufficient and deleting either is "
            "unobservable. The Python is written the same way; the redundancy "
            "is being ported faithfully, not introduced here."
        ),
    ),
    Mutation(
        "build ignores affordability",
        "transition.rs",
        "let affords = state.armies[at] >= cost[at];",
        "let affords = true;",
    ),
    Mutation(
        "build proximity surcharge dropped",
        "transition.rs",
        "cost[(r * w + c) as usize] += surcharge;",
        "cost[(r * w + c) as usize] += 0;",
    ),
    Mutation(
        "loser cells not transferred",
        "transition.rs",
        "state.ownership[winner][i] |= loser_mask[i];",
        "state.ownership[winner][i] |= false;",
    ),
    Mutation(
        "neutral flag survives a loss",
        "transition.rs",
        "state.ownership_neutral[i] &= !loser_mask[i];",
        "state.ownership_neutral[i] &= true;",
        note=(
            "Unobservable by construction: `ownership_neutral` and "
            "`ownership[seat]` are disjoint on every well-formed state, so the "
            "mask it clears is always empty. Faithful to the Python, and only "
            "a malformed state could tell the difference."
        ),
    ),
    Mutation(
        "visibility is 4-neighbour",
        "observe.rs",
        "for dr in -1..=1 {",
        "for dr in 0..=0 {",
    ),
    Mutation(
        "structures under fog read as fog",
        "observe.rs",
        "TYPE_STRUCTURE_FOG\n        } else {",
        "TYPE_FOG\n        } else {",
    ),
    Mutation(
        "fogged armies leak",
        "observe.rs",
        "obs.army_grid[i] = if visible[i] { state.armies[i] } else { 0 };",
        "obs.army_grid[i] = state.armies[i];",
    ),
    Mutation(
        "half-split legal at two army",
        "action.rs",
        "let allow_half = src_army > 2;",
        "let allow_half = src_army >= 2;",
    ),
    Mutation(
        "build on unexplored ground",
        "action.rs",
        "if !memory.known_passable_base[at] {",
        "if false && !memory.known_passable_base[at] {",
    ),
    Mutation(
        "moves into mountains are legal",
        "action.rs",
        "if dest_t == TYPE_MOUNTAIN || dest_t == TYPE_STRUCTURE_FOG {",
        "if false {",
    ),
)


def _cargo_env() -> dict[str, str]:
    env = os.environ.copy()
    cargo_bin = Path.home() / ".cargo" / "bin"
    if cargo_bin.is_dir():
        env["PATH"] = f"{cargo_bin}{os.pathsep}{env.get('PATH', '')}"
    return env


def _build() -> bool:
    result = subprocess.run(
        ["cargo", "build", "--release", "--manifest-path", str(BOT_DIR / "Cargo.toml")],
        env=_cargo_env(),
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _parity(args: list[str]) -> tuple[bool, str]:
    """`(all kinds agreed, output)` for one parity run."""
    result = subprocess.run(
        [sys.executable, str(BOT_DIR / "tests" / "parity_cases.py"), *args],
        cwd=str(REPO),
        capture_output=True,
        text=True,
    )
    return result.returncode == 0, result.stdout + result.stderr


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="use the whole corpus")
    parser.add_argument("--output", type=Path, default=None, help="write a JSON report")
    args = parser.parse_args(argv)
    parity_args = [] if args.full else ["--smoke"]

    originals = {name: (SRC / name).read_text() for name in {m.file for m in MUTATIONS}}
    results = []
    try:
        if not _build():
            print("baseline build failed", file=sys.stderr)
            return 2
        clean, output = _parity(parity_args)
        if not clean:
            print("baseline parity is already failing; fix that first:\n" + output)
            return 2
        print(f"baseline: parity clean ({'corpus' if args.full else 'smoke'})\n")

        for mutation in MUTATIONS:
            path = SRC / mutation.file
            source = originals[mutation.file]
            if mutation.before not in source:
                results.append({"name": mutation.name, "outcome": "stale"})
                print(f"  STALE     {mutation.name} (pattern no longer in {mutation.file})")
                continue
            path.write_text(source.replace(mutation.before, mutation.after, 1))
            try:
                if not _build():
                    outcome = "uncompilable"
                else:
                    agreed, _ = _parity(parity_args)
                    outcome = "survived" if agreed else "caught"
            finally:
                path.write_text(source)
            results.append(
                {"name": mutation.name, "outcome": outcome, "note": mutation.note}
            )
            label = {"caught": "CAUGHT  ", "survived": "SURVIVED", "uncompilable": "NOBUILD "}[outcome]
            print(f"  {label}  {mutation.name}")
            if outcome == "survived" and mutation.note:
                print(f"              {mutation.note.splitlines()[0]}")
    finally:
        for name, source in originals.items():
            (SRC / name).write_text(source)
        _build()

    caught = sum(1 for r in results if r["outcome"] == "caught")
    survived = [r for r in results if r["outcome"] == "survived"]
    print(f"\n{caught}/{len(results)} caught, {len(survived)} survived")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {"scope": "corpus" if args.full else "smoke", "results": results},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.output}")

    # Survivors with a recorded explanation are accepted; an unexplained one is
    # a hole in the harness and fails the run.
    unexplained = [r for r in survived if not r.get("note")]
    for r in unexplained:
        print(f"unexplained survivor: {r['name']}", file=sys.stderr)
    return 1 if unexplained else 0


if __name__ == "__main__":
    raise SystemExit(main())
