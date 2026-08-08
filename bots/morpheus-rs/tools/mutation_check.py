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
    # --- M2: memory, hashing, tensor, symmetry ---
    Mutation(
        "a fogged structure is always a mountain",
        "memory.rs",
        "next.known_castle[i] = was_passable;\n                next.known_mountain[i] = !was_passable;",
        "next.known_castle[i] = false;\n                next.known_mountain[i] = true;",
    ),
    Mutation(
        "plain fog erases a known castle",
        "memory.rs",
        "next.known_passable_base[i] = true;\n                next.known_mountain[i] = false;\n            }",
        "next.known_passable_base[i] = true;\n                next.known_mountain[i] = false;\n                next.known_castle[i] = false;\n            }",
    ),
    Mutation(
        "castle owner refreshes on any sighting",
        "memory.rs",
        "if t == TYPE_CASTLE {\n                next.remembered_castle_owner[i] = owner;\n            }",
        "next.remembered_castle_owner[i] = owner;",
    ),
    Mutation(
        "dynamic memory refreshes through fog",
        "memory.rs",
        "if visible {\n            next.ever_visible[i] = true;",
        "if true {\n            next.ever_visible[i] = true;",
    ),
    Mutation(
        "owner planes hashed as int32",
        "hashing.rs",
        "hasher.update(&[value as i8 as u8]);\n    }\n    for &value in &memory.remembered_army[..n] {",
        "hasher.update(&value.to_le_bytes());\n    }\n    for &value in &memory.remembered_army[..n] {",
    ),
    Mutation(
        "board dimensions left out of the digest",
        "hashing.rs",
        "hasher.update(&(memory.h as i32).to_le_bytes());",
        "hasher.update(&[]);",
    ),
    Mutation(
        "army compression in single precision",
        "tensor.rs",
        "let out = x.max(0.0).ln_1p() / denom;",
        "let out = ((x.max(0.0) as f32).ln_1p() / denom as f32) as f64;",
    ),
    Mutation(
        "coordinate planes in double precision",
        "tensor.rs",
        "set(P_ROW_FROM_GENERAL, r, c, (r as f32 - gr as f32) / 20.0);",
        "set(P_ROW_FROM_GENERAL, r, c, ((r as f64 - gr as f64) / 20.0) as f32);",
        note=(
            "Equivalent on this board: rows run 0..20 and the general sits on "
            "an integer cell, so (r - gr)/20 is exactly representable in f32 "
            "and the two widths agree bit for bit. The width still matters as "
            "documentation of what the Python does — a plane with non-integer "
            "inputs would diverge."
        ),
    ),
    Mutation(
        "sight age left unclamped",
        "tensor.rs",
        "set(P_SIGHT_AGE, r, c, age.clamp(0.0, 1.0) as f32);",
        "set(P_SIGHT_AGE, r, c, age as f32);",
        note=(
            "Unreachable by construction. Age is only computed for a cell that "
            "has been seen, so `last_seen_turn` is a real turn in [0, turn]; "
            "the numerator is therefore in [0, 1200] and the quotient already "
            "in [0, 1]. The clamp can only bind on a memory the game cannot "
            "produce. Faithful to the Python, which clamps for the same "
            "belt-and-braces reason."
        ),
    ),
    Mutation(
        "neutral plane ignores mountains",
        "tensor.rs",
        "let passable_vis = visible && !memory.known_mountain[i];",
        "let passable_vis = visible;",
    ),
    Mutation(
        "half-splits paint as full moves",
        "tensor.rs",
        "if split == 1 { 0.5 } else { 1.0 };",
        "1.0;",
    ),
    Mutation(
        "bulk growth countdown off by one",
        "tensor.rs",
        "(((50 - ((turn + 1) % 50)) % 50) as f64) / 49.0",
        "(((50 - (turn % 50)) % 50) as f64) / 49.0",
    ),
    Mutation(
        "constants painted over the padding",
        "tensor.rs",
        "for r in 0..h {\n            for c in 0..w {\n                tensor[plane * PLANE_CELLS + r * PAD + c] = v;",
        "for r in 0..PAD {\n            for c in 0..PAD {\n                tensor[plane * PLANE_CELLS + r * PAD + c] = v;",
    ),
    Mutation(
        "rot90+flip composed in the other order",
        "symmetry.rs",
        "Symmetry::Rot90FlipH => {\n                let (r2, c2) = rot90_rc(r, c);\n                flip_h_rc(r2, c2)\n            }",
        "Symmetry::Rot90FlipH => {\n                let (r2, c2) = flip_h_rc(r, c);\n                rot90_rc(r2, c2)\n            }",
    ),
    Mutation(
        "directions do not follow rotations",
        "symmetry.rs",
        "Symmetry::Rot90 => rot90[d],",
        "Symmetry::Rot90 => d,",
    ),
    # --- M3: the network graph and the arithmetic around it ------------------
    #
    # The `net` surface compares against TorchScript with a *tolerance*, not
    # bit-exactly, which makes these mutations more load-bearing than the
    # tier-1 ones: a budget wide enough to absorb a real mistake is a budget
    # that proves nothing. Several below are deliberately small — an eps, a
    # padding column, a precision — to find out where the budget stops seeing.
    Mutation(
        "group norm epsilon an order of magnitude out",
        "network.rs",
        "const GROUP_NORM_EPS: f64 = 1e-5;",
        "const GROUP_NORM_EPS: f64 = 1e-4;",
    ),
    Mutation(
        "group norm sums over the pad columns",
        "network.rs",
        "let plane = &buf[c * STRIDE..c * STRIDE + CELLS];",
        "let plane = &buf[c * STRIDE..(c + 1) * STRIDE];",
        note=(
            "Equivalent, and equivalent for a reason worth having: the seven "
            "pad columns per plane are zero, so summing them adds nothing to "
            "either accumulator. This mutation surviving is evidence *for* the "
            "zero-padding invariant the module doc claims. The observable half "
            "of 'include the pad columns' is the divisor, which is the next "
            "mutation and is caught."
        ),
    ),
    Mutation(
        "group norm divides by the padded width",
        "network.rs",
        "let n = (per_group * CELLS) as f64;",
        "let n = (per_group * STRIDE) as f64;",
    ),
    Mutation(
        "group norm over the whole layer, not eight groups",
        "network.rs",
        "pub const GROUP_NORM_GROUPS: usize = 8;",
        "pub const GROUP_NORM_GROUPS: usize = 4;",
    ),
    Mutation(
        "dilation cycle flattened to 1",
        "network.rs",
        "pub const DILATION_CYCLE: [usize; 3] = [1, 2, 4];",
        "pub const DILATION_CYCLE: [usize; 3] = [1, 1, 1];",
    ),
    Mutation(
        "ReLU6 with no upper clamp",
        "network.rs",
        "*v = v.clamp(0.0, 6.0);",
        "*v = v.max(0.0);",
    ),
    Mutation(
        "residual connection dropped",
        "network.rs",
        "s.trunk[i] += s.residual[i];",
        "s.trunk[i] += 0.0;",
    ),
    Mutation(
        "depthwise kernel transposed",
        "network.rs",
        "let wv = w[ky * 3 + kx];",
        "let wv = w[kx * 3 + ky];",
    ),
    Mutation(
        "global mean ignores the board mask",
        "network.rs",
        "sum += (v * mask[p]) as f64;",
        "sum += v as f64;",
    ),
    Mutation(
        "global max ignores the board mask",
        "network.rs",
        "let candidate = if mask[p] < 0.5 { MASK_FILL } else { v };",
        "let candidate = v;",
    ),
    Mutation(
        "global mean channels reversed",
        "network.rs",
        "self.scratch.feat[c] = (sum / safe) as f32;",
        "self.scratch.feat[TRUNK - 1 - c] = (sum / safe) as f32;",
    ),
    Mutation(
        "stem weights laid out with the tap axis outermost",
        "network.rs",
        "stem[oc * IN_CHANNELS * 9 + ic * 9 + k] =",
        "stem[oc * IN_CHANNELS * 9 + k * IN_CHANNELS + ic] =",
    ),
    Mutation(
        "1x1 head bias dropped",
        "network.rs",
        "Some(&head.bias),",
        "None,",
    ),
    Mutation(
        "softmax in double precision",
        "network.rs",
        "let e = (v - max).exp();",
        "let e = ((v - max) as f64).exp() as f32;",
        note=(
            "Equivalent at this width. The result is rounded back to f32 "
            "immediately, and `exp` is correctly rounded closely enough that "
            "computing it in double and narrowing lands on the same float for "
            "every logit the network produces. What the f32 contract actually "
            "pins is the *sum and the division*, which this does not touch; "
            "widening those needs `exps` widened too, which is a rewrite "
            "rather than a one-line break, and the `prior` surface's 5e-6 cap "
            "is what holds them."
        ),
    ),
    Mutation(
        "illegal actions keep their softmax mass",
        "network.rs",
        ".map(|i| if mask[i] { (exps[i] / total) as f64 } else { 0.0 })",
        ".map(|i| (exps[i] / total) as f64)",
        note=(
            "Equivalent because the fill underflows. An illegal logit is set "
            "to `f32::MIN`, so `exp(MIN - max)` is exactly 0.0 and the "
            "multiply by the mask removes nothing. The Python multiplies too, "
            "for the same non-reason. Keeping the explicit zero is still "
            "right: it makes 'illegal means zero' a property of the code "
            "rather than of float underflow, and the harness checks it "
            "without tolerance."
        ),
    ),
    Mutation(
        "backup value not negated off the root",
        "network.rs",
        "    if from_root {\n        v\n    } else {\n        -v\n    }",
        "    let _ = from_root;\n    v",
    ),
    Mutation(
        "WDL value is win minus draw",
        "network.rs",
        "((exps[0] / total) - (exps[2] / total)) as f64",
        "((exps[0] / total) - (exps[1] / total)) as f64",
    ),
    Mutation(
        "GEMM bias applied to the wrong row",
        "gemm.rs",
        "let add = bias.map_or(0.0, |v| v[m0 + i]);",
        "let add = bias.map_or(0.0, |v| v[m0]);",
    ),
    Mutation(
        "linear head drops its bias",
        "gemm.rs",
        "y[m] = sum + bias[m];",
        "y[m] = sum;",
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
    parser.add_argument(
        "--only",
        default=None,
        help="substring filter on mutation names; for iterating on a new one",
    )
    parser.add_argument("--output", type=Path, default=None, help="write a JSON report")
    args = parser.parse_args(argv)
    parity_args = [] if args.full else ["--smoke"]

    selected = [m for m in MUTATIONS if not args.only or args.only in m.name]
    if not selected:
        print(f"no mutation matches {args.only!r}", file=sys.stderr)
        return 2
    originals = {name: (SRC / name).read_text() for name in {m.file for m in selected}}
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

        for mutation in selected:
            path = SRC / mutation.file
            source = originals[mutation.file]
            if mutation.before not in source:
                results.append({"name": mutation.name, "outcome": "stale"})
                print(f"  STALE     {mutation.name} (pattern no longer in {mutation.file})")
                continue
            # A mutation that lands in `#[cfg(test)]` breaks a test instead of
            # the bot, and then "survives" for a reason that says nothing about
            # the harness. That happened once — the pad-column mutation matched
            # a line in `group_norm`'s own unit test, sailed through, and was
            # about to be written up as a coverage hole. `replace(..., 1)` hits
            # the first occurrence, so the check is simply whether that
            # occurrence is below the test module.
            tests_at = source.find("#[cfg(test)]")
            if tests_at >= 0 and source.index(mutation.before) > tests_at:
                results.append({"name": mutation.name, "outcome": "in-test"})
                print(f"  IN-TEST   {mutation.name} (pattern only matches inside #[cfg(test)])")
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

    if args.output and not args.only:
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
    elif args.output:
        print("--only: report not written (it would record a partial run)")

    # Survivors with a recorded explanation are accepted; an unexplained one is
    # a hole in the harness and fails the run.
    unexplained = [r for r in survived if not r.get("note")]
    for r in unexplained:
        print(f"unexplained survivor: {r['name']}", file=sys.stderr)
    # A mutation that never reached production code proves nothing either way.
    misaimed = [r for r in results if r["outcome"] in ("stale", "in-test")]
    for r in misaimed:
        print(f"misaimed mutation: {r['name']} ({r['outcome']})", file=sys.stderr)
    return 1 if unexplained or misaimed else 0


if __name__ == "__main__":
    raise SystemExit(main())
