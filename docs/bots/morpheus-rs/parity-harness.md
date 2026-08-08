# Morpheus-rs parity harness

How a ported surface is proved equal to the Python oracle, and — the part that
turned out to matter more — how we know the proof is worth anything.

Established at milestone M1 of the [rewrite plan](rewrite-plan.md), §5 tier 1.
The corpus it runs on is described in [parity-corpus.md](parity-corpus.md).

## Shape

```bash
pytest bots/morpheus-rs/tests/          # smoke slice, ~1.5 s
bots/morpheus-rs/tools/run_parity.sh    # full corpus + mutation check, minutes
```

`tests/parity_cases.py` builds cases, runs the Python oracle on them, invokes
`morpheus-rs parity <kind>` on the same cases, and compares. Five surfaces are
checked, each bit-exact:

| kind | what it compares |
| --- | --- |
| `transition` | next state (all eight planes, time, winner) plus `GameInfo` |
| `order` | which seat resolves first |
| `observe` | the fogged observation a seat receives |
| `mask` | the 3970-long legal mask |
| `cost` | the live build-cost grid |

The corpus supplies **states, not answers**. Both implementations run fresh, so
a case can exercise action pairs the recorded game never played — a build, a
deathtouch, a move from a cell the seat does not own. `transition` has to agree
on invalid input too: it validates internally and silently no-ops, and
"silently" is exactly where two implementations drift apart unnoticed.

## Wire format, and a deviation from the plan

A flat stream of whitespace-separated integers, documented positionally in
`crates/core/src/parity.rs` and mirrored in `tests/parity_cases.py`. The plan
says "canonical JSON"; this is a narrow, deliberate departure. Rust's standard
library has no JSON, so the options were a dependency inside the *shipped*
binary or a hand-rolled parser, for a machine-to-machine channel whose entire
payload is integers. The 10,000-file unpacked limit is the binding sandbox
constraint and the crate is zero-dependency to protect it. Integers also
compare bit-exactly with no float formatting to argue about.

The parity subcommands live in the shipped binary rather than a test-only
build, because the thing under test has to be the thing that plays.

## Why a green run is not enough

A parity run proves the two implementations agree **on the cases it ran**. It
says nothing about whether those cases reach the behaviour under test. On this
port that gap was not hypothetical:

- Deleting the **50-tick army growth** passed 672 recorded transition cases.
  No recorded state sat at `time % 50 == 49`.
- Deleting the **NumPy negative-index wrap** in `_determine_move_order` passed
  too — and kept passing after 1,528 cases aimed at move ordering. The wrap
  reads row `h-1`, and the competition preset pads smaller boards to 21×21
  with mountains, so on a padded board that row is border and can never be
  owned. The recorded corpus *cannot* distinguish the two implementations.

Both were found by `tools/mutation_check.py`, which breaks one behaviour at a
time in the real source and checks the harness reports a mismatch. A mutation
that survives is not a bug in the port; it means nothing in the case set can
tell the two behaviours apart, which is a hole in the harness.

The fix was not to relax the check but to stop relying on replay alone.
`synthetic_states()` builds positions that sit on the growth and deathtouch
boundaries, own the wrap cell, hold a finished game, and put enough army on a
plain cell to make a build reachable. Everything the corpus cannot reach lives
there, each entry with the mutation that motivated it.

## Current results

Full corpus (1,328 heavy frames from 20 games), all surfaces bit-exact:

| kind | cases |
| --- | ---: |
| `transition` | 127,692 |
| `order` | 344,101 |
| `observe` | 21,282 |
| `mask` | 1,396 |
| `cost` | 1,396 |

Mutation check: **15 of 17 caught**
([report](../../research/measurements/morpheus-rs-mutation-check.json)). The
two survivors are equivalent mutants, not gaps, and the tool fails only on an
*unexplained* survivor:

- **`ownership_neutral` cleared on a loss** — `ownership_neutral` and
  `ownership[seat]` are disjoint on every well-formed state, so the mask that
  line clears is always empty. Only a malformed state could tell.
- **`army_to_move` clamp** — the invariant is guarded twice, once in the `raw`
  formula and again in the clamp, so either alone suffices. The Python is
  written the same way; the redundancy is ported, not introduced.

## Adding a surface

1. Add the kind to `run()` in `crates/core/src/parity.rs`, reading its inputs
   through the positional helpers and writing integers out.
2. Mirror the encode/decode in `tests/parity_cases.py` and add the comparison
   branch, reporting *where* it differs — a cell index, an action index — so a
   failure is actionable without a debugger.
3. Add the kind to `KINDS` in `tests/test_parity_tier1.py`.
4. **Add mutations for it** in `tools/mutation_check.py` and run them. A
   surface with no mutation coverage is a surface nobody has shown the harness
   can check.
