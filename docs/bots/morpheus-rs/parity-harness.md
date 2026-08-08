# Morpheus-rs parity harness

How a ported surface is proved equal to the Python oracle, and — the part that
turned out to matter more — how we know the proof is worth anything.

Established at milestone M1 of the [rewrite plan](rewrite-plan.md) and
extended at M2 to tier 2. The corpus it runs on is described in
[parity-corpus.md](parity-corpus.md).

## Shape

```bash
pytest bots/morpheus-rs/tests/          # smoke slice, ~1.5 s
bots/morpheus-rs/tools/run_parity.sh    # full corpus + mutation check, minutes
```

`tests/parity_cases.py` builds cases, runs the Python oracle on them, invokes
`morpheus-rs parity <kind>` on the same cases, and compares. Nine surfaces are
checked. Tier 1 is bit-exact by specification; tier 2 allows 1e-6 — and is
*also* enforced bit-exact, for the reason in the next section.

| kind | tier | what it compares |
| --- | --- | --- |
| `transition` | 1 | next state (all eight planes, time, winner) plus `GameInfo` |
| `order` | 1 | which seat resolves first |
| `observe` | 1 | the fogged observation a seat receives |
| `mask` | 1 | the 3970-long legal mask |
| `cost` | 1 | the live build-cost grid |
| `memory` | 1 | `VisibleMemory` after folding one observation in |
| `hash` | 1 | all five SHA-256 digests the search keys on |
| `tensor` | 2 | the 49×21×21 observation tensor |
| `symmetry` | 1 | the D4 coordinate/direction maps and the policy permutation |

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

Floats ride the same integer stream as their raw `f32` bit patterns rather
than as decimal text. A float that round-trips through formatting is a float
whose last bits depend on two languages agreeing about printing — exactly the
argument tier 2 exists to avoid having. The channel is lossless, so a tensor
mismatch is always real arithmetic, never transport.

The parity subcommands live in the shipped binary rather than a test-only
build, because the thing under test has to be the thing that plays.

## Tier 2 is enforced tighter than it is specified

§5 budgets 1e-6 for the tensor's float planes. The port matches to the **last
bit**, and the harness enforces that rather than the budget, because the budget
turned out to be unable to see a real difference: a mutation computing
`army_value` in single instead of double precision stayed inside 1e-6 on every
case and survived. Enforcing bit-exactness catches it.

The looser figure is still the floor. If a platform ever produces genuine
drift, the failure message says "not bit-identical", reports the max |Δ|, and
separately reports how many cells exceed 1e-6 — so the decision to fall back to
the specified tolerance is made deliberately, with the number in hand, rather
than by a check that quietly never failed.

Matching to the last bit is not luck. The Python mixes `float64` and `float32`
on purpose — `army_value` computes its log in double and casts down, while the
coordinate planes are single throughout — and the port mirrors the width at
each site. Computing everything in `f64` and casting at the end would be *more*
accurate and would fail, because the network was trained on the rounding the
Python actually does.

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

Full corpus (1,328 heavy frames from 20 games), all surfaces bit-exact —
**500,072 cases**:

| kind | cases |
| --- | ---: |
| `transition` | 127,692 |
| `order` | 344,101 |
| `observe` | 21,282 |
| `mask` | 1,396 |
| `cost` | 1,396 |
| `memory` | 1,399 |
| `hash` | 1,399 |
| `tensor` | 1,399 |
| `symmetry` | 8 |

Mutation check: **28 of 32 caught**
([report](../../research/measurements/morpheus-rs-mutation-check.json)). All
four survivors are equivalent mutants, not gaps, and the tool fails only on an
*unexplained* survivor:

- **`ownership_neutral` cleared on a loss** — `ownership_neutral` and
  `ownership[seat]` are disjoint on every well-formed state, so the mask that
  line clears is always empty. Only a malformed state could tell.
- **`army_to_move` clamp** — the invariant is guarded twice, once in the `raw`
  formula and again in the clamp, so either alone suffices. The Python is
  written the same way; the redundancy is ported, not introduced.
- **sight-age clamp** — age is only computed for a cell already seen, so the
  quotient is in [0, 1] before the clamp runs. It can bind only on a memory the
  game cannot produce.
- **coordinate planes in double precision** — rows run 0..20 and the general
  sits on an integer cell, so `(r - gr)/20` is exactly representable either
  way. The width is documentation of what the Python does; a plane with
  non-integer inputs would diverge.

M2 added three surfaces' worth of mutations and two of them found real holes:
no case had a **remembered castle observed as plain fog** (an emitted board
cannot produce one — a castle in fog encodes as type 5), and no previous action
carried a **half-split**, leaving the tensor's move-kind plane constant. Both
are now crafted cases in `_crafted_memory_pairs` and the action rotation.

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
