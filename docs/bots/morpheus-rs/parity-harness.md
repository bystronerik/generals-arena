# Morpheus-rs parity harness

How a ported surface is proved equal to the Python oracle, and — the part that
turned out to matter more — how we know the proof is worth anything.

Established at milestone M1 of the [rewrite plan](rewrite-plan.md), extended at
M2 to tier 2, and at M3 to the first surface that cannot be bit-exact. The
corpus it runs on is described in [parity-corpus.md](parity-corpus.md).

## Shape

```bash
pytest bots/morpheus-rs/tests/          # smoke slice, ~7 s
bots/morpheus-rs/tools/run_parity.sh    # full corpus + mutation check, minutes
```

`tests/parity_cases.py` builds cases, runs the Python oracle on them, invokes
`morpheus-rs parity <kind>` on the same cases, and compares. Eleven surfaces
are checked. Tier 1 is bit-exact by specification; tier 2 allows 1e-6 — and is
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
| `net` | 2 | all eleven network heads, through all three entry points |
| `prior` | 2 | the legal-normalized prior and the search backup value |

The smoke slice grew from ~1.5 s to ~7 s at M3, almost all of it importing
torch and loading three TorchScript modules. That is the price of having the
network oracle in CI at all, and it buys the surface the milestone exists for.
AGENTS.md's 12 s ceiling governs the repo's `tests/`, which this is not part of;
if this suite is ever folded in, the torch import is the thing to move behind a
marker rather than the coverage to drop.

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

## The first surface where the tolerance does real work

`net` breaks the pattern of the first nine surfaces: it **cannot** be
bit-exact. Two engines summing the same products in different orders, with the
Rust kernels fusing their multiply-adds, will not land on the same float. So
§5's 1e-5 MAE budget is finally a budget rather than a formality — measured
worst is 4.05e-6, about 2.5× under it, against the thousandfold slack the
tensor's 1e-6 turned out to have.

Because the budget is real, a second check backs it: a cap on the worst single
element, set from measurement (1.05e-5 observed, 5e-5 enforced) rather than
from the specification. An MAE over 3,970 logits can absorb one badly wrong
value; the element cap cannot.

Every float surface now reports its **worst observed |Δ|** at the end of a run:

```
worst observed |Δ| (the headroom under each budget):
  net.full.pass_logit.mae            4.05e-06
  net.full.enemy_army_bins.max       1.05e-05
  prior.max                          6.56e-07
  tensor.max                         0
```

That line is the direct descendant of M2's finding. A tolerance is only
meaningful next to the number it is not being reached by, and the way a
precision bug survived M2 was by hiding under a budget nothing ever
approached.

### `prior` is the width rule again

The first version of `legal_normalized_policy` computed its softmax in f64 —
more accurate — and disagreed with the oracle in the eighth decimal.
`torch.softmax` runs at the tensor's dtype, so the Python's prior is an f32
quantity widened on the way out, and every decision morpheus has made was made
on that rounding. The port mirrors the width and keeps the f64 *result*,
exactly as the tensor builder mirrors `float32`/`float64` site by site. What is
left after that is one f32 ulp: two implementations of `exp`, and two orders of
summation over 3,970 terms.

Illegal actions are checked separately and without tolerance. They must be
exactly zero, because a mask that leaked a millionth of the probability mass
onto an unplayable move is a bug, not a rounding.

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

Full corpus (1,328 heavy frames from 20 games) — **502,870 cases**. The first
nine surfaces are bit-exact; the two M3 surfaces are inside the budgets in the
right-hand column, with the worst value a run actually reached.

| kind | cases | agreement |
| --- | ---: | --- |
| `transition` | 127,692 | bit-exact |
| `order` | 344,101 | bit-exact |
| `observe` | 21,282 | bit-exact |
| `mask` | 1,396 | bit-exact |
| `cost` | 1,396 | bit-exact |
| `memory` | 1,399 | bit-exact |
| `hash` | 1,399 | bit-exact |
| `tensor` | 1,399 | bit-exact (budget 1e-6) |
| `symmetry` | 8 | bit-exact |
| `net` | 1,399 | MAE ≤ 4.05e-6 (budget 1e-5), max 1.05e-5 (cap 5e-5) |
| `prior` | 1,399 | max 6.56e-7 (cap 5e-6) |

Mutation check: **44 of 51 caught**
([report](../../research/measurements/morpheus-rs-mutation-check.json)). All
seven survivors are equivalent mutants, not gaps; the tool fails on an
*unexplained* survivor, and — since M3 — on a mutation that never reached
production code at all.

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
- **GroupNorm sums over the pad columns** — the seven pad columns are zero, so
  summing them adds nothing. This one surviving is evidence *for* the
  zero-padding invariant; the observable half is the divisor, which is its own
  mutation and is caught.
- **softmax computed in double**, **illegal actions keep their softmax mass** —
  both explained under the M3 additions below.

M2 added three surfaces' worth of mutations and two of them found real holes:
no case had a **remembered castle observed as plain fog** (an emitted board
cannot produce one — a castle in fog encodes as type 5), and no previous action
carried a **half-split**, leaving the tensor's move-kind plane constant. Both
are now crafted cases in `_crafted_memory_pairs` and the action rotation.

M3 added nineteen more for the network, the GEMM, and the arithmetic between
the network and the search. Two of them are equivalent mutants worth naming:

- **softmax computed in double** — the result is narrowed back to f32
  immediately, and `exp` is close enough to correctly rounded that the two
  land on the same float for every logit the network produces. What the f32
  contract pins is the sum and the division, which that mutation does not
  touch.
- **illegal actions keep their softmax mass** — an illegal logit is filled
  with `f32::MIN`, so `exp(MIN − max)` is exactly zero and the mask multiply
  removes nothing. The Python multiplies too, for the same non-reason. The
  explicit zero stays because it makes "illegal means zero" a property of the
  code rather than of float underflow.

### A mutation can also miss the code entirely

The pad-column mutation for GroupNorm survived its first run, and the
write-up was one edit away from claiming a coverage hole. It had matched a
line inside `#[cfg(test)]` — it broke a unit test, never the bot, and the
parity harness had nothing to notice. `replace(..., 1)` takes the first
occurrence, and the first occurrence was in the test module.

So the tool now refuses any mutation whose match falls below `#[cfg(test)]`,
reports it as `in-test`, and fails the run alongside `stale`. A misaimed
mutation is worse than a missing one: a missing mutation is a known gap,
while a misaimed one reads as evidence.

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
