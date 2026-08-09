# Morpheus-rs parity harness

How a ported surface is proved equal to the Python oracle, and — the part that
turned out to matter more — how we know the proof is worth anything.

Established at milestone M1 of the [rewrite plan](rewrite-plan.md), extended at
M2 to tier 2, at M3 to the first surface that cannot be bit-exact, and at M4 to
the first surfaces that consume **randomness**. The corpus it runs on is
described in [parity-corpus.md](parity-corpus.md).

## Shape

```bash
pytest bots/morpheus-rs/tests/          # smoke slice, ~9 s
bots/morpheus-rs/tools/run_parity.sh    # full corpus + mutation check, minutes
```

`tests/parity_cases.py` builds cases, runs the Python oracle on them, invokes
`morpheus-rs parity <kind>` on the same cases, and compares. Twenty-one
surfaces are checked. Tier 1 is bit-exact by specification; tier 2 allows 1e-6 — and is
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
| `npsum` | 1 | NumPy's pairwise `f64` reduction |
| `argsort` | 1 | `np.argsort(-scores)`, ties included |
| `summary` | 1 | the six belief planes, plus ESS in full precision |
| `propose` | 1 | each particle's proposal distribution *and* its sampled action |
| `filter` | 1 | the filtered belief: weights, states, enemy memories, history |
| `rejuvenate` | 1 | the belief a bounded beam replay reconstructs |
| `maxent` | 1 | the maximum-entropy reconstruction, or its refusal |
| `reservoir` | 1 | Algorithm R admission, sampling, and replace-from-belief |
| `toplegal` | 2 | the masked softmax and the ranked candidate list |
| `initbelief` | 1 | the initial prior's support, and the belief sampled from it |

The smoke slice grew from ~1.5 s to ~7 s at M3, almost all of it importing
torch and loading three TorchScript modules, and to ~8 s at M4. That is the
price of having the network oracle in CI at all, and it buys the surface the
milestone exists for. AGENTS.md's 12 s ceiling governs the repo's `tests/`,
which this is not part of; if this suite is ever folded in, the torch import is
the thing to move behind a marker rather than the coverage to drop.

## Randomness: recorded, replayed, and argued with

M4's surfaces are the first that draw. The plan's §5 settled the approach —
record NumPy's draws, replay them in Rust, never reimplement the bit generator
— and the harness closes it: the Python oracle runs behind the *same*
`RecordingGenerator` the M0 capture wraps the live bot in, seeded per case, and
its draw log is encoded into the case stream.

`rng::Replay` then **verifies every argument** before handing back a recorded
result. NumPy's `size=None` and `size=1` are different calls and stay
distinguishable; so are populations of different sizes, `replace`, and whether
a `p` vector was passed. A port that samples one extra time, or from a
differently sized population, fails on that call with the draw index and both
signatures in the message rather than diverging quietly three decisions later.

**Replay creates a new way to be wrong, and one surface exists to close it.**
Because `Replay` returns the oracle's index whatever this side computed,
checking only the *sampled actions* would pass a port whose every probability
was wrong. So `propose` emits each particle's distribution — sparsely, since a
uniform mask has a few hundred non-zeros out of 3,970 — alongside the action,
and compares both.

One draw site is not on the shared stream at all: `replace_from_belief` builds
`np.random.default_rng(0)` *inside* the method. The harness patches the factory
for the duration so those draws land in the same ordered log, which means the
Rust side has to make them at the same point in the sequence. What the port
does not reproduce is *which* particles that seed picks; see
[belief.md](belief.md) for why, and for what is preserved instead.

## Two NumPy behaviours have their own surfaces

`npsum` and `argsort` check functions that exist only to reproduce NumPy, and
they exist as surfaces because both are **host-conditional**:

- NumPy's `f64` reduction is pairwise, and a host whose NumPy vectorizes it
  differently would change the ESS, which decides whether a resample runs,
  which consumes a draw — a desynchronized replay stream three calls after the
  cause.
- NumPy ≥ 2.0 dispatches `argsort` on 64-bit dtypes to `x86-simd-sort` under
  AVX-512-SKX, a different unstable algorithm with a different tie order. Under
  the deployed uniform proposal *every* legal action ties, so on such a host the
  Python bot itself would rank recovery candidates differently.

Neither is a bug the port can fix. Making each a surface turns "this host's
NumPy is not the one this was written against" into one named failure instead
of a recovery mismatch nobody can trace.

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

Full corpus (1,328 heavy frames from 20 games) — **515,710 cases**. Everything
except the network is bit-exact; the two surfaces that cannot be are inside the
budgets in the right-hand column, with the worst value a run actually reached.

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
| `npsum` | 34 | bit-exact |
| `argsort` | 13 | bit-exact |
| `summary` | 1,352 | bit-exact, ESS included |
| `propose` | 1,352 | bit-exact, distributions, actions and telemetry |
| `filter` | 4,028 | bit-exact |
| `rejuvenate` | 310 | bit-exact |
| `maxent` | 1,328 | bit-exact |
| `reservoir` | 4 | bit-exact |
| `toplegal` | 4,197 | bit-exact (floor 1e-15) |
| `initbelief` | 222 | bit-exact, prior support included |

`initbelief` exists because mutation testing found `initialize_belief`
completely uncovered: the corpus records the beliefs that exist, never the
frame that created one, so deleting the minimum-separation rule from the prior
changed nothing anywhere. The surface emits the candidate **support** as well
as the sampled belief, because a belief drawn from a wrong support can agree by
luck.

`toplegal` was expected to need a tolerance and does not. Two implementations
of `exp` over 3,970 f64 logits land on the same double for every logit here,
and `npsum` reproduces the reduction exactly, so bit-exactness is what the
harness enforces and the 1e-15 figure is only the floor to fall back to.
Which leaves `net` as the only surface where a budget does real work.

Mutation check: **71 of 86 caught**
([report](../../research/measurements/morpheus-rs-mutation-check.json)). All
fifteen survivors are equivalent mutants, not gaps; the tool fails on an
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

### M4: the milestone where the mutation pass found the most

Thirty-five new mutations, and the first run caught only twenty-one of them.
Fourteen unexplained survivors is by far the worst result any milestone has had
— and every one of them was a real hole, not a bug in the port. The parity run
had been green the whole time.

Four kinds of hole showed up, and they generalize:

- **A whole entry point was uncovered.** `initialize_belief` had no surface at
  all: deleting the minimum-separation rule from the initial prior changed
  nothing anywhere, because the corpus records beliefs that already exist and
  never the frame that created one. Fixed by the `initbelief` surface.
- **The answer encoding hid the difference.** `filter` emitted each particle's
  history as a *length*. `_append_history` drops from the old end when the
  window overflows, so swapping that for a truncation keeps the length and the
  wrong eight frames. Now every frame's action pair and timestep rides out.
- **A comparison was written and then not made.** The proposal information key
  reaches exactly one observable on the deployed path —
  `n_unique_info_keys` — and the comparison checked only the particle count.
  A key that ignored the previous action passed. Now all five counters are
  compared.
- **The corpus is regular in ways that hide arithmetic.** Every captured belief
  holds eight particles whose weights already sum to one, all of them `1/8`.
  Normalizing a normalized set is the identity, `1/8` is exact in `f32`, and
  eight equal values sum the same in any order — so `ess`'s division,
  `summarize_belief`'s rescale, the belief planes' `f64` accumulator and
  NumPy's pairwise sum were *all* unreachable at once. Four mutations, one
  cause. Fixed by synthetic beliefs with unnormalized, irregular, negative and
  all-zero weights, and by a filter variant that gives every particle the same
  enemy action so more than one of them survives to be normalized.

The last one is the lesson worth keeping. **A corpus is not a sample of inputs;
it is a sample of the inputs the deployed configuration produces**, and a
configuration that happens to use a power-of-two particle count with equal
weights makes a whole class of arithmetic bugs invisible. The synthetic cases
are not there to be realistic. They are there to be irregular.

### A mutation can also miss the code entirely

The pad-column mutation for GroupNorm survived its first run, and the
write-up was one edit away from claiming a coverage hole. It had matched a
line inside `#[cfg(test)]` — it broke a unit test, never the bot, and the
parity harness had nothing to notice. `replace(..., 1)` takes the first
occurrence, and the first occurrence was in the test module.

M4 hit the other half of that rule. The depthwise-kernel mutation went `stale`:
M3's halo rewrite replaced the `ky`/`kx` loop with a flat tap index, so the
pattern stopped matching and the mutation had silently stopped testing anything
some time before anyone looked. Failing on `stale` is what surfaced it; the
mutation is re-aimed at the current source.

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
3. Add the kind to `KINDS` in `tests/test_parity_tier1.py` and to the `--kinds`
   default in `parity_cases.py`.
4. **Add mutations for it** in `tools/mutation_check.py` and run them. A
   surface with no mutation coverage is a surface nobody has shown the harness
   can check.
5. If the surface is the *only* one that can see a given source file, add it to
   `FILE_SURFACES` in the same file — see below.

### Scoping, and why a survivor ignores it

Running all twenty surfaces per mutation was the honest default until M4 made
it the dominant cost: eighty-odd mutations against a torch import and two
million integers through a pipe. `FILE_SURFACES` maps each source file to the
kinds that could possibly see a break in it, and the board layer
(`transition.rs`, `observe.rs`, `action.rs`, `memory.rs`, `state.rs`) still
maps to *everything*, because every belief and every tensor is downstream of it.

An over-narrow entry would report a caught mutation as a survivor, which reads
as a coverage hole and is the most expensive kind of wrong. So a survivor is
never allowed to rest on the scoping: before one is recorded, the full surface
set is re-run. A narrow map therefore costs a slow run, not a false finding.
