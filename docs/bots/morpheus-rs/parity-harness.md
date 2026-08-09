# Morpheus-rs parity harness

How a ported surface is proved equal to the Python oracle, and — the part that
turned out to matter more — how we know the proof is worth anything.

Established at milestone M1 of the [rewrite plan](rewrite-plan.md), extended at
M2 to tier 2, at M3 to the first surface that cannot be bit-exact, at M4 to the
first surfaces that consume **randomness**, at M5 to the first surface where
the *oracle* is not reproducible and to tier 3 — the decision itself — and at M6
to the coverage M5 said it did not have. The corpus it runs on is described in
[parity-corpus.md](parity-corpus.md).

## Shape

```bash
pytest bots/morpheus-rs/tests/          # smoke slice, ~9 s
bots/morpheus-rs/tools/run_parity.sh    # full corpus + mutation check, minutes
```

`tests/parity_cases.py` builds cases, runs the Python oracle on them, invokes
`morpheus-rs parity <kind>` on the same cases, and compares. Thirty-one
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
| `matrix` | 2 | the regret-matching cycle end to end |
| `runtime` | 1 | the two controller functions `decide` cannot reach |
| `evict` | 1 | which enemy table the retention rule drops, from a stated set |
| `playmask` | 1 | the play mask, and how many bits each sub-rule removed |
| `candidates` | 1 | mandatory actions, then the stable policy ordering |
| `planners` | 1 | every planner's answer, each with a present/absent flag |
| `shaping` | 2 | `heuristic_action_scores` and the blended prior |
| `constrain` | 1 | the action the hard rules commit |
| `search` | 1 | the whole tree after three batches (see below) |
| `decide` | 3 | the whole no-search decision, network included |

The smoke slice grew from ~1.5 s to ~7 s at M3, almost all of it importing
torch and loading three TorchScript modules, to ~8 s at M4, and to ~10 s at M5. That is the
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

Full corpus (1,328 heavy frames from 20 games) — **533,726 cases**. Everything
except the network and the BLAS-fed matrix math is bit-exact; the surfaces that
cannot be are inside the budgets in the right-hand column, with the worst value
a run actually reached.

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
| `matrix` | 15 | max 8.88e-16 (cap 5e-12) |
| `runtime` | 7 | bit-exact |
| `evict` | 7 | bit-exact |
| `playmask` | 1,391 | bit-exact |
| `candidates` | 5,564 | bit-exact |
| `planners` | 1,391 | bit-exact |
| `shaping` | 1,391 | bit-exact (budget 1e-9) |
| `constrain` | 5,564 | bit-exact |
| `search` | 1,364 | bit-exact vs the pairwise-dot oracle, **every node** |
| `decide` | 1,322 | identical action on **1,322 / 1,322** |

`initbelief` exists because mutation testing found `initialize_belief`
completely uncovered: the corpus records the beliefs that exist, never the
frame that created one, so deleting the minimum-separation rule from the prior
changed nothing anywhere. The surface emits the candidate **support** as well
as the sampled belief, because a belief drawn from a wrong support can agree by
luck.

`decide` is the tier-3 gate and it reads 100%, not the 99% §5 asks for. The
closest call over the whole corpus separated the committed action from the
runner-up by **3.42e-05** of shaped prior — 52× the `prior` surface's worst
disagreement, so no frame came near a tie flip. `shaping` matching to the last
bit is why: with the scores identical, the only float that could move the
decision is the network prior, and 6.6e-7 does not move it.

`search` runs 1,364 trees to completion and matches the pairwise-dot oracle
exactly — every node of every tree, not only the root. Against the *shipped*
oracle, BLAS moved 2,094 statistic vectors and flipped the regret-matching
branch on 567 enemy tables, deciding on magnitudes up to 8.88e-16. That is the
oracle's number, not the port's.

`toplegal` was expected to need a tolerance and does not. Two implementations
of `exp` over 3,970 f64 logits land on the same double for every logit here,
and `npsum` reproduces the reduction exactly, so bit-exactness is what the
harness enforces and the 1e-15 figure is only the floor to fall back to.
Which leaves `net` as the only surface where a budget does real work.

### M5: the pass that says the coverage is not finished

**118 of 167 caught, 49 survivors** — and unlike M4's fourteen, most of these
are still open. The number moved twice under its own findings: 103 → 115 → 118,
across three passes.

The first pass came back with 63 survivors and one shape. *Every* rule gated on
a **game phase** survived — the garrison window, the castle window, a live
threat, a reachable kill — and so did every behaviour that needs a **deep or
contended tree**. Neither is reachable from the smoke slice: seven recorded
frames from a capture host, plus `synthetic_states()`, which was built for M1's
transition boundaries on a 5×5 board.

The response was `tactical_positions()`: eleven boards built from the rule
backwards, one gate at a time, with the observation and memory constructed
directly rather than emitted from a `GameState` — the tactical layer reads only
those two, and fog would otherwise have to be arranged rather than stated.
Plus richer `search` configurations (two enemy tables against four particles to
force eviction, twenty-four nodes to force the cap's refusal, a `freeze` flag
the corpus never sets) and a `runtime` surface for the two controller functions
`decide` cannot reach — `highest_prior_legal`'s mask never binds on a
legal-normalized prior, and `nearest_rank_p99` only appears in an admission
decision the harness does not replay.

That closed fifteen. What is left is diagnosed rather than mysterious, and it
falls into four kinds:

- **Two gates at once.** The castle anchor standing down for a threat, the
  kill/defence tie, the garrison release refusing mid-emergency, the tithe
  yielding to an enemy take — each needs one board where *both* rules are in
  range, and the eleven positions each isolate one.
- **An exact numeric coincidence.** A threat whose arrival equals
  `garrison + d/2`; a reinforcement at exactly `d - 1` rather than `d`; a
  half-split whose remainder straddles the floor, which needs a garrison of
  precisely `2 * floor - 1`.
- **An exemption shadowed by the next one.** `blocks_oscillation` returns early
  on enemy land, then again on new vision. Deleting the first changes nothing
  unless the reverse destination is enemy-owned *and* its 3×3 box is already
  fully visible; deleting the second changes nothing unless the destination
  borders fog. The two positions added for these reach neither condition —
  a diagnosis, not a mystery.
- **Tree contention.** Eviction, the retention score, the enemy-hash cache's
  version key, child reuse, terminal leaves. Four batches of four over a
  24-node cap still does not visit the same child twice.

Three survivors are now **explained** rather than open, and all three are
equivalent by construction: owning a cell already makes its 3×3 box visible, so
the reveal grid's early return only saves work; a legal move's destination is
adjacent to a reachable source and therefore reachable, so `path_progress`
cannot leave the goal component; and scores are clamped non-negative, so
commitment hysteresis's `> 0` guard excludes only zero, and `0 × 1.5` is zero.

The honest summary: **M5's parity result is proved to the standard the earlier
milestones set, and its mutation coverage is not.** The 533,390 recorded and
synthetic cases agree bit-for-bit and the decision surface reads 100%, but
forty-six behaviours in the tactical and search layers have no case that can
tell them from their negation. Each is named above with what it would take.

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

## M5: a surface whose oracle is not reproducible

Every surface through M4 could in principle be matched to the last bit. `matrix`
cannot, and for a reason that is the *oracle's*, not the port's.

The Python writes `q_eff @ sigma_enemy`, and NumPy sends `@` on `f64` to BLAS —
Accelerate on the laptop, OpenBLAS on the x86 container. The reduction order is
the vendor's, and measured against 3,000 random simplex vectors at the widths
the search uses it matches neither a sequential sum (mean 0.5 ulp, max 3) nor
NumPy's own pairwise reduction (mean 0.4 ulp, max 3). There is no order to
copy: **the Python bot disagrees with itself across hosts here**, exactly as
`np.argsort` does under AVX-512.

So `matrix.rs` reduces with `npsum`, the surface carries a tolerance set from
measurement (5e-12, against a worst observed 4.4e-16), and the decision gate
expects the resulting flips on near-ties rather than being surprised by them.
That is the third host-conditional behaviour the harness has had to name, after
`npsum` and `argsort`, and the pattern is now clear enough to state as a rule:
**when a surface will not go bit-exact, check whether the oracle is stable
before assuming the port is wrong.**

## M6: the tactical positions M5's eleven could not reach

M5 grouped its tactical survivors three ways, and each group needed a different
kind of board. Fifteen positions were added, one rule each, and the arithmetic
that puts a gate on its boundary is written into the source rather than tuned
by trial — a position that only *happens* to sit on a boundary stops sitting on
it the next time a constant moves, and then quietly stops testing anything.

- **Two gates in range at once.** The castle anchor standing down for a general
  threat needs a build site 18 cells from the general (any nearer and the build
  surcharge disqualifies it), 16 from the enemy (the safe-distance rule wants
  4), *and* a 30-army stack two steps from a garrison of 3. The kill/defence
  tie needs a march that reaches the enemy general in two steps and a threat
  that reaches ours in two. The tithe yielding to a take needs a tithe turn, an
  underfunded site, and an action that captures.
- **An exact numeric coincidence.** A split leaves `ceil(a/2)` and the mutation
  leaves `floor(a/2)`: the two differ only on an odd garrison and only change
  the ban when the floor falls exactly between them, so the garrison must be
  `2 * floor - 1` — 19 against a floor of 10, and no other value works. The
  same position is the only one where the release *factor* is decidable, for
  the mirror-image reason.
- **An exemption shadowed by the next one.** M5's position for "a reverse onto
  enemy land is not oscillation" reached the *vision* exemption first and
  returned there, so the enemy-land branch was never the reason for the answer.
  Owning both cells flanking the enemy makes its 3×3 box already visible, and
  then only the enemy-land rule can allow the reverse. The vision exemption's
  own position had the opposite problem: it reversed onto own land, and an
  owned cell reveals nothing by definition.
- **A window pointed at the wrong move.** The oscillation history keeps the
  last eight moves. M5's corridor had ten and reversed the eighth — which both
  the last-eight and the first-eight window contain. Nine moves reversing the
  ninth is decidable; ten reversing the eighth is not.

One rule needed something no board can supply. The shaping blend clips a
heuristic to a 10× nudge either way, and every own-land non-progressing move
scores ~0.01 against a top of ~309, so the heuristic term alone spans 100×. A
position cannot make the redirect prefer a retreat; only a *prior* can, and it
has to insist by more than 100×. `_frame_like` therefore takes an optional
unshaped prior, and that case states one — which is exactly the situation the
rule exists to overrule: the network insisting on a retreat.

## M6: what the `search` surface was not looking at

M5 ended with forty-six behaviours in the tactical and search layers that no
case could tell from their negation, and said closing that was the first thing
M6 should do. Half of them fell to positions built the way M5's eleven were.
The other half did not, and the reason was never the positions: **the surface
was comparing the wrong thing, on a tree that could not diverge, fed by an
evaluator that made a whole class of arithmetic invisible.** Three causes, each
found by asking why a purpose-built case still could not see a break.

### The four particles were one board

`_search_setups` built four particles that "disagree about the enemy's armies"
by calling `state._replace()` and writing into the copy. `_replace` copies the
*tuple*; every array inside it is shared. All four particles were one board, the
last write won on all of them, they hashed to one enemy view, and the node
therefore never held more than one enemy table — so eviction, the LRU, pinning
and protection had nothing to decide, on any case, ever. That is the whole
explanation for four M5 survivors, and the comment above them asserted the
opposite. `_copy_state` fixes it; the contended setup below then reaches
seventeen over-capacity evictions, ten where the protected table carried the
lowest retention score and four where a pinned one did.

### A root-only comparison cannot see a child

Even with the trees fixed, a mutation inside a child node survived. The
surface compared the root's statistics, its enemy tables, and a handful of
counters. Nothing else can reach those:

- a leaf value comes from the evaluator and is applied **unchanged** to every
  edge on the path, so a child's own statistics never travel upward;
- `Replay` returns the oracle's sampled index whatever distribution this side
  built, so a divergence inside a child does not even change the tree's shape.

The surface now emits **every node** in creation order — N, turn, memory
digest, reservoir size, candidates, prior/regret/average, and every enemy
table's columns, LRU counters and joint statistics. The memory digest is what
catches a child folding its observation into the wrong memory; the rest is
what catches everything a child computes.

### A constant leaf value hides every weighting

The last cause is the subtlest and the most general. `ScriptedEvaluator` returns
one number for every leaf — chosen at M5 so a search comparison could not fail
on the last bit of a softmax. But with every leaf worth the same, every `q`
entry ends up equal, and **an average of equal numbers does not depend on its
weights**. The enemy-hash cache key, the reservoir's particle weights and the
marginal aggregation were all unobservable for that one reason.

The fix keeps what the scripted evaluator is for and drops what it cost: the
value is now a deterministic function of the leaf's own observation payload
(an integer sum folded into `[-1, 1]`), so it varies per leaf without a network
or a softmax anywhere, and lands on the same double on both sides. It is armed
per setup — the recorded frames keep the constant evaluator, so M5's tally of
what the oracle's BLAS moves is still measured on real frames.

### Some rules cannot be reached by playing at all

`evict` exists for the reason `runtime` does. The retention score is
`last_used + 0.25 * ln1p(touch_count)` and `last_used` is an integer, so the
touch term can only ever decide an exact tie — and in a real tree there are no
ties, because `last_used` is the node's visit counter at the table's last
backup and a backup advances it. Measured over the search setups: 680 backups,
never two candidates sharing a `last_used`. The surface states the table set
instead of playing for it, and the seven cases are written from the rule
backwards: the touch term deciding, an exact tie, a pinned minimum, a protected
minimum, a double eviction, an all-pinned refusal, and nothing to do.

One case in it is worth naming because the first version was wrong in the way
this whole harness is about. The touch-term case lists the *well-used* table
first on purpose: with the term deleted the two scores tie and the first
minimum wins, which in the other order is the same answer the rule gives. The
order of the list is the test.

## The `search` surface, and the two oracles it needs

`decide` runs at zero simulations by construction and `matrix` checks the
arithmetic without the storage that feeds it, so neither reaches the search.
`search` does: it builds a real tree from a corpus frame — selection,
progressive widening, enemy-table installation and eviction, leaf expansion,
backup — and compares every statistic it accumulates.

The evaluator is **scripted**, not the network. The two engines' priors agree to
6.6e-7, which is enough to reorder a near-tie in a candidate list, and a search
comparison that could fail on the last bit of a softmax would prove nothing
about the search. With identical priors on both sides, a disagreement in the
tree is the tree's.

Except that the first run of the surface disagreed anyway, and the cause was
the BLAS reduction above — amplified. `regret_matching_strategy` branches on
`sum(max(regret, 0)) <= 0`, and on the first backup of a fresh enemy table the
true regret is exactly zero, so whether the accumulated float is `0.0` or
`2.8e-17` decides between "fall back to the prior" and "normalize the positive
entries". One ulp becomes a different mixed strategy. Re-running the oracle with
its `@` replaced by NumPy's own pairwise reduction — the one substitution the
port makes — reproduces the Rust answer bit-for-bit.

So the surface carries **two oracles**. The port must match the pairwise-dot
one to the last bit; the shipped BLAS one is compared against *that* and every
statistic it moves is tallied:

```
search: the oracle's BLAS moved N statistic vector(s) over M case(s),
        flipping the regret-matching branch on K enemy table(s)
```

That line is a measurement of the oracle, not a budget the port is spending.
It is also the reason M6 should not expect identical search decisions at parity
knobs: the Python bot's search is a function of whichever BLAS NumPy was built
against, and no port can be faithful to all of them at once.

## Tier 3: the decision, in the form a frame can answer

§5 asks for "identical chosen action on ≥99% of frames with frozen RNG and
fixed simulation count", and separately that no-search frames agree *exactly*.

A full-search decision cannot be reconstructed from one frame: it depends on the
tree, the rolling history digest and the estimator windows, none of which a
frame carries, and the corpus is a record of what happened rather than a script
that reproduces it. The **no-search** decision has no such dependency — and it
is not a corner case. M0 measured belief update plus root inference at 166.7 ms
against a 140 ms internal deadline, so zero completed simulations is what the
bot does on a large fraction of turns.

`decide` therefore runs the recorded root tensor through both engines, blends
the prior, takes `highest_prior_legal`, and applies the hard rules: exactly the
path `runtime.decide` commits at zero simulations. The tensor rides the wire
rather than being rebuilt, so a divergence lands on the decision layer and not
on the tensor builder `tensor` already checks.

Each case also emits the **tie margin** — how much shaped prior separates the
committed action from the runner-up. §5 accepts a divergence only when it traces
to a within-tolerance tie, and without the margin that judgement is an argument
rather than a number.

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
