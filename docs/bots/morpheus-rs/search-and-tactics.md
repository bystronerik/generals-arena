# Morpheus-rs search, tactics, and the runtime controller

What milestone M5 of the [rewrite plan](rewrite-plan.md) ported, and the four
decisions that were not obvious. Parity method: [parity-harness.md](parity-harness.md).

M5 is the largest and least glamorous slice of the rewrite — `matrix.py`,
`tree.py`, `search.py`, `tactics.py`, `runtime.py`, `deployment.py`, about
5,200 lines of Python with, by the plan's own §7 estimate, no speed story to
motivate most of it. It is here because the decision depends on it and the
shipped binary cannot call Python. What follows is only the part a reader
cannot get from the source.

## The object graph becomes an arena

`InfoNode` in the Python holds direct references to its children and a `dict`
of enemy tables. The port keeps nodes in one `Vec` addressed by a `u32` handle,
with children in a digest-keyed map to handles — rewrite-plan §6's layout, and
also the only shape Rust's ownership rules accept without reference-counting
every node.

Two orderings inside that structure are load-bearing and are reproduced
deliberately rather than inherited from a container:

- `node.enemy_tables` is a `dict`, so `enemy_weights` walks it in **insertion
  order** and eviction removes from the middle. The port uses a `Vec` with
  linear lookup — at most eight entries — because a `HashMap` would iterate in
  an order the oracle never had.
- Python's `min` and `max` keep the **first** extreme; Rust's `max_by_key` keeps
  the last. On two adjacent stacks of equal size that is a different capture.
  Every selection in the port spells the rule out.

The node cap stays a refusal rather than a crash: the Python raises
`MemoryError` and `search.py` catches it to turn the path into a leaf
evaluation, so the refusal is part of the contract.

## One generator, two owners

`runtime.py` builds a single `np.random.Generator` and hands the *same object*
to `SearchController`. The belief's draws and the search's draws therefore
interleave in one stream, which is not incidental: the replay harness checks
draw order across the whole turn, and two independent generators would produce
a different sequence from the same seed. Rust cannot lend one `&mut` to two
structs, so the sharing is explicit — `rng::SharedRng`, a refcounted cell whose
borrow lasts exactly one call. Safe here because the crate is single-threaded
by construction and no draw site re-enters another.

## The one reduction that cannot be bit-exact, and why the oracle is at fault

Everything M1 through M4 ported matched the Python to the last bit.
`search/matrix.rs`
does not, and the reason is worth stating plainly because it is the same shape
as M4's `np.argsort` finding.

The Python writes `q_eff @ sigma_enemy` and `sigma_self @ u_self`. NumPy sends
`@` on `f64` to **BLAS** — Accelerate on the laptop, OpenBLAS on the x86
container — and the reduction order is the vendor's. Measured against 3,000
random simplex vectors at the widths the search actually uses (8, 12 and 16):

| candidate reduction | mean disagreement with `a @ b` | max |
| --- | ---: | ---: |
| sequential `for` loop | 0.48–0.67 ulp | 3 ulp |
| NumPy's own pairwise `sum` | 0.36–0.52 ulp | 3 ulp |

There is no order to copy. `np.dot` is a different answer on a different host,
so **the Python bot disagrees with itself across machines here**. The port
reduces with `npsum` — NumPy's pairwise order, the closest thing to a
convention this crate already reproduces — the `matrix` parity surface carries
a tolerance instead of bit-exactness, and the decision gate expects the
resulting flips on near-ties.

Measured over the corpus the tolerance is barely used: worst observed |Δ| is
8.88e-16 on a vector and 2.22e-16 on a scalar, against a 5e-12 cap. The cap is
set from measurement with headroom for a different BLAS, not from the
specification.

### …and regret matching plus turns that ulp into a different strategy

The `matrix` surface makes the BLAS gap look harmless. It is not, and the
`search` surface is where that shows.

`regret_matching_strategy` branches on `sum(max(regret, 0)) <= 0`: with no
positive regret it falls back to the prior, otherwise it normalizes the
positive entries. On the **first backup of a fresh enemy table** the true
regret is exactly zero — every joint entry is still at first-play urgency, so
`u_enemy` equals `v` — and whether the accumulated float lands on `0.0` or on
`2.8e-17` decides between those two branches. One ulp, deterministically, a
qualitatively different mixed strategy:

```
prior        [0, 0, 1]
regret 0.0       -> sigma [0,    0,    1  ]   (fall back to the prior)
regret 2.8e-17   -> sigma [0.5,  0.5,  0  ]   (normalize the positives)
```

That is not a porting artefact. Re-running the *oracle* with its `@` replaced
by NumPy's own pairwise reduction — the one substitution the port makes —
reproduces the Rust answer bit-for-bit, on every corpus case. **The Python
bot's search is a function of whichever BLAS NumPy was built against**, at
every new enemy table, on every turn.

Two consequences worth carrying forward:

- The `search` parity surface compares strictly against a *pairwise-dot*
  oracle, and separately tallies how many statistics the shipped BLAS oracle
  moves and how many regret branches it flips. The port has to match the first
  exactly; the second is a measurement of the oracle, reported rather than
  tolerated.
- **M6 should not expect identical search decisions at parity knobs**, and a
  divergence there is not automatically R2's "unfound logic bug". The plan's
  M6 exit gate reads "at identical configuration the Rust bot should play the
  same bot"; that holds for the no-search decision (measured: 100% over 1,322
  frames) and does not hold, even in principle, once a simulation completes.

## `constrain_nn_action` does not read the belief

rewrite-plan §5 records a "final-state wrinkle": the hard-rule layer *consumes*
the belief, so decision parity on any frame needs the captured belief snapshot
as an input, and the `decide` parity subcommand must accept it explicitly.

It does not. `runtime.py` passes the argument, `constrain_nn_action` accepts
it, and nothing in the body touches it — verified against the declared-final
tree. The belief does reach the decision, through `heuristic_action_scores` on
the shaping path, but not through the hard rules, so a `constrain` parity case
needs no belief at all and the surface deliberately does not send one.

The parameter is kept on the Rust signature anyway. Dropping it would hide the
discrepancy from the next reader of §5, and the argument the Python passes is
the thing that made the claim plausible in the first place.

## What the tactical port is checked against

Five surfaces, split by subsystem exactly as R5's fallback proposes (play mask →
shaping → hard rules → planners), plus the decision they combine into:

| surface | what it pins |
| --- | --- |
| `playmask` | the mask, plus how many bits the garrison floor and castle anchor removed |
| `shaping` | `heuristic_action_scores` and `blend_prior`, bit-exact in f64 |
| `candidates` | `mandatory_action_indices` and the stable policy ordering |
| `planners` | every planner's answer, each with an explicit present/absent flag |
| `constrain` | the action the hard rules commit, and both oscillation paths |
| `search` | the whole tree after N batches, against a pairwise-dot oracle |
| `runtime` | the two controller functions `decide` cannot reach |
| `decide` | the whole no-search decision, network included |

`playmask` emits the *removed-bit count* alongside the mask because both
sub-rules are subtractive and both withdraw entirely when they would leave no
non-pass action — so a port that never applied either would produce a mask
identical to `legal_mask` on most frames and differ only where it matters.

`planners` emits a present/absent flag per answer rather than a sentinel cell,
because a planner that returns nothing everywhere is the failure mode worth
catching and `(-1, -1)` is a plausible-looking answer.

### `decide` is the tier-3 gate in the form a frame can answer

§5 asks for "identical chosen action on ≥99% of frames with frozen RNG and
fixed simulation count". A *full-search* decision depends on cross-turn state —
the tree, the rolling history digest, the estimator windows — that no single
frame carries, and the corpus is a record of what happened rather than a script
that can be re-run.

The **no-search** decision has no such dependency, and it is not a corner: M0
measured belief update plus root inference at 166.7 ms against a 140 ms
internal deadline on the laptop, so zero completed simulations is a common
outcome, and §5 requires those frames to agree *exactly*. The surface therefore
runs the recorded root tensor through both engines, blends, takes
`highest_prior_legal`, and applies the hard rules — the exact path
`runtime.decide` commits when the search does not finish. The tensor rides the
wire rather than being rebuilt, so a divergence lands on the decision layer
instead of on the tensor builder `tensor` already checks.

Each case also emits the **tie margin**: how much shaped prior separates the
committed action from the runner-up. §5 accepts a divergence only when it traces
to a within-tolerance tie, and the margin is what makes that judgeable rather
than arguable.

## What is deliberately not faster yet

`blocks_oscillation` calls `newly_revealed_cells`, which runs two whole-board
visibility dilations, and `best_prior_legal_action` calls it once per legal
action — thousands of dilations per redirect in the Python. The port hoists a
single `reveal_count_grid`, which is the same quantity by construction, and the
`constrain` surface runs *both* paths on every case so the equivalence is
checked rather than asserted.

Nothing else is optimised. `transition` still clones a state per call, the
enemy-prior LRU is a linear scan, and the tactical layer recomputes distance
fields that could be shared. All of that is M7's business: M5's job was to make
the two bots play the same game, and a rewrite that also changed the arithmetic
would have made every parity failure ambiguous.
