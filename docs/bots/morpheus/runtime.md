# Runtime and deadline

## Decision

Morpheus treats the 150 ms reply limit as a hard deadline. The design default
is an internal deadline of **125 ms** after the complete frame is parsed, with
a 25 ms reserve — an **initial guess** for serialization, scheduling, and
timing variation. The shipped `deployment.json` plays a **140 ms** deadline
with a 10 ms reserve; that coupled budget predates live search and does not
fit under 150 ms once search runs — see
[thread pinning](thread-pinning.md#known-consequence-the-budget-no-longer-fits).

The design target is **32 completed simulations per normal move** (the code
default), with a minimum search target of 8. Both are **initial guesses**; the
shipped `deployment.json` targets **16** with minimum 8.

## First move

The first frame has 10 seconds of grace. Morpheus uses at most **8.5 seconds**
and reserves the remainder. This value is an **initial guess**.

First-move work is limited to:

- loading and validating the frozen weight artifact;
- creating the float32 TorchScript inference session;
- warming every network head and batch shape;
- classifying the initial terrain;
- creating the initial particle belief;
- allocating bounded tree storage.

No map seed or opponent-specific data is loaded.

## Work order

Every turn follows this order:

1. Store pass as the protocol-safe fallback.
2. Update visible memory and the particle belief.
3. Build legal masks.
4. Run one root network evaluation.
5. Store the highest-prior legal action as the policy fallback.
6. Run search in batches of up to 4 pending leaf evaluations.
7. Stop before the internal deadline.
8. Serialize the best available action.

When the policy proposal is enabled, belief proposals use one separate batch
of up to `max_proposal_batch` unique enemy tensors (design 64; deployed 8).
The deployed configuration uses the uniform proposal and spends zero forwards
on the belief path
([`belief-ablation-macaria.md`](../../research/measurements/belief-ablation-macaria.md)).
Search batch size 4 is an **initial guess**.

## Per-turn network budget

A forward-equivalent is one tensor through the network, even when a batch
executes many tensors together. This table is the Part 07 design budget; the
code enforces the 113 total plus per-component admission, not the per-consumer
rows.

| Consumer | Design maximum | Batch form |
| --- | ---: | --- |
| belief enemy-action proposals | 64 | one deduplicated batch (0 when the uniform proposal is active — the deployed default) |
| root policy and value | 1 | one tensor |
| search leaf values | 32 | batches of 4 |
| new enemy-table priors | 16 | separate `enemy_prior_batch` calls |
| total | 113 | hard accounting maximum, enforced in code |

The enemy-prior row is an **initial guess** and is not separately enforced.
Cached or deduplicated priors consume zero new forward-equivalents.

The total is a limit, not proof of feasibility. A valid deployment
configuration must fit belief update, root, target search, transitions, and
reply inside 125 ms at measured p99. Particle count and simulation target are
selected together. If belief plus root cannot fit, the artifact is rejected.

## Admission control

Morpheus keeps separate moving estimates for belief tensor construction,
belief-proposal batch, particle transitions, hashing, root inference, leaf
batches, enemy-prior batches, backup, and reply cost. It starts a network batch
only when its measured p99 time plus a guard fits before the internal
deadline. The design default guard is 10 ms, an **initial guess**; the shipped
`deployment.json` sets `admission_guard_ms = 0` because its `offline_p99_ms`
block already carries honest (higher) forecasts — see
[`morpheus-offline-p99-recalibration.md`](../../research/measurements/morpheus-offline-p99-recalibration.md)
and
[`morpheus-float32-p99.md`](../../research/measurements/morpheus-float32-p99.md).

The clock is monotonic. Morpheus checks it before selection, before inference,
and before starting another simulation. Work already in a network call is not
assumed cancelable.

Only a simulation that reaches backup changes tree statistics. A simulation
that stops at the deadline is discarded.

## Degradation path

Morpheus uses the following deterministic path:

| Completed simulations | Decision |
| ---: | --- |
| 1 or more | root average strategy (visit and prior tie-breaks) |
| 0 | highest-prior legal root action |
| no root result | pass |

The former 1–7 visit band is retired: three selectors switching by sim count
flipped the decision rule 220 times in one diagnosed 573-turn game (an
unpublished diagnostic recorded in commit `a6d2aa2`), which showed
up as abandoned attacks. At low sim counts the average strategy is dominated
by the prior, so the single selector degrades to prior-argmax on its own. The
`visit` fallback level remains in the telemetry schema for old traces only.

When the forecast falls below 16 completed simulations, search stops widening
before it selects more paths. This is a search-time rule, not a final-decision
rule.

If belief filtering consumes too much time, Morpheus keeps the last valid
particle set, applies visible contradictions, lowers `belief_ess`, and proceeds.
It performs full recovery on the next turn only when admission control permits.

## Memory bounds

The initial tree limit is 4,096 nodes, an **initial guess**. Morpheus evicts
least-recently-used nodes outside the current root subtree. Particle reservoirs
are bounded by the configured particle count.

Model, tree, particles, runtime, and process must stay below 2 GB. The design
target is below 256 MB resident memory, an **initial guess** that leaves a
large safety margin; the shipped `deployment.json` records a measured
`resident_memory_target_mb` of ~310 MB from qualification.

## Fault policy

Morpheus does not spend the 50-fault allowance as a time resource. A late,
missing, or malformed reply is an evaluation failure even when the match does
not yet forfeit.

Invalid game actions are also unacceptable. Legal masking and tuple
serialization must always produce a valid five-integer line.

## Alternatives rejected

Using all 150 ms is rejected because scheduler and output variance can turn a
small search gain into a fault. A fixed simulation count is rejected because
leaf cost changes with matrix width, belief recovery, and tree reuse.

## Failure mode

The 32-simulation target can be unrealistic for the chosen inference runtime.
The bot remains protocol-safe through the degradation path, but repeated
policy-only turns mean the deployed system is not the specified search bot.
Runtime acceptance must therefore measure completed simulations and p99 reply
time together.
