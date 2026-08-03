# Part 05: Belief filter

## Deliverable

Implement weighted full-state particles, exact observation filtering, ESS
resampling, bounded rejuvenation, collapse recovery, and per-node particle
reservoir input.

The same shared policy proposes enemy actions from the enemy perspective.

**Touches**

- `bots/`: add belief, proposal, recovery, and particle-summary modules.
- `arena/`: use trajectory replay as a measurement source without changing it.
- `scripts/`: add belief-recovery and opponent-belief measurements.
- `data/bot_versions/`: none.

## Prerequisites

- [Part 02: Transition kernel](02-transition-kernel.md)
- [Part 03: Observation and actions](03-observation-actions.md)
- [Part 04: Network and export](04-network-export.md)

## Source specifications

- [Belief state](../bots/morpheus/belief-state.md)
- [Observation tensor belief planes](../bots/morpheus/observation-tensor.md#plane-contract)
- [Opponent belief open question](../bots/morpheus/open-questions.md#opponent-belief-approximation)

## Defaults and replacement measurement

Start with these current defaults:

- 64 particles;
- ESS resampling below half the particle count;
- 8-turn recovery lag;
- beam width 8;
- 16 completed histories;
- 128 replayed transitions;
- no recursive opponent particles.

Replace the recovery bounds only from exact recovery rate and p99 cost after
forced proposal mismatch on recorded trajectories.

Replace level-zero opponent belief only from enemy-action log loss and
real-observation particle survival. Compare one bounded extra belief level only
if the default misses critical actions within the same deadline.

## Implementation boundary

Keep every particle rule-consistent. Visible cells, types, owners, armies,
turn, and public totals must match exactly after each real observation.

Batch and deduplicate enemy information tensors. Keep proposal policy
injection separate from filtering so deterministic fixtures do not require a
trained checkpoint.

Recovery must never replace a valid belief with an invalid one. A maximum-
entropy reconstruction must expose reduced confidence through `belief_ess`.

## Isolated test

```bash
python -m pytest bots/morpheus/tests/test_belief_filter.py \
  bots/morpheus/tests/test_belief_recovery.py \
  -m morpheus -q
```

```bash
python scripts/morpheus_measure.py belief-recovery \
  --trajectories data/trajectories \
  --force-proposal-mismatch \
  --output docs/research/measurements/morpheus-belief-recovery.json
```

```bash
python scripts/morpheus_measure.py opponent-belief \
  --trajectories data/trajectories \
  --output docs/research/measurements/morpheus-opponent-belief.json
```

```bash
python competition-module/competition/matchup.py \
  bots/morpheus/run.sh bots/smoke/run.sh \
  --mode competition --seed 0
```

## Specification gaps

The conditioned initial-general prior has no repository API. The design does
not define its sampling algorithm or its probability weights.

The terms `vision-changing action` and `maximum-entropy hidden allocation` are
not executable definitions. Particle importance weighting and acceptance
weights after policy-guided proposals are also not fully specified.

This part must record those definitions before implementation. It must not use
an unconstrained random fill as a silent substitute.

## Exit criterion

Answer `yes` if initialization, filtering, resampling, and recovery preserve
all rules and observations; forced mismatches recover whenever a legal history
exists inside the configured bounds; and both measurement reports include
recovery rate, p99 cost, log loss, and survival.

Answer `no` on any impossible state, public-total mismatch, silent collapse, or
unbounded recovery.
