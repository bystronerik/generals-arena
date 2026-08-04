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
- [Part 00c: Measurement corpus](00c-measurement-corpus.md)

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

Replace the half-count ESS threshold only from a threshold sweep that reports
unique-particle count, real-observation survival, hidden-state calibration,
recovery frequency, and p99 update cost. Record the selected threshold in the
deployment configuration.

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
  --trajectories data/trajectories/morpheus-bootstrap \
  --force-proposal-mismatch \
  --output docs/research/measurements/morpheus-belief-recovery.json
```

```bash
python scripts/morpheus_measure.py opponent-belief \
  --trajectories data/trajectories/morpheus-bootstrap \
  --output docs/research/measurements/morpheus-opponent-belief.json
```

```bash
python scripts/morpheus_measure.py ess-threshold \
  --trajectories data/trajectories/morpheus-bootstrap \
  --output docs/research/measurements/morpheus-ess-threshold.json
```

```bash
python competition-module/competition/matchup.py \
  bots/morpheus/run.sh bots/smoke/run.sh \
  --mode competition --seed 0
```

## Specification gaps

Resolved in this part and recorded in
[belief-state.md](../bots/morpheus/belief-state.md#part-05-executable-definitions):

- conditioned initial-general prior (uniform over legal candidates);
- vision-changing action (visibility-mask difference vs enemy pass);
- maximum-entropy hidden allocation (uniform general and land cells; even army
  split; minimum `belief_ess`);
- filter importance ratio `1` under the shared-policy proposal; rejuvenation
  acceptance weight is the product of policy probabilities.

## Exit criterion

Answer `yes` if initialization, filtering, resampling, and recovery preserve
all rules and observations; forced mismatches recover whenever a legal history
exists inside the configured bounds; and the measurement reports include
recovery rate, p99 cost, log loss, survival, and a selected ESS threshold.

Answer `no` on any impossible state, public-total mismatch, silent collapse, or
unbounded recovery.
