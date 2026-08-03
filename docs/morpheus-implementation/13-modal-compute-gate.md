# Part 13: Modal compute gate

## Deliverable

Produce a measured `yes` or `no` decision for the complete training design
before the main trainer is implemented or launched.

The report must state:

- selected self-play backend;
- physical CPU cores per game;
- whether the two seat searches are sequential or parallel;
- games per worker and concurrent games per container;
- worker containers paired with one A100;
- games/hour and positions/hour;
- games/checkpoint and checkpoint count;
- every A100-hour charge;
- total fit against the 48-hour budget.

**Touches**

- `bots/`: benchmark the deployment-compatible stack without changing play.
- `arena/`: use engine and replay APIs without storing benchmark games.
- `scripts/`: add Modal benchmark orchestration.
- `data/bot_versions/`: none.
- `training/morpheus/`: add benchmark workers and report schema.

## Prerequisites

- [Part 00: Modal JAX preflight](00-modal-jax-preflight.md)
- [Part 00c: Measurement corpus](00c-measurement-corpus.md)
- [Part 09: Online qualification](09-online-qualification.md)
- [Part 10: Curriculum data](10-curriculum-data.md) with a resolved promotion
  rule.
- [Part 11: Self-play league](11-self-play-league.md)
- [Part 12: Training objective](12-training-objective.md)

## Source specifications

- [Training compute](../bots/morpheus/training.md#training-compute)
- [Training compute open question](../bots/morpheus/open-questions.md#training-compute)
- [Coupled online compute](../bots/morpheus/open-questions.md#coupled-online-compute)
- [Arena decision rule](../arena/decision-rule.md)

## Defaults and replacement measurement

Training uses Modal through the Python library, one Nvidia A100 80 GB learner,
and a total budget of approximately 48 A100 hours.

The current training default permits larger search settings, followed by a
deployment-matched self-play and calibration phase. It does not provide those
larger settings.

No CPU pairing, parallelism, games/hour, games/checkpoint, or checkpoint count
is assumed. This part must measure and state all of them.

## Implementation boundary

Keep Modal orchestration in thin scripts and training-only modules. Do not add
Modal or mutable training state to the bot closure. Do not store qualification
games in arena rating data.

### Resource layouts to compare

Always measure a CPU baseline. Compare one physical core per game with
sequential seat searches against two physical cores per game with one seat
search per core. Select by completed games per CPU-hour and end-to-end game
latency.

If Part 00 returns `yes`, also measure:

1. a hybrid path with CPU tree control and JAX-batched engine and network work;
2. a static-shape JAX path only if belief, selection, hashing, and backup avoid
   per-step host synchronization.

One A100 learner reads immutable shards from a Modal Volume. CPU self-play uses
Modal function mapping with a measured container cap. The final report must
name the exact worker count paired with the learner.

## Required calculations

Use steady-state completed games, not engine steps:

```text
games_per_worker_hour = completed_games / worker_hours
aggregate_games_per_hour = workers * games_per_worker_hour
positions_supply_per_hour = aggregate_games_per_hour * positions_per_game
total_games = aggregate_games_per_hour * self_play_wall_hours
games_per_checkpoint = floor(total_games / checkpoint_count)
```

The learner is supplied only when measured shard production meets measured
training consumption.

Count all A100 work:

```text
a100_hours =
    jax_preflight
  + throughput_qualification
  + learning_curve_pilot
  + objective_ablations
  + gpu_self_play
  + main_training
  + deployment_calibration
```

The result must be at most 48 hours. Report CPU hours separately.

## Minimum useful game count

The specs do not define enough games. Test candidate checkpoint cadences in a
bounded pilot. For each cadence, compare the later checkpoint with the prior
trained checkpoint on the fixed held-out panel and matched seeds.

The smallest useful cadence is the first one that:

- passes the curriculum confidence rule from Part 10;
- does not regress held-out belief calibration;
- receives `improvement` from the pairwise contrast in the arena decision rule.

If no tested cadence meets all three conditions, this part returns `no`. It
must not define success as all games that happen to fit the budget.

## Isolated test

```bash
python -m pytest training/morpheus/tests/test_compute_accounting.py \
  -m morpheus -q
```

```bash
modal run scripts/morpheus_modal.py::qualify_compute \
  --config training/morpheus/configs/modal-qualification.json
```

The command writes
`docs/research/measurements/morpheus-modal-qualification.{json,md}`.

## No-result path

If the first schedule returns `no`, remeasure in this order:

1. reduced training-time particles, simulations, or depth while preserving the
   same belief, matrix, action, tensor, and reward semantics;
2. fewer checkpoints with more games per checkpoint;
3. a narrower research scope that is explicitly not promotable.

Do not remove the deployment-matched final phase.

## Specification gaps

The minimum useful games/checkpoint, checkpoint count, training-time search
settings, and CPU price budget are absent. This part owns the cadence decision
rule above and must record its matched arena evidence.

The A100 can accelerate vectorized transitions and networks. It does not
automatically accelerate dynamic particle and tree control. Only the full
both-seat game measurement can select the backend.

## Exit criterion

Answer `yes` only if the report states an exact CPU/GPU layout, the learner is
not starved, the measured useful games/checkpoint and checkpoint count fit,
the cadence passes the stated curriculum, calibration, and pairwise rules,
deployment calibration fits, and total A100 usage is at most 48 hours.

Answer `no` otherwise and name the selected fallback or non-promotable scope.
No main training implementation starts without this answer.
