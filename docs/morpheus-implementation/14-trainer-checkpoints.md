# Part 14: Trainer and checkpoints

## Deliverable

Implement the resumable Modal training loop, immutable run manifest, replay
buffer reader, optimizer state, checkpoint writer, league updates, and final
deployment-matched calibration.

Training checkpoints remain outside the bot closure until Part 15 freezes one.

**Touches**

- `bots/`: import the shared model contract without writing mutable weights.
- `arena/`: none.
- `scripts/`: add Modal train, resume, inspect, and download entry points.
- `data/bot_versions/`: none.
- `training/morpheus/`: add trainer, checkpoint, and run-manifest modules.

## Prerequisites

- [Part 10: Curriculum data](10-curriculum-data.md) with an executable
  non-degenerate-WDL rule.
- [Part 11: Self-play league](11-self-play-league.md) with a named fixed panel.
- [Part 12: Training objective](12-training-objective.md) with selected explicit
  loss and exploration values.
- [Part 13: Modal compute gate](13-modal-compute-gate.md) with a `yes` result.

## Source specifications

- [Training](../bots/morpheus/training.md)
- [Network training behavior](../bots/morpheus/network.md#training-behavior)
- [Checkpoint promotion](../bots/morpheus/training.md#checkpoint-promotion)

## Defaults and replacement measurement

This part has no hidden defaults. It consumes the explicit objective settings
from Part 12 and resource, search, cadence, and budget settings from Part 13.

The 70/30 opponent mixture remains the Part 11 starting default until its
measurement replaces it.

## Implementation boundary

Store shards and checkpoints in a named Modal Volume. A checkpoint contains
model, optimizer, scheduler, league, curriculum, RNG, consumed-shard, and
budget-accounting state.

Make checkpoint writes atomic and immutable. Resume only when schema, engine
era, tensor schema, action schema, and run manifest match.

Periodically save training snapshots without promoting them. Before a
checkpoint can reach Part 15, run final self-play and calibration with the
deployment settings selected in Part 09.

Do not change `GameRecord`. Training samples remain outside `data/games/`.

## Isolated test

```bash
python -m pytest \
  training/morpheus/tests/test_trainer_step.py \
  training/morpheus/tests/test_checkpoint_resume.py \
  -m morpheus -q
```

The test must overfit a tiny deterministic shard, stop, resume, and produce the
same next update as an uninterrupted run.

```bash
modal run scripts/morpheus_modal.py::train \
  --config training/morpheus/configs/promotable-run.json
```

```bash
modal run scripts/morpheus_modal.py::inspect_run \
  --run-id <run_id>
```

## Specification gaps

Optimizer, learning rate, schedule, batch size, replay-window policy, class
balance, checkpoint cadence, and stopping rule are not specified. Parts 12 and
13 must provide explicit measured values before this part starts.

The spec uses `promotion` for curriculum movement and arena checkpoint
acceptance. The run manifest must use distinct names for those events.

## Exit criterion

Answer `yes` if interruption and resume are deterministic, every checkpoint is
immutable and self-describing, A100 accounting remains within the Part 13
schedule, and one candidate completes deployment-matched self-play and
calibration.

Answer `no` for state loss, mutable checkpoint names, schema drift, untracked
compute, an unresolved curriculum or panel gate, or a candidate that skips
deployment calibration.
