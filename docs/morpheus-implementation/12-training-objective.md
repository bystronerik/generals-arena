# Part 12: Training objective

## Deliverable

Implement terminal WDL, root-average-strategy policy targets, hidden-state
auxiliary targets, final-margin targets, termination targets, and joint board
symmetry augmentation.

Add a run configuration that makes every loss weight and exploration setting
explicit.

**Touches**

- `bots/`: use the network and schema definitions; do not add training loops to
  the bot closure.
- `arena/`: none.
- `scripts/`: add objective-ablation entry points.
- `data/bot_versions/`: none.
- `training/morpheus/`: add targets, losses, augmentation, and configuration.

## Prerequisites

- [Part 03: Observation and actions](03-observation-actions.md)
- [Part 04: Network and export](04-network-export.md)
- [Part 10: Curriculum data](10-curriculum-data.md)
- [Part 11: Self-play league](11-self-play-league.md)

## Source specifications

- [Training targets](../bots/morpheus/training.md#targets)
- [Search noise and exploration](../bots/morpheus/training.md#search-noise-and-exploration)
- [Training losses open question](../bots/morpheus/open-questions.md#training-losses-and-exploration)

## Defaults and replacement measurement

There are no current values for auxiliary-loss weights, root noise, action
temperature, or deterministic turn.

The implementation must fail closed when a training run omits them. Rated play
always disables root noise and action temperature.

Select values only through controlled ablations for belief calibration, policy
entropy, action coverage, cycling, and held-out arena strength.

## Implementation boundary

Use only terminal reward:

```text
win = +1
draw = 0
loss = -1
discount = 1
```

Land, army, castles, sight, and game length are labels or diagnostics, never
reward terms.

Policy targets come from normalized root average strategy. Value targets use
final WDL from the sample seat. Hidden-state labels use engine truth.
Enemy-army labels use the versioned bin edges from Part 04.

Apply rotations and reflections to tensor, legal mask, action channels, memory,
belief planes, generals, and spatial targets as one transform.

## Isolated test

```bash
python -m pytest \
  training/morpheus/tests/test_targets.py \
  training/morpheus/tests/test_losses.py \
  training/morpheus/tests/test_augmentation.py \
  -m morpheus -q
```

```bash
modal run scripts/morpheus_modal.py::ablate_objective \
  --config training/morpheus/configs/objective-ablation.json
```

The ablation writes
`docs/research/measurements/morpheus-objective-ablation.{json,md}` and charges
its A100 time to Part 13.

## Specification gaps

The specs do not define loss functions for the scalar heads, class balancing,
optimizer, learning-rate schedule, batch size, regularization, or the candidate
grid for exploration.

The main run cannot begin until the ablation records explicit values. This is a
measurement gate, not permission to add hidden defaults.

## Exit criterion

Answer `yes` if hand-computed target and loss fixtures match, every symmetry
round-trips, reward contains no shaping term, omitted values fail closed, and
the selected run manifest cites the ablation evidence.

Answer `no` for an implicit weight, a nonterminal reward, a symmetry mismatch,
or exploration enabled in rated mode.
