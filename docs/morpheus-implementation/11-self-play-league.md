# Part 11: Self-play league

## Deliverable

Implement an in-process self-play producer that runs the complete belief and
simultaneous-search stack on both seats.

The producer samples immutable checkpoint opponents and a fixed bot panel,
records policy and truth targets outside arena game records, and emits replay-
verifiable training shards.

**Touches**

- `bots/`: load the shared runtime model and decision stack.
- `arena/`: use competition engine and trajectory verification APIs; do not
  store self-play shards as rated games.
- `scripts/`: add local and Modal self-play entry points.
- `data/bot_versions/`: none for training-only checkpoints.
- `training/morpheus/`: add league, sampler, self-play driver, and shard schema.

## Prerequisites

- [Part 00: Modal JAX preflight](00-modal-jax-preflight.md)
- [Part 06: Simultaneous search](06-simultaneous-search.md)
- [Part 07: Runtime controller](07-runtime-controller.md)
- [Part 10: Curriculum data](10-curriculum-data.md)

## Source specifications

- [Opponent mixture](../bots/morpheus/training.md#opponent-mixture)
- [Training compute](../bots/morpheus/training.md#training-compute)
- [Opponent-mixture open question](../bots/morpheus/open-questions.md#opponent-mixture)

## Defaults and replacement measurement

Start with 70% checkpoint league and 30% fixed existing-bot panel.

Replace the ratio and sampling weights only from held-out exploitability,
cycling, opponent coverage, decisive rate, and pairwise arena contrast.

Use the Part 05 default of no recursive opponent particles.

## Implementation boundary

Both seats must use the same tensor, action, belief, matrix, transition, and
reward semantics. Seat and map assignment remain random.

Snapshots are immutable during an epoch. The league includes the current
learner, promoted best, recent snapshots, and measured exploiters. The fixed
panel excludes classic and remote results.

If Part 00 passes, first test fixed-shape JAX batching for states, transitions,
and compatible network batches. Keep CPU control when particles, hashes, tree
updates, or regret backup would force harmful host synchronization.

Training shards must include enough information to replay the game and verify
targets. They must not enter `data/games/` or the rating fit.

## Isolated test

```bash
python -m pytest training/morpheus/tests/test_self_play.py \
  training/morpheus/tests/test_league.py \
  -m morpheus -q
```

```bash
python scripts/morpheus_self_play.py \
  --config training/morpheus/configs/self-play-smoke.json \
  --games 2 \
  --output /tmp/morpheus-self-play
```

```bash
python scripts/morpheus_self_play.py verify \
  --shards /tmp/morpheus-self-play
```

## Specification gaps

The fixed panel is described by category but has no bot list. Exploiter
selection, recent-snapshot window, opponent coverage metric, and epoch boundary
are not defined.

The specs allow larger training search settings but provide no values. Part 13
must select them from throughput and learning evidence.

No training shard schema or retention path exists in `AGENTS.md`. The
training-package placement rule must land before these files.

## Exit criterion

Answer `yes` if a seeded batch is reproducible, both seats run the full stack,
every shard replays to the same actions and outcome, snapshot mutation is
rejected, and the sampled source proportions match the configured mixture.

Answer `no` for one-seat shortcuts, mutable opponents, perfect-information
inputs, unreplayable shards, or any write into rated game storage.
