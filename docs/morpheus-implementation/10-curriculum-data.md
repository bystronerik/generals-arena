# Part 10: Curriculum data

## Deliverable

Build curriculum items as an engine version, map seed, source label, and legal
joint-action prefix. Replay each item from turn 0 to reconstruct both seats'
observations, memory, belief, and action history.

Classify reachable items into the five curriculum classes without adding fields
to `GameRecord`.

**Touches**

- `bots/`: import the runtime memory and belief contracts for reconstruction.
- `arena/`: use trajectory read, era guard, and replay APIs without changing the
  game-record schema.
- `scripts/`: add curriculum record, classify, and verify commands.
- `data/bot_versions/`: existing source bots register through normal matches.
- `training/morpheus/`: add curriculum manifests and samplers.

## Prerequisites

- [Part 02: Transition kernel](02-transition-kernel.md)
- [Part 03: Observation and actions](03-observation-actions.md)
- [Part 05: Belief filter](05-belief-filter.md)
- [Part 00c: Measurement corpus](00c-measurement-corpus.md)

## Source specifications

- [Sparse-reward curriculum](../bots/morpheus/training.md#sparse-reward-solution)
- [Executable curriculum rules](../bots/morpheus/curriculum.md)
- [Curriculum promotion open question](../bots/morpheus/open-questions.md#curriculum-promotion)
- [Arena trajectories](../arena/trajectories.md)

## Defaults and replacement measurement

Use the current promotion default: do not move sampling weight toward an
earlier class until every active class has a non-degenerate WDL target.

Replace this rule only from WDL intervals by class, full-start decisive rate,
and held-out arena strength. Training loss alone cannot replace it.

This part owns a bounded curriculum pilot. Before it exits, the pilot must write
an executable confidence rule with interval method, sample requirements, and
thresholds to the curriculum manifest. The values must be selected before the
main run and linked to the WDL, decisive-rate, and held-out evidence.

## Implementation boundary

Initial classes 1–4 use legal trajectories from the exact panel in
`scripts/configs/morpheus/bootstrap-panel.json`. Class 5 uses new competition
maps. Source labels remain on every item.

Use `arena.records.trajectories.replay_states` and the era guard. Verify the
recorded state digests before extracting prefixes. Reconstruct fog observations
for each seat from engine state.

ResBot replays are never curriculum items, policy labels, value labels, or
hidden-state labels.

## Isolated test

```bash
python -m pytest training/morpheus/tests/test_curriculum.py \
  -m morpheus -q
```

```bash
python scripts/morpheus_curriculum.py build \
  --panel scripts/configs/morpheus/bootstrap-panel.json \
  --trajectories data/trajectories/morpheus-bootstrap \
  --output training/morpheus/manifests/curriculum.json
```

```bash
python scripts/morpheus_curriculum.py verify \
  --manifest training/morpheus/manifests/curriculum.json
```

The verify command must reproduce every available trajectory digest and both
seat observations at each sampled prefix.

## Specification gaps

Resolved in [curriculum.md](../bots/morpheus/curriculum.md):

- contact, sight, tactical sequence, exclusive class assignment;
- pre-contact BFS distance between generals;
- belief RNG seed from immutable item fields;
- pilot Wilson confidence rule written into the manifest.

## Exit criterion

Answer `yes` if every item replays in the matching engine era, every sampled
prefix reproduces engine and fog state for both seats, class and source labels
are explicit, the manifest contains an executable non-degenerate-WDL
confidence rule, and no ResBot replay enters the manifest.

Answer `no` for an invented state, replay mismatch, missing source, cross-era
item, or undefined class assignment.
