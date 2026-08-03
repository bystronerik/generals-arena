# Part 16: Arena promotion

## Deliverable

Measure a frozen candidate and the current frozen checkpoint in the same
competition rounds, then return the repository's pairwise rating verdict.

The report also covers protocol safety, reply latency, completed simulations,
belief ESS and recovery, outcome type, board size, seat, and behavioral
distance.

**Touches**

- `bots/`: create a temporary frozen baseline copy for the matched comparison.
- `arena/`: use tournament, store, registry, ratings, and submission checks.
- `scripts/`: add Morpheus panel and report orchestration.
- `data/bot_versions/`: register candidate and temporary baseline through the
  tournament parent.

## Prerequisites

- [Part 15: Artifact freeze](15-artifact-freeze.md)

## Source specifications

- [Evaluation and artifact identity](../bots/morpheus/evaluation.md)
- [Diversity](../bots/morpheus/diversity.md)
- [Decision rule](../arena/decision-rule.md)
- [Experiment protocol](../research/experiment-protocol.md)

## Defaults and replacement measurement

The ResBot distance-7 castle result remains diagnostic-only.

Replace that default only with held-out action reconstruction that proves the
spend cell and build turn. The result still cannot specify ResBot internals and
never enters arena ratings.

No fixed promotion panel or held-out seed split exists. This part must record
both before games start. The panel must contain at least five bots, span the
rating range, include `cm_expander`, include strong heuristic and research
bots, and include the prior Morpheus checkpoint when one exists.

## Implementation boundary

Re-run both arms in the same rounds with identical panel and map seeds and
`--seat-policy alternate`. Store every game before refitting the whole pool.

Use a temporary frozen baseline bot directory, as required by the existing
decision workflow. Remove it only after the report and registry evidence are
complete.

Read the decision from:

```python
delta = fit.delta(baseline_entity, candidate_entity)
```

Report `delta.value`, `delta.se`, `delta.ci`, `delta.p_stronger`, games per arm,
decisive games per arm, and `games_to_resolve` when unproven. Never use
leaderboard rank.

Safety checks can reject a candidate. They cannot promote a weaker candidate.
Part 08 must accept the exact candidate bundle with zero faults.

## Isolated test and measurement

```bash
python scripts/morpheus_evaluate.py run \
  --config training/morpheus/configs/promotion-panel.json \
  --baseline bots/morpheus_base/run.sh \
  --candidate bots/morpheus/run.sh \
  --round morpheus-promotion \
  --round-seed 7 \
  --seat-policy alternate \
  --strict-versions
```

The orchestration must satisfy the minimum games, per-opponent games, decisive
games, connectivity, and same-era gates in the decision rule.

```bash
python -m arena.records.ratings --print --lineage morpheus
```

```bash
python scripts/morpheus_evaluate.py contrast \
  --baseline <baseline_entity> \
  --candidate <candidate_entity>
```

```bash
python scripts/morpheus_measure.py resbot-castle-reconstruction \
  --replays competition-replays/ResBot \
  --output docs/research/measurements/morpheus-resbot-castles.json
```

The ResBot command is optional for promotion while the diagnostic default
holds.

## Specification gaps

The exact panel, held-out seed rule, acceptable belief ESS, recovery threshold,
and completed-simulation threshold are not defined. Parts 09 and 13 must record
safety floors before the outcomes are inspected.

There is no first-checkpoint promotion rule because the decision rule requires
a non-provisional baseline. The first accepted artifact can enter the arena as
a provisional bootstrap baseline, but it cannot be called an improvement.
Normal promotion starts with a later candidate unless the specification adds a
separate bootstrap rule.

The repository's diversity matrix does not yet contain Morpheus. Promotion must
add its measured row without changing the strategy specification.

## Exit criterion

Answer `yes` only if all decision-rule gates pass, the pairwise verdict is
`improvement`, every safety check passes, action traces are not more than 90%
identical to a roster owner on shared seeds, and the five-axis verdict remains
distinct.

Answer `no` for a regression or safety failure. Answer `unproven` when the
decision rule cannot resolve the contrast, and report the additional games
required.
