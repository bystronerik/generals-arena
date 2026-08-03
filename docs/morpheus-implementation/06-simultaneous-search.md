# Part 06: Simultaneous search

## Deliverable

Implement root-sampled information-set MCTS with one simultaneous zero-sum
matrix per information node and enemy information hash.

The part includes candidate widening, regret-matching-plus selection and
backup, pending leaf batches, bounded enemy tables, child observation hashes,
and tree reuse.

**Touches**

- `bots/`: add search, matrix, hashing, tree, and tactical-test modules.
- `arena/`: none.
- `scripts/`: add the search-resource measurement command.
- `data/bot_versions/`: none.

## Prerequisites

- [Part 02: Transition kernel](02-transition-kernel.md)
- [Part 04: Network and export](04-network-export.md)
- [Part 05: Belief filter](05-belief-filter.md)

## Source specifications

- [Search](../bots/morpheus/search.md)
- [Belief search representation](../bots/morpheus/belief-state.md#search-representation)
- [Search-resource open question](../bots/morpheus/open-questions.md#search-resources)

## Defaults and replacement measurement

Start with these search defaults:

- self widening cap 16 and coefficient 2.0;
- enemy widening cap 12 and coefficient 1.5;
- depth 16;
- 8 enemy tables per node;
- 4096 tree nodes;
- exploration floor 0.05 and numerator 0.5;
- pending leaf batch up to 4.

Replace them only from tactical suites for chase, reinforcement, castle,
mutual capture, and deathtouch, plus table hit rate, eviction loss, memory,
completed simulations, and arena contrast.

## Implementation boundary

All values use the root-player perspective. Backup must not alternate signs.
Only a fully backed-up simulation can change root statistics.

Keep matrix math pure and separate from tree storage. Make deterministic
sampling injectable for tests. Candidate lists must always include pass and
the mandatory tactical actions named by the search specification.

The transition comes only from Part 02. Do not copy strategy logic from an
existing bot.

## Isolated test

```bash
python -m pytest \
  bots/morpheus/tests/test_matrix_backup.py \
  bots/morpheus/tests/test_search_tactics.py \
  bots/morpheus/tests/test_tree_reuse.py \
  -m morpheus -q
```

```bash
python scripts/morpheus_measure.py search-resources \
  --suite bots/morpheus/tests/fixtures/tactical-suite.json \
  --output docs/research/measurements/morpheus-search-resources.json
```

```bash
python competition-module/competition/matchup.py \
  bots/morpheus/run.sh bots/smoke/run.sh \
  --mode competition --seed 0
```

## Specification gaps

The specs do not define the exact observable-history hash, collision policy,
particle-reservoir replacement rule, weighted-LRU score, or eviction-loss
metric.

First-play urgency uses a node network value, but the specs do not define how
particle-dependent values become one information-set value.

The particle-weighted aggregation across enemy information hashes is described
without an exact estimator. These definitions must be recorded before code
uses them.

## Exit criterion

Answer `yes` if hand-computed matrix fixtures match, every tactical suite case
selects from the correct simultaneous matrix, tree reuse occurs only on an
exact information-state match, bounds are enforced, and the competition gate
finishes.

Answer `no` for alternating-player leakage, hidden-state keys, partial backup,
unbounded storage, or a tactical-suite mismatch.
