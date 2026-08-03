# boom

`bots/boom/`. Grounded idea: **fast-expand economy — refuse to fight until
the economy has won, then cash it in with one fist.** Migrated from the
generals-bot repo as a grid-native rewrite. Baselines for comparison:
`expand_plus`, `castle_builder`, `smoke`.

## Strategy

1. **Defend / evict**: a measured home guard (home power vs decayed threat)
   locks the bank when short; raiders inside our half are retaken first.
2. **Free captures**: frontier stacks onto neutral land, never skipped —
   land bonuses compound.
3. **Attack**: an endgame latch (army ratio ≥ 1.5, opponent collapse, or
   land exhausted; forced at turn 950) collects one fist sized against the
   opponent's mobile army and walks it at the estimated general.
4. **Castle builds**: the source's city purchases became castle builds —
   cheapest owned plain cell, funded by collect-walking field surplus,
   gated by `cost × 2 ≤ spare`. Never funded by resting the general.
5. **Expansion runs**: `army ≥ distance` break-even scheduling.

Spec: [`docs/research/strategies/boom.md`](../research/strategies/boom.md).

## Experiment

Provisional keep pending full seed grid.

## Verification

```bash
source .venv/bin/activate
python competition-module/competition/matchup.py \
  bots/boom/run.sh \
  bots/smoke/run.sh \
  --mode competition --seed 0
```
