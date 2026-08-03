# army_convey

`bots/army_convey/`. Grounded idea: **interior conveyance**. Baseline for comparison: `smoke`, `expand_plus`.

## Strategy

1. From frontier cells (owned tiles adjacent to passable unowned cells),
   execute the highest-scoring adjacent capture (`army * 10`, 2× for opponent).
2. When no frontier capture exists, convey interior stacks along a BFS
   distance field toward the nearest frontier cell
   (`score = army / (distance + 1)`).
3. Gather: move the largest interior stack adjacent to a frontier cell
   onto that frontier cell.
4. Fallback: first legal move or pass.
5. Never builds castles (`pass=2`).

Spec: [`docs/research/strategies/army_convey.md`](../research/strategies/army_convey.md).

## Verification

```bash
source .venv/bin/activate
python competition-module/competition/matchup.py \
  bots/army_convey/run.sh \
  bots/smoke/run.sh \
  --mode competition --seed 0
```
