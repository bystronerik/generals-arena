# fog_scout

`bots/fog_scout/`. Grounded idea: **fog probing**. Baseline for comparison: `smoke`, `expand_plus`.

## Strategy

1. Attack a visible enemy general immediately when capture is possible.
2. Expand onto fog, opponent, or visible neutral cells; score =
   `army * 10 * (3 if fog/unseen) * (2 if opponent)`.
3. When no adjacent capture exists, BFS-march the largest owned stack one
   step toward the nearest never-seen passable cell.
4. Fallback: first legal move or pass.
5. Never builds castles (`pass=2`).

Spec: [`docs/research/strategies/fog_scout.md`](../research/strategies/fog_scout.md).

## Experiment

[`docs/research/experiments/004-fog-scout-systematic-probing.md`](../research/experiments/004-fog-scout-systematic-probing.md).

## Verification

```bash
source .venv/bin/activate
python competition-module/competition/matchup.py \
  bots/fog_scout/run.sh \
  bots/smoke/run.sh \
  --mode competition --seed 0
```
