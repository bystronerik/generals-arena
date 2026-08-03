# splitter

`bots/splitter/`. Grounded idea: **split moves**. Baseline for comparison: `expand_plus`, `smoke`.

## Strategy

1. Greedy capture: same scoring as `expand_plus`, but evaluate both
   `split=0` (move all-but-one) and `split=1` (move half army) for each
   capturable adjacent tile.
2. Use `split=1` when half the army still captures with margin and at
   least one of: a second capturable direction from the source (multi-front),
   the source borders opponent/fog (garrison), or the stack is oversized
   for a cheap neutral target.
3. Never split off the general tile; opponent captures always commit full
   army. Tie-break exact score ties to `split=0`.
4. When no capture is available, BFS frontier march and any-valid-move
   fallback — unchanged from `expand_plus` (`split=0` only).

## Verification

```bash
source .venv/bin/activate
python competition-module/competition/matchup.py \
  bots/splitter/run.sh \
  bots/smoke/run.sh \
  --mode competition --seed 0
```
