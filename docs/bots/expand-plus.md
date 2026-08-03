# expand_plus

`bots/expand_plus/`. Grounded idea: **expand**. Baseline for comparison: `expander_python`, `smoke`.

## Strategy

1. Greedy capture: same scoring as `expander_python` — pick the move with
   the highest `army * (10 if expansion) * (2 if opponent)` among all
   capturable adjacent tiles.
2. When no capture is available this turn, march instead of wandering:
   run a multi-source BFS seeded from every visible capturable tile
   (neutral or opponent-owned), then move the largest owned stack one step
   toward the nearest one. This replaces the "first legal move" fallback
   used by `smoke` / `expander_python`.
3. Never builds castles (`pass=2`).

## Experiment

Kept; full-game telemetry only confirms no regression (still finishes clean, no faults), since land/army are not yet in the game-record schema.

## Verification

```bash
source .venv/bin/activate
python competition-module/competition/matchup.py \
  bots/expand_plus/run.sh \
  competition-module/competition/agents/expander_python/run.sh \
  --mode competition --seed 0
```

Observed (seeds 0–2 vs `smoke` and `expander_python`): all draws, 1200 turns, no faults.
