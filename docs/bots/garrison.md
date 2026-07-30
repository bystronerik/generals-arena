# garrison

`bots/garrison/`. Grounded idea: **defensive reserves**. Baseline for comparison: `smoke`, `expand_plus`.

## Strategy

1. Locate the own general once and keep a phase-based army floor on it
   (8 → 16 → 24 → 12 after turn 800).
2. Build passable and own-territory distance fields from the general each
   turn.
3. Estimate visible opponent pressure near the general; compute a defense
   deficit and reinforce one own step toward the general when needed.
4. Expand only when no urgent reinforcement or emergency defense move
   exists; reject captures that break the reserve or open a route to the
   general.
5. From turn 800, treat deathtouch routes as critical: intercept adjacent
   runners, chase source cells, and keep movable stacks on distance 1–2
   cells around the general.
6. Never builds castles (`pass=2`).

Spec: [`docs/research/strategies/garrison.md`](../research/strategies/garrison.md).

## Experiment

[`docs/research/experiments/006-garrison-defensive-reserves.md`](../research/experiments/006-garrison-defensive-reserves.md).

## Verification

```bash
source .venv/bin/activate
python competition-module/competition/matchup.py \
  bots/garrison/run.sh \
  bots/smoke/run.sh \
  --mode competition --seed 0
```

Observed (seed 0 vs `smoke`): draw, 1200 turns, no faults.
