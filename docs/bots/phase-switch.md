# phase_switch

`bots/phase_switch/`. Grounded idea: **explicit early / mid / late phases**. Baselines for comparison: `expand_plus`, `castle_builder`, `general_hunter`, `smoke`.

## Strategy

1. **Early** (turn < 80): pure `expand_plus` — greedy capture plus BFS march fallback. No castles, no resting.
2. **Mid** (80 <= turn < 800): `castle_builder` build/relocate/rest on top of `expand_plus` expansion (same cost model, cap 2 castles, margin 15, cooldown 40).
3. **Late** (turn >= 800): if the enemy general was ever sighted, `general_hunter` beeline/execute. If hunt returns no move, expand only (no build). If never sighted, mid loop continues (build + expand).

Enemy-general sighting is recorded every tick from turn 0 but not acted on until turn 800.

## Experiment

[`docs/research/experiments/010-phase-switch-gated-builds.md`](../research/experiments/010-phase-switch-gated-builds.md) — provisional keep pending full seed grid.

## Verification

```bash
source .venv/bin/activate
python competition-module/competition/matchup.py \
  bots/phase_switch/run.sh \
  bots/smoke/run.sh \
  --mode competition --seed 0
```
