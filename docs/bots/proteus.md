# proteus

`bots/proteus/`. Grounded idea: **classify the opponent, switch between
the four migrated strategies with hysteresis.** Composes the cores of
`blitz`, `boom`, `metro`, and `aegis` around one shared opponent model.
Migrated from the generals-bot repo (source name: adaptive). Baselines for
comparison: the four pure cores and `phase_switch`.

## Strategy

1. **Classify** every turn: rusher / boomer / citier (castle-builder) /
   turtler / unknown, from exact aggregates + vision signals.
2. **Counter map**: blitz vs rushers, boomers, and castle-builders; boom
   vs passive turtlers; blitz as the from-turn-0 default spine.
3. **Hysteresis**: 12-turn streak at confidence ≥ 0.45 to switch, 50-turn
   cooldown; switching *into* the aegis defensive posture is fast (5-turn
   streak, no cooldown), leaving it is slow (60-turn streak).
4. **Warm handover**: inactive cores observe every turn, so a mid-game
   switch starts with current beliefs and threat memories.

Telemetry reports the active core, opponent label, and switch count.

Spec: [`docs/research/strategies/proteus.md`](../research/strategies/proteus.md).

## Experiment

[`docs/research/experiments/016-proteus-adaptive-switching.md`](../research/experiments/016-proteus-adaptive-switching.md)
— provisional keep pending full seed grid.

## Verification

```bash
source .venv/bin/activate
python competition-module/competition/matchup.py \
  bots/proteus/run.sh \
  bots/smoke/run.sh \
  --mode competition --seed 0
```
