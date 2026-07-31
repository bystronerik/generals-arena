# proteus

`bots/proteus/`. Grounded idea: **classify the opponent, switch between
strategy cores with hysteresis.** Composes the cores of `blitz` and `boom`
around one shared opponent model. Migrated from the generals-bot repo
(source name: adaptive). Baselines for comparison: the two pure cores and
`phase_switch`.

## Strategy

1. **Classify** every turn: aggressor / economy / unknown, from the exact
   aggregates plus a home-pressure latch.
2. **Counter map**: blitz against an opponent that has walked a sized stack
   at our general; boom against everything else; blitz as the from-turn-0
   spine.
3. **Hysteresis**: leaving the spine needs a long streak at confidence;
   returning to it needs six turns and ignores the cooldown. The asymmetry
   is priced from the core grid — being boom against a real aggressor is the
   worst cell (0.36), being blitz against an economy costs at most 0.16.
4. **Warm handover**: the inactive core observes every turn, so a mid-game
   switch starts with current beliefs and threat memories.

Two cores, not four: `metro` and `aegis` were constructed and warmed by
every previous version behind a counter map that could never select either,
and the core grid says that was right — both are dominated and uniquely best
against nothing. See the spec §2.

Telemetry reports the active core, the label, the switch count, and the
evidence behind the aggressor branch (`stack_near`, `turns_near`,
`duel_turn`) plus the inferred opponent structure count.

Spec: [`docs/research/strategies/proteus.md`](../research/strategies/proteus.md).

## Experiment

[`docs/research/experiments/016-proteus-adaptive-switching.md`](../research/experiments/016-proteus-adaptive-switching.md)

## Verification

```bash
source .venv/bin/activate
python competition-module/competition/matchup.py \
  bots/proteus/run.sh \
  bots/smoke/run.sh \
  --mode competition --seed 0
```
