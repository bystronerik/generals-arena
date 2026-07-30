# castle_rush

`bots/castle_rush/`. Grounded idea: **aggressive early castle building**. Baselines for comparison: `castle_builder`, `expand_plus`, `smoke`.

## Strategy

1. **Setup** (turn < 10): pure `expand_plus` expansion. No resting, no building.
2. **Rush** (turn >= 10 and own castles < 4): rest the general and build at the cheapest reachable cell with tighter parameters than `castle_builder` (start 10, cap 4, cooldown 25, min land 5, surplus margin 5).
3. **Sustain** (own castles >= 4): full `expand_plus` expansion; existing castles keep producing.

Enemy-general sighting is recorded every tick but not acted on (economy bot, not a hunter).

## Experiment

[`docs/research/experiments/011-castle-rush-aggressive-builds.md`](../research/experiments/011-castle-rush-aggressive-builds.md) — provisional keep pending full seed grid.

## Verification

```bash
source .venv/bin/activate
python competition-module/competition/matchup.py \
  bots/castle_rush/run.sh \
  bots/smoke/run.sh \
  --mode competition --seed 0
```
