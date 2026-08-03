# metro

`bots/metro/`. Grounded idea: **castle network + pressure — army spent on
land is spent once; army spent on production keeps paying.** Migrated from
the generals-bot repo as a grid-native redesign (the source captured
neutral cities; the arena builds castles). Baselines for comparison:
`castle_builder`, `castle_rush`, `phase_switch`, `expand_plus`.

## Strategy

1. **Castle programme** (turn ≥ 60, up to 3 castles): score owned plain
   cells by build cost + distance from the enemy anchor, collection-walk
   the main stack onto the site, build when funded. Never funds builds by
   resting the general.
2. **Enemy-castle cash-in**: capturing a visible enemy castle is a double
   production swing and outranks expansion.
3. **Garrisons**: front-line castles are locked bastions; rich castles are
   war chests reserved from expansion.
4. **Pressure**: from turn 140 there is always a wave in the field, sized
   against the biggest enemy stack, with takeover hysteresis so home
   castles cannot hijack an advance.
5. **Defence**: pessimistic garrison-vs-biggest-stack test; kill adjacent
   threats where they stand.

Spec: [`docs/research/strategies/metro.md`](../research/strategies/metro.md).

## Experiment

Provisional keep pending full seed grid.

## Verification

```bash
source .venv/bin/activate
python competition-module/competition/matchup.py \
  bots/metro/run.sh \
  bots/smoke/run.sh \
  --mode competition --seed 0
```
