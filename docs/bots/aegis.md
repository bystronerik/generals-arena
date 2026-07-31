# aegis

`bots/aegis/`. Grounded idea: **turtle + counterattack — hold a compact
ring, destroy incoming attacks, then commit into the window their spent
attack opens.** Migrated from the generals-bot repo as a grid-native
rewrite. Baselines for comparison: `garrison`, `choke_control`, `blitz`,
`smoke`.

## Strategy

1. **The keep**: the reserve lives one cell off the general — it can step
   back onto the door in one move, and the same army hunts raiders and
   leads counters.
2. **Measured defence**: garrison sized against the largest stack they
   have shown or a share of their mobile army, never more than half our
   total; active interception, general sorties, and leashed hunts.
3. **Counterattack**: event-driven — their total-army collapse or a dead
   stack opens a committed 80-turn window; late-game parity commits too
   (the draw guard).
4. **Compact growth**: perimeter claims that plug holes and hug walls;
   border raids (enemy castles first); short marches.
5. **Safe castle builds**: the source's city grabs became home castle
   builds within 10 cells of the general, funded once the garrison is
   covered — compounding production while turtling.

Spec: [`docs/research/strategies/aegis.md`](../research/strategies/aegis.md).

## Experiment

[`docs/research/experiments/015-aegis-turtle-counter.md`](../research/experiments/015-aegis-turtle-counter.md)
— provisional keep pending full seed grid.

## Verification

```bash
source .venv/bin/activate
python competition-module/competition/matchup.py \
  bots/aegis/run.sh \
  bots/smoke/run.sh \
  --mode competition --seed 0
```
