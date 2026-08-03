# blitz

`bots/blitz/`. Grounded idea: **general rush — one big strike stack,
launched early and repeatedly at the enemy general.** Migrated from the
generals-bot repo as a grid-native rewrite. Baselines for comparison:
`late_rush`, `army_convey`, `expand_plus`, `smoke`.

## Strategy

1. **Finish**: capture the enemy general the moment any cell of ours can
   (from turn 800, any 2-army neighbour).
2. **Defend**: only when the general is genuinely losable per a pessimistic
   threat model (one reinforcing stack counts, not a naive sum).
3. **Opening** (turn < 50): chain expansion biased toward the enemy anchor —
   the chains double as the strike corridor.
4. **Assault**: wave cycle rebuild → rally → push. Waves are sized against
   the opponent's *mobile* army (`opp_army − opp_land`) and pathed with a
   Dijkstra potential field that discounts own cells and surcharges
   defended enemy cells.
5. **Expand**: leftover moves grow land; a whiffed rush must not idle.

No castle builds — blitz spends every army on tempo.

Spec: [`docs/research/strategies/blitz.md`](../research/strategies/blitz.md).

## Experiment

Provisional keep pending full seed grid.

## Verification

```bash
source .venv/bin/activate
python competition-module/competition/matchup.py \
  bots/blitz/run.sh \
  bots/smoke/run.sh \
  --mode competition --seed 0
```
