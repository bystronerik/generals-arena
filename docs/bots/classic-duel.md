# classic_duel

`bots/classic_duel/`. Remote-only champion for **classic generals.io**. This bot
never enters competition Elo or `data/games/`.

## Strategy

Three pillars (see spec):

1. Expansion engine — interior-to-frontier logistics (`army_convey`-style).
2. City policy — capture neutral pre-placed cities on classic boards.
3. Kill conversion and general defense — finish with a stack strictly larger
   than the enemy general; hold a reserve at home.

Hard classic-safe rules: no build (`pass=2`), no deathtouch branch, no turn-800
or turn-1200 clocks.

Spec: [`docs/research/strategies/classic_duel.md`](../research/strategies/classic_duel.md).

Plan: [`docs/research/strategies/human-95-plan.md`](../research/strategies/human-95-plan.md).

## Verification

Competition gate (still required before any remote block):

```bash
source .venv/bin/activate
python competition-module/competition/matchup.py \
  bots/classic_duel/run.sh \
  bots/smoke/run.sh \
  --mode competition --seed 0
```

Classic harness (proxy for remote tuning):

```bash
python -m arena.classic_match \
  bots/classic_duel/run.sh \
  bots/army_convey/run.sh \
  --seed 0
```

Remote dry-run (no network):

```bash
python scripts/remote_play.py --bot classic_duel --mode dry-run
```

Remote results log to `data/remote_games/` and never update `data/ratings/`.
