# general_hunter

`bots/general_hunter/`. Grounded idea: **late-game hunt**. Baseline for comparison: `expander_python`, `smoke`.

## Strategy

1. Expand greedily, same scoring as `expander_python`, for the whole
   early/mid game.
2. Whenever the enemy general is ever glimpsed (`type==4`, `owner==2`),
   remember its cell forever — generals never move, so one sighting stays
   exact even after it fades back into fog.
3. From turn 800 (deathtouch — see
   [`docs/competition/deathtouch.md`](../competition/deathtouch.md)), if
   the enemy general's cell is known:
   - If already adjacent with army ≥ 2, execute onto it immediately —
     this wins outright regardless of the defender's army.
   - Otherwise, BFS from the target and advance whichever owned cell
     (army ≥ 2) is currently closest one step further along the shortest
     path.

## Experiment

[`docs/research/experiments/003-general-hunter-deathtouch-beeline.md`](../research/experiments/003-general-hunter-deathtouch-beeline.md) — kept, but unproven in full games: the beeline/execute logic was checked directly against a synthetic observation and behaves correctly, but no game in the seed grid ever sighted the opposing general before truncation (none of the current opponents probe past the shared frontier).

## Verification

```bash
source .venv/bin/activate
python competition-module/competition/matchup.py \
  bots/general_hunter/run.sh \
  competition-module/competition/agents/expander_python/run.sh \
  --mode competition --seed 0
```

Observed (seeds 0–2 vs `smoke` and `expander_python`): all draws, 1200 turns, no faults, no sighting of the enemy general in this grid.
