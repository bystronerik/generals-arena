# castle_builder

`bots/castle_builder/`. Grounded idea: **castle-aware**. Baseline for comparison: `expander_python`, `smoke`.

## Strategy

1. Expand greedily, same scoring as `expander_python`.
2. Once land reaches 8, stop spending the **general's own** army on
   captures (every other owned cell keeps expanding) so the general's
   stack idles and grows via normal army generation.
3. Once the rested general can afford the cheapest reachable neighbor
   cell (`cost = 35 + Σ max(0, 14 − 2·manhattan_dist)` over own
   structures — see [`docs/competition/build-castles.md`](../competition/build-castles.md)),
   relocate the whole stack there in one hop.
4. Build (`pass=2`) on that cell the next turn once it can afford it with
   a margin. Cap: 2 castles, turn window 20–900.

An earlier, simpler version tried to build only when some already-owned
cell happened to have spare army (no dedicated reserve step). Instrumented
runs showed the largest single-cell army never got within ~20–30 of the
35 base cost under pure greedy expansion — any big stack gets spent on
the next capture before it can idle. That version is rejected; see the
experiment note for detail.

## Experiment

[`docs/research/experiments/002-castle-builder-early-investment.md`](../research/experiments/002-castle-builder-early-investment.md) — kept (rested-general mechanism; it is the only version that ever builds). Economic payoff itself is unverified: the game-record schema does not yet capture final land/army.

## Verification

```bash
source .venv/bin/activate
python competition-module/competition/matchup.py \
  bots/castle_builder/run.sh \
  competition-module/competition/agents/expander_python/run.sh \
  --mode competition --seed 0
```

Observed (seeds 0–2 vs `smoke` and `expander_python`): 2 castles built every game, all draws, 1200 turns, no faults.
