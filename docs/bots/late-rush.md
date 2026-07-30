# late_rush

`bots/late_rush/`. Grounded idea: **timed commitment**. Baseline for comparison: `smoke`, `expand_plus`.

## Strategy

1. Expand quickly during turns 1–449; keep a small home reserve on the
   own general.
2. From turn 450, select a rally cell on the enemy-facing frontier and
   move supporting stacks toward it.
3. From turn 600, designate the largest eligible stack as the main stack;
   stop low-value expansion from that stack.
4. Commit between turns 650 and 700 (earlier when the enemy general is
   known) and route the main stack irreversibly toward the selected target.
5. From turn 800, prioritize any legal move onto the known enemy general
   for deathtouch contact; probe frontier or remembered enemy routes when
   the general is unknown.
6. Never builds castles (`pass=2`).

Spec: [`docs/research/strategies/late_rush.md`](../research/strategies/late_rush.md).

## Experiment

[`docs/research/experiments/007-late-rush-timed-commitment.md`](../research/experiments/007-late-rush-timed-commitment.md).

## Verification

```bash
source .venv/bin/activate
python competition-module/competition/matchup.py \
  bots/late_rush/run.sh \
  bots/smoke/run.sh \
  --mode competition --seed 0
```

Observed (seed 0 vs `smoke`): late_rush (player 0) captured smoke's general on turn 685. Match finished cleanly, no faults.
