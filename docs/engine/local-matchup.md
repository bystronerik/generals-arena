# Local matchup

Run two stdio bots under the competition ruleset.

## Command

From the repo root:

```bash
python competition-module/competition/matchup.py \
  path/to/bot_a/run.sh \
  path/to/bot_b/run.sh \
  --mode competition --seed 0
```

Defaults (no agent paths): two copies of `competition-module/competition/agents/expander_python/run.sh`.

## Required flag

Always pass `--mode competition` for graded-style local matches. That pins `GeneralsEnv(mode="competition")` (build castles, deathtouch, map sizes, fog, 1200-turn cap).

## Verification gate

A change is not verified until a match **finishes** under `--mode competition` (win, loss, or draw / truncation).

## Optional

- `--gui` — human inspection (not required for CI).
- `build.sh` next to `run.sh` runs once before the match if present.

## Protocol

See [`../competition/protocol.md`](../competition/protocol.md).
