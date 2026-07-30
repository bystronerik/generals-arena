# Smoke bot

Minimal stdio bot under `bots/smoke/`. Purpose: prove competition mode + protocol end-to-end. Not a competitive strategy.

## Strategy

1. Expand onto a visible neutral when a capture is possible.
2. Else take any legal move.
3. Else pass.
4. Never builds castles (`pass=2`).

## Verification command

```bash
source .venv/bin/activate
python competition-module/competition/matchup.py \
  bots/smoke/run.sh \
  competition-module/competition/agents/expander_python/run.sh \
  --mode competition --seed 0
```

## Observation (seed 0, 2026-07-31)

| Field | Value |
|-------|--------|
| Result | Draw (truncated) |
| Turns | 1200 |
| Winner | none |
| Castles built | smoke 0, expander_python 0 |
| Build mechanic | Enabled by competition mode; neither bot issued a build action |
| Deathtouch | Enabled from turn 800; no general capture before truncation |

Match finished cleanly. Stdio protocol and `GeneralsEnv(mode="competition")` work from repo root via `matchup.py`.
