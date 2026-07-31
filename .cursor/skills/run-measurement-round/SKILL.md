---
name: run-measurement-round
description: >-
  Runs the fixed heuristic measurement grid through scripts/measure_heuristics.py,
  stores games under data/games/, updates ratings, and writes
  docs/research/measurements/round<N>.json and .md. Use when measuring a batch of
  bots, starting a new round, or refreshing the round report after bots change.
---

# Run measurement round

## Model split

- Think model: nothing
- Composer: runs the script, reads the round Markdown, reports draw rate, mean turns, decisive games, and bots with zero decisive games

**Composer must not invent a threshold.** When a value is absent from the specification, Composer stops and asks the think model.

## Environment

```bash
source .venv/bin/activate   # if present; prefer python3.12
```

## Command

```bash
python scripts/measure_heuristics.py --round round<N>
```

Prefer `arena/run_match.run_and_store` (schema v2 with bot telemetry) when
adding single matches outside the grid script. Stored v2 fields include
`final_land_*`, `final_army_*`, and sighting metrics parsed from `[telemetry]`
stderr lines.

Flags:

- `--wait` — poll until every bot `run.sh` exists (default: fail fast if missing)
- `--no-ratings` — store games without a global Elo update

## Grid

The `NEW_BOTS` and `BASELINE_BOTS` lists in `scripts/measure_heuristics.py` define the grid. Edit those lists in the script; do not pass an ad-hoc grid.

Create [`docs/research/measurements/`](../../../docs/research/measurements/) when the directory is absent.

## After the run

Read `docs/research/measurements/round<N>.md` and report:

- draw rate
- mean turns
- decisive games
- any bot with zero decisive games
- whether stored games use schema v2 telemetry (`final_land_a`, sighting fields)

`scripts/measure_heuristics.py` stores schema v2 through `run_and_store`. Legacy
games under `data/games/` from before E1 may still be v1 until the grid is
re-run.

## Rules

- Never update ratings from stdout. Store first, then rate (see **update-leaderboard**).
- Always `--mode competition` (enforced by the script).
- Do not write a new runner — use the script.

Authority: [`docs/research/strategies/skills-workflow.md`](../../../docs/research/strategies/skills-workflow.md).

## Changelog

- 2026-07-31 — Initial skill from skills-workflow taxonomy (cause: skills-workflow build)
