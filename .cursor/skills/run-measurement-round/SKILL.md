---
name: run-measurement-round
description: >-
  Runs the heuristic measurement grid through scripts/measure_heuristics.py
  with a parallel competition worker pool, stores games under
  data/games/<round>/, rebuilds ratings once, and writes
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
python scripts/measure_heuristics.py \
  --round round<N> \
  --games-per-pair 50 \
  --round-seed <int>
```

Default grid is Rule C: unordered pairs among `DEFAULT_ROSTER`, random map
seeds per pair, no seat swap. Games land in `data/games/<round>/`.

Prefer `arena/matches/run_match.run_and_store` (schema v2 with bot telemetry) when
adding single matches outside the grid script.

Flags:

- `--games-per-pair` — random seeds per pair (default: 50)
- `--round-seed` — RNG seed for map-seed generation
- `--jobs` — workers (default: physical cores; capped)
- `--bots` — override roster bot ids
- `--seeds` — fixed seed list (overrides random generation)
- `--wait` — poll until every bot `run.sh` exists
- `--no-ratings` — store games without a global Elo rebuild
- `--legacy-grid` — old tagged seat-swap grid (sequential)

## Grid

The `DEFAULT_ROSTER` / `NEW_BOTS` / `BASELINE_BOTS` lists in
`scripts/measure_heuristics.py` define the default roster. Pass `--bots` to
override.

Create [`docs/research/measurements/`](../../../docs/research/measurements/) when the directory is absent.

## After the run

Read `docs/research/measurements/round<N>.md` and report:

- draw rate
- mean turns
- decisive games
- any bot with zero decisive games
- whether stored games use schema v2 telemetry (`final_land_a`, sighting fields)

## Rules

- Never update ratings from stdout. Store first, then rate (see **update-leaderboard**).
- Always competition mode (enforced by the script).
- Do not write a new runner — use the script.

Authority: [`docs/research/strategies/skills-workflow.md`](../../../docs/research/strategies/skills-workflow.md).

## Changelog

- 2026-07-31 — Rule C parallel pool + per-round game folders
- 2026-07-31 — Initial skill from skills-workflow taxonomy (cause: skills-workflow build)
