---
name: run-measurement-round
description: >-
  Runs the heuristic measurement grid through scripts/measure_heuristics.py
  with a parallel competition worker pool, stores games under
  data/games/<round>/, refits ratings once, and writes
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
seeds per pair. **Seat is drawn per game** from the seeded stream, so the round
does not need to double its games to control for seat order. Games land in
`data/games/<round>/`.

Prefer `arena/matches/run_match.run_and_store` when adding single matches
outside the grid script.

Flags:

- `--games-per-pair` — random seeds per pair (default: 50)
- `--round-seed` — RNG seed for map-seed generation
- `--jobs` — workers (default: physical cores; capped)
- `--bots` — override roster bot ids
- `--seeds` — fixed seed list (overrides random generation)
- `--wait` — poll until every bot `run.sh` exists
- `--seat-policy` — `random` (default) or `alternate` (exact 50/50, matched map seeds; use it for decision arms)
- `--no-ratings` — store games without a global refit
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
- whether stored games are schema v4 (`bot_a_content_hash`, `engine_version`, `round`)

The round report's own rating table is **round-local**: fitted over this round's
games only, anchored on the round's most-played bot, keyed on `bot_id` rather
than the content hash. It is not comparable to `data/ratings/leaderboard.md`.
Report the global fit for anything that matters.

## Rules

- Never rate from stdout. Store first, then refit (see **update-leaderboard**).
- Never quote a round-report rating as a decision. Thresholds and the contrast
  live in [`docs/arena/decision-rule.md`](../../../docs/arena/decision-rule.md).
- Always competition mode (enforced by the script).
- Do not write a new runner — use the script.

Authority: [`docs/research/strategies/skills-workflow.md`](../../../docs/research/strategies/skills-workflow.md).

## Changelog

- 2026-07-31 — Per-game seat draw, `--seat-policy`, batch refit, round-local rating caveat
- 2026-07-31 — Rule C parallel pool + per-round game folders
- 2026-07-31 — Initial skill from skills-workflow taxonomy (cause: skills-workflow build)
