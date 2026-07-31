---
name: evaluate-bot-change
description: >-
  Measures a bot change with a fixed seed grid or Rule C games-per-pair round,
  before and after winrate, and an Elo delta from stored games. Use when
  comparing bot versions, running an A/B tournament, judging a parameter
  revision, or deciding keep versus revert from data/games metrics.
---

# Evaluate bot change

Follow [`docs/research/experiment-protocol.md`](../../../docs/research/experiment-protocol.md). One **parameter group** per experiment when possible.

## Model split

- Think model: writes the hypothesis, chooses the seed grid and opponents, interprets results
- Composer: runs matches or the measurement script, stores games, reports metrics

**Composer must not invent a threshold.** When a value is absent from the specification, Composer stops and asks the think model.

## Workflow

1. **Hypothesis** — one claim; optional short note under `docs/research/experiments/`.
2. **Baseline** — fix opponent set + seed list. Run before the change (or use stored games that match that grid).
3. **Treat** — apply the change; re-run the same seeds and opponents.
4. **Store** — every game via `arena/matches/run_match.run_and_store`,
   `arena/tournament.py`, or `scripts/measure_heuristics.py` into
   `data/games/<round>/` before ratings. Prefer `run_and_store` for single
   schema v2 matches.
5. **Report** — winrate, draw rate, mean turns, decisive games, Elo delta, sample size.
6. **Decide** — keep or revert from stored metrics only.

## Commands

```bash
source .venv/bin/activate   # if present; prefer python3.12

# Single stored match (schema v2 telemetry)
python -m arena.matches.run_match bots/<a>/run.sh bots/<b>/run.sh --mode competition --seed <n>

# Rule C heuristic round (parallel pool + round report)
python scripts/measure_heuristics.py \
  --round round<N> --games-per-pair 50 --round-seed <int>
```

Round reports: [`docs/research/measurements/`](../../../docs/research/measurements/).
Override roster with `--bots`; default is `DEFAULT_ROSTER` in the script.

## Seat-order swap

For small fixed-seed A/B claims, include both seat orders. Large Rule C rounds
with random `--games-per-pair` seeds skip seat swap by default.

## Draw-heavy decision rule

When both arms draw every game, mark the result **unproven**, not neutral. Ask for a decisive opponent or a different grid before changing thresholds.

## Metrics checklist

- [ ] Same `--round-seed` / `--seeds` and `--games-per-pair` before and after
- [ ] Same opponents
- [ ] Both seat orders when using a small fixed-seed grid
- [ ] Competition mode on every match
- [ ] Games under `data/games/<round>/` before Elo update
- [ ] One changed parameter group per experiment
- [ ] Rating snapshot / delta via `arena/records/ratings.py` (see **update-leaderboard**)

Schema: [`docs/arena/game-record-schema.md`](../../../docs/arena/game-record-schema.md).

## Rules

- Do not merge strategy changes without a measured delta in `data/`.
- No strategy content in this skill — link `docs/` for domain knowledge.

## Changelog

- 2026-07-31 — Rule C parallel rounds + per-round folders
- 2026-07-31 — Seat-order swap, draw rule, measure_heuristics pointer (cause: skills-workflow build)
