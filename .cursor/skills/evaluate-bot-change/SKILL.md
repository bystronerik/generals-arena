---
name: evaluate-bot-change
description: >-
  Measures a bot change with a fixed seed grid, both seat orders, before and
  after winrate, and an Elo delta from stored games. Use when comparing bot
  versions, running an A/B tournament, judging a parameter revision, or deciding
  keep versus revert from data/games metrics.
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
4. **Store** — every game via `arena/run_match.py`, `arena/tournament.py`, or `scripts/measure_heuristics.py` into `data/games/` before ratings.
5. **Report** — winrate, draw rate, mean turns, decisive games, Elo delta, sample size.
6. **Decide** — keep or revert from stored metrics only.

## Commands

```bash
source .venv/bin/activate   # if present; prefer python3.12

# Single stored match
python arena/run_match.py bots/<a>/run.sh bots/<b>/run.sh --mode competition --seed <n>

# Full heuristic round (fixed grid + round report)
python scripts/measure_heuristics.py --round round<N>
```

Round reports: [`docs/research/measurements/`](../../../docs/research/measurements/). Edit bot lists in the script; do not pass an ad-hoc grid.

## Seat-order swap

Every grid must include both seat orders (`A vs B` and `B vs A`) for the same seeds when judging strength.

## Draw-heavy decision rule

When both arms draw every game, mark the result **unproven**, not neutral. Ask for a decisive opponent or a different grid before changing thresholds.

## Metrics checklist

- [ ] Same seeds before and after
- [ ] Same opponents
- [ ] Both seat orders where strength is claimed
- [ ] `--mode competition` on every match
- [ ] Games under `data/games/` before Elo update
- [ ] One changed parameter group per experiment
- [ ] Rating snapshot / delta via `arena/ratings.py` (see **update-leaderboard**)

Schema: [`docs/arena/game-record-schema.md`](../../../docs/arena/game-record-schema.md).

## Rules

- Do not merge strategy changes without a measured delta in `data/`.
- No strategy content in this skill — link `docs/` for domain knowledge.

## Changelog

- 2026-07-31 — Seat-order swap, draw rule, measure_heuristics pointer (cause: skills-workflow build)
