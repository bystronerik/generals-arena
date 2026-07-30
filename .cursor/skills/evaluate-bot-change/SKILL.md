---
name: evaluate-bot-change
description: >-
  Measure a bot change with a fixed seed grid, before/after winrate, and
  rating delta. Use when comparing bot versions, running A/B tournaments,
  or deciding keep vs revert from stored data/games metrics.
---

# Evaluate bot change

Follow [`docs/research/experiment-protocol.md`](../../../docs/research/experiment-protocol.md). One change per experiment when possible.

## Workflow

1. **Hypothesis** — one claim; optional short note under `docs/research/`.
2. **Baseline** — fix opponent set + seed list. Run before the change (or use stored games that match that grid).
3. **Treat** — apply the change; re-run the same seeds and opponents.
4. **Store** — every game via `arena/run_match.py` or `arena/tournament.py` / `scripts/` into `data/games/` before ratings.
5. **Report** — winrate, draw rate, mean turns, Elo delta, sample size.
6. **Decide** — keep or revert from stored metrics only.

## Commands (Phase 2+)

```bash
# Single stored match
python arena/run_match.py bots/<a>/run.sh bots/<b>/run.sh --mode competition --seed <n>

# Seed × pair grid
python arena/tournament.py ...   # or scripts/ CLI wrapping the same
```

Prefer CPython 3.12 and `source .venv/bin/activate` when available.

## Metrics checklist

- [ ] Same seeds before and after
- [ ] Same opponents
- [ ] `--mode competition` on every match
- [ ] Games under `data/games/` before Elo update
- [ ] Rating snapshot / delta via `arena/ratings.py` (see **update-leaderboard**)

Schema: [`docs/arena/game-record-schema.md`](../../../docs/arena/game-record-schema.md).

## Rules

- Do not merge strategy changes without a measured delta in `data/`.
- No strategy content in this skill — link `docs/` for domain knowledge.
