---
name: update-leaderboard
description: >-
  Rebuild Elo leaderboard snapshots from stored data/games/ via
  arena/ratings.py. Use when refreshing ratings, publishing leaderboard
  JSON/Markdown under data/ratings/, or after a tournament finishes.
---

# Update leaderboard

## Preconditions

1. Every match in the rebuild set is stored under `data/games/*.json`.
2. Records use the schema in [`docs/arena/game-record-schema.md`](../../../docs/arena/game-record-schema.md) (`winner`: `a` / `b` / `draw`, competition `mode`).
3. Do **not** update ratings from live match stdout alone — store first.

## Rebuild

```bash
source .venv/bin/activate   # if present; prefer python3.12
python arena/ratings.py     # or scripts/ CLI that wraps arena/ratings.py
```

Typical flow after a grid:

1. `arena/tournament.py` (or `arena/run_match.py`) writes `data/games/`.
2. `arena/ratings.py` reads stored games, applies elote `EloCompetitor` (`beat` / `tied` / `lost_to`).
3. Persist competitor state + leaderboard under `data/ratings/`.

## Outputs

| Path | Contents |
| --- | --- |
| `data/ratings/` | Competitor state + leaderboard snapshot (JSON and/or Markdown) |

## Rules

- Prefer explicit result recording from the game store over opaque arena helpers that skip persistence.
- After Phase 2, store then rate — see root `AGENTS.md` verification gate.
- Experiment reporting: [`docs/research/experiment-protocol.md`](../../../docs/research/experiment-protocol.md).
