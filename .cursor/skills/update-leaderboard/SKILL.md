---
name: update-leaderboard
description: >-
  Rebuilds Elo leaderboard snapshots from stored data/games/ through
  arena/records/ratings.py and publishes JSON and Markdown under data/ratings/. Use when
  refreshing ratings, publishing the leaderboard, or closing a tournament or
  measurement round.
---

# Update leaderboard

## Model split

- Think model: decides when a rating snapshot is worth committing
- Composer: runs `arena/records/ratings.py` after games are stored

**Composer must not invent a threshold.** When a value is absent from the specification, Composer stops and asks the think model.

## Preconditions

1. Every match in the rebuild set is stored under `data/games/*.json`.
2. Records use the schema in [`docs/arena/game-record-schema.md`](../../../docs/arena/game-record-schema.md) (`winner`: `a` / `b` / `draw`, competition `mode`).
3. Do **not** update ratings from live match stdout alone — store first.

## Rebuild

```bash
source .venv/bin/activate   # if present; prefer python3.12
python -m arena.records.ratings     # or scripts/ CLI that wraps arena/records/ratings.py
```

Typical flow after a grid:

1. `arena/tournament.py` (or `arena/run_match.py` / `scripts/measure_heuristics.py`) writes `data/games/`.
2. `arena/records/ratings.py` reads stored games, applies elote `EloCompetitor` (`beat` / `tied` / `lost_to`).
3. Persist competitor state + leaderboard under `data/ratings/`.

## Outputs

| Path | Contents |
| --- | --- |
| `data/ratings/` | Competitor state + leaderboard snapshot (JSON and/or Markdown) |

## Git hygiene

Rebuild writes JSON and Markdown under `data/ratings/`. Those files are
**gitignored** — keep them local. Commit round reports under
`docs/research/measurements/` when publishing results. Do **not** commit
`data/games/*.json` unless the user asks for the raw games.

## Rules

- Prefer explicit result recording from the game store over opaque arena helpers that skip persistence.
- Store then rate — see root `AGENTS.md` verification gate.
- Experiment reporting: [`docs/research/experiment-protocol.md`](../../../docs/research/experiment-protocol.md).

## Changelog

- 2026-07-31 — Git hygiene line for data/games vs data/ratings (cause: skills-workflow build)
