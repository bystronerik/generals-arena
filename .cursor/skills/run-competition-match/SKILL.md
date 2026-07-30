---
name: run-competition-match
description: >-
  Run Generals Competition local matches with --mode competition, using
  matchup.py or arena/run_match.py. Use when starting a match, debugging
  stdio bots, verifying the competition gate, or writing results under
  data/games/.
---

# Run competition match

## Environment

- Prefer **CPython 3.12** (competition sandbox pin).
- From repo root: `source .venv/bin/activate` when a project venv exists.
- Engine: `pip install -e competition-module`. Arena deps: `pip install -r requirements.txt`.

## Quick verify (no store)

```bash
python competition-module/competition/matchup.py \
  bots/<bot_a>/run.sh \
  bots/<bot_b>/run.sh \
  --mode competition --seed 0
```

Always pass `--mode competition`. The match must finish (win, loss, or draw / truncation).

## Arena store (Phase 2+)

Prefer the wrapper when rating or logging:

```bash
python arena/run_match.py \
  bots/<bot_a>/run.sh \
  bots/<bot_b>/run.sh \
  --mode competition --seed 0
```

Thin CLIs (when present): `scripts/` → same paths.

## Outputs

| Path | Contents |
| --- | --- |
| stdout / stderr | Match logs from `matchup.py` / bot processes |
| `data/games/<game_id>.json` | Stored record after arena runner finishes |
| `data/ratings/` | Updated only after games are stored (see update-leaderboard) |

Schema: [`docs/arena/game-record-schema.md`](../../../docs/arena/game-record-schema.md).

## Rules

- Do not treat classic / non-competition presets as verification.
- Store the game under `data/games/` before updating ratings.
- Domain detail: [`docs/engine/local-matchup.md`](../../../docs/engine/local-matchup.md), [`docs/competition/protocol.md`](../../../docs/competition/protocol.md).
