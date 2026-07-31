---
name: run-competition-match
description: >-
  Runs Generals Competition local matches with --mode competition through
  matchup.py or arena/run_match.py, and checks the result for bot faults. Use
  when starting a match, debugging a stdio bot, verifying the competition gate,
  swapping seat order, or writing a game record under data/games/.
---

# Run competition match

## Model split

- Think model: decides when a match claim needs both seat orders or castle telemetry
- Composer: runs the command, captures stdout/stderr, stores the game when asked

**Composer must not invent a threshold.** When a value is absent from the specification, Composer stops and asks the think model.

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

Prefer the wrapper when rating or logging. Single matches use the in-process
runner (`arena/competition_match.py`) via `run_and_store`:

```bash
python -m arena.run_match \
  bots/<bot_a>/run.sh \
  bots/<bot_b>/run.sh \
  --mode competition --seed 0
```

Batch grids use `arena/tournament.py` / `scripts/measure_heuristics.py` with
`--round <name>` (games under `data/games/<round>/`).

## Fault detection

A match can finish and still fail. After every run, read stderr and stdout for:

- bot faults and tracebacks
- protocol errors
- move-limit overruns (150 ms per move)

Report a **fault** as a failure even when the engine reports a draw or records a winner.

## Seat-order swap

Map generation is not symmetric. For a small fixed-seed A/B claim, run both
seat orders (`A vs B` and `B vs A`) on the same seeds. Large Rule C rounds
(`--games-per-pair` with random seeds) skip seat swap by default.

## Castle telemetry

When castles matter, capture stdout. The only castle signal in reports is:

```text
[matchup] castles built: <a> (<label>) vs <b> (<label>)
```

## Draw baseline

Most stored games end as draws at the 1200-turn cap. A draw passes the verification gate. A draw is **not** evidence of strength.

## Outputs

| Path | Contents |
| --- | --- |
| stdout / stderr | Match logs from `matchup.py` / bot processes |
| `data/games/<game_id>.json` or `data/games/<round>/<game_id>.json` | Stored record after arena runner finishes |
| `data/ratings/` | Updated only after games are stored (see **update-leaderboard**) |

Schema: [`docs/arena/game-record-schema.md`](../../../docs/arena/game-record-schema.md).

## Rules

- Do not treat classic / non-competition presets as verification.
- Store the game under `data/games/` before updating ratings.
- Domain detail: [`docs/engine/local-matchup.md`](../../../docs/engine/local-matchup.md), [`docs/competition/protocol.md`](../../../docs/competition/protocol.md).

## Changelog

- 2026-07-31 — In-process runner + per-round batch paths
- 2026-07-31 — Initial taxonomy alignment from skills-workflow.md (cause: skills-workflow build)
