---
name: build-bot-from-spec
description: >-
  Scaffolds bots/<name>/ from bots/smoke/ and implements agent.py from a
  strategy specification under docs/research/strategies/, keeping every
  threshold as a named constant. Use when adding a competition bot, turning a
  strategy spec into code, copying the smoke template, or wiring agent.py,
  main.py, and run.sh for matchup.py.
---

# Build bot from spec

## Model split

- Think model: writes or revises the strategy specification under `docs/research/strategies/`
- Composer: scaffolds `bots/<name>/`, implements `agent.py`, runs one verification match

**Composer must not invent a threshold.** When a value is absent from the specification, Composer stops and asks the think model.

## Input

One path: `docs/research/strategies/<bot>.md`. Read it before writing code.

Reference example: [`docs/research/strategies/garrison.md`](../../../docs/research/strategies/garrison.md).

## Scaffold

```bash
cp -R bots/smoke bots/<name>
```

Layout:

```text
bots/<name>/
├── agent.py    # actions from observation; named constants for every threshold
├── main.py     # stdio protocol IO (keep unless protocol changes)
└── run.sh      # entry for matchup / arena
```

Optional: `build.sh` beside `run.sh` for compile / weight prep.

## Implementation rules

- Every threshold in the specification becomes a **named module constant** in `agent.py`. Later parameter revisions edit constants only.
- Change strategy only in `agent.py`. Leave `main.py` and `run.sh` unchanged unless the wire protocol changes.
- Return a legal move or a pass inside the **150 ms** move limit.
- Do not read hidden engine state. Use competition observation only.
- Put game knowledge in `docs/`, not in `AGENTS.md` or this skill.

## Verify

```bash
source .venv/bin/activate   # if present; prefer python3.12
python competition-module/competition/matchup.py \
  bots/<name>/run.sh \
  bots/smoke/run.sh \
  --mode competition --seed 0
```

Or use skill **run-competition-match** / `arena/matches/run_match.py` so the game lands in `data/games/`.

## Rules

- Do not edit `competition-module` agents in place; wrap under `bots/`.
- Process docs: [`docs/bots/adding-a-bot.md`](../../../docs/bots/adding-a-bot.md), protocol: [`docs/competition/protocol.md`](../../../docs/competition/protocol.md).

## Changelog

- 2026-07-31 — Renamed from new-competition-bot; spec-driven constants (cause: skills-workflow build)
