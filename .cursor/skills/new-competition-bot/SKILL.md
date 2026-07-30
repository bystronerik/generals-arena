---
name: new-competition-bot
description: >-
  Scaffold a new stdio competition bot from bots/smoke/. Use when adding a
  bot under bots/<name>/, copying the smoke template, or wiring agent.py,
  main.py, and run.sh for matchup.py.
---

# New competition bot

## Scaffold

1. Copy the template:

```bash
cp -R bots/smoke bots/<name>
```

2. Keep the layout:

```text
bots/<name>/
├── agent.py    # actions from observation
├── main.py     # stdio protocol IO (keep unless protocol changes)
└── run.sh      # entry for matchup / arena
```

3. Change strategy only in `agent.py` (and docs under `docs/bots/` or `docs/research/`). Leave protocol IO intact unless the wire format requires it.

Optional: `build.sh` beside `run.sh` for compile / weight prep.

## Verify

```bash
source .venv/bin/activate   # if present; prefer python3.12
python competition-module/competition/matchup.py \
  bots/<name>/run.sh \
  bots/smoke/run.sh \
  --mode competition --seed 0
```

Or use skill **run-competition-match** / `arena/run_match.py` so the game lands in `data/games/`.

## Rules

- Do not edit `competition-module` agents in place; wrap under `bots/`.
- Put game knowledge in `docs/`, not in `AGENTS.md` or this skill.
- Process docs: [`docs/bots/adding-a-bot.md`](../../../docs/bots/adding-a-bot.md), protocol: [`docs/competition/protocol.md`](../../../docs/competition/protocol.md).
