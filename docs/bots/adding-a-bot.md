# Adding a bot

Scaffold for Phase 1+.

## Layout

```text
bots/<name>/
├── agent.py    # decide actions from observation
├── main.py     # stdio protocol IO
└── run.sh      # entry the matchup runner executes
```

Optional: `build.sh` beside `run.sh` for compile / weight prep.

## Pattern

Copy from `competition-module/competition/agents/expander_python/`. Keep the same wire protocol as [`../competition/protocol.md`](../competition/protocol.md).

## Register for matches

Pass your `run.sh` to matchup:

```bash
python competition-module/competition/matchup.py \
  bots/<name>/run.sh \
  competition-module/competition/agents/expander_python/run.sh \
  --mode competition --seed 0
```

## Rules

- Competition mode only for verification.
- Do not edit submodule agents in place; wrap under `bots/`.
- Put strategy notes in `docs/bots/` or research notes, not in `AGENTS.md`.

## Smoke bot

Phase 1 owns `bots/smoke/`. Observation notes go in `docs/bots/smoke.md` after the first verified match.
