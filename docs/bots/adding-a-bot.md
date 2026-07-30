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

## Phase 3 heuristic bots

Scaffolded with the `build-bot-from-spec` skill from a spec under
`docs/research/strategies/`. One doc
file each, one experiment note each under `docs/research/experiments/`:

| Bot | Idea | Doc | Experiment note |
| --- | --- | --- | --- |
| `bots/expand_plus/` | expand | [`expand-plus.md`](expand-plus.md) | [`001-expand-plus-frontier-march.md`](../research/experiments/001-expand-plus-frontier-march.md) |
| `bots/castle_builder/` | castle-aware | [`castle-builder.md`](castle-builder.md) | [`002-castle-builder-early-investment.md`](../research/experiments/002-castle-builder-early-investment.md) |
| `bots/general_hunter/` | late-game hunt | [`general-hunter.md`](general-hunter.md) | [`003-general-hunter-deathtouch-beeline.md`](../research/experiments/003-general-hunter-deathtouch-beeline.md) |

Every one of these strategy changes needs a note before merge — see
[`docs/research/experiment-protocol.md`](../research/experiment-protocol.md)
and skill `evaluate-bot-change`.
