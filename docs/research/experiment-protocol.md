# Experiment protocol

Use this for every measurable bot or arena change.

## Steps

1. **Hypothesis** — one change, one claim. Write a short note under `docs/research/` when useful.
2. **Match grid** — fixed seed list; same opponent set before and after.
3. **Run** — always `--mode competition`. Store each game under `data/games/` (Phase 2+).
4. **Metrics** — winrate, draw rate, mean turns, Elo delta, sample size.
5. **Decision** — keep or revert from stored metrics. Do not merge strategy changes without a measured delta in `data/`.

## Rules

- Prefer one change per experiment.
- Update ratings only after games are stored.
- Verification gate: matches must finish under `--mode competition` (see root `AGENTS.md`).
