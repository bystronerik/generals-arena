# Experiment protocol

Use this for every measurable bot or arena change.

## Steps

1. **Hypothesis** — one change, one claim. Write a short note under `docs/research/` when useful.
2. **Match grid** — default Rule C: `--games-per-pair N` (default 50) random map seeds per unordered pair, no seat swap; pin `--round-seed` for a reproducible round. Use `--seeds` only for fixed-seed A/B debug.
3. **Run** — always competition mode. Store games under `data/games/<round>/` via `arena/tournament.py` or `scripts/measure_heuristics.py`.
4. **Metrics** — winrate, draw rate, mean turns, Elo delta, sample size.
5. **Decision** — keep or revert from stored metrics. Do not merge strategy changes without a measured delta in `data/`.

## Rules

- Prefer one change per experiment.
- Parallel batches store with ratings off inside workers; rebuild Elo once after the pool (default tournament / measure behavior).
- Verification gate: matches must finish under competition mode (see root `AGENTS.md`).
