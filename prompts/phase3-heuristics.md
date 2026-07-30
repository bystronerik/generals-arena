# Phase 3 kickoff — heuristic bots

Start only after Phase 2 verification: store + ratings work under `--mode competition`.

## Loop

1. Hypothesis (one change) → note under `docs/research/` when useful.
2. Scaffold or edit via `new-competition-bot` / `bots/<name>/`.
3. Evaluate with `evaluate-bot-change` (fixed seed grid, same opponents).
4. Keep or revert from `data/games/` metrics; refresh with `update-leaderboard`.

## Constraints

- Always `--mode competition`.
- Store every game before Elo updates.
- Process in `AGENTS.md` and `.cursor/skills/`; strategy only in `docs/`.
