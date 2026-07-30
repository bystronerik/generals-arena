# Phase 2 kickoff — arena + skills

Goal: match store, ratings, tournament CLIs, and Cursor skills. No heuristic maximization yet.

## Order

1. Arena core: `arena/run_match.py`, `arena/store.py`, `arena/ratings.py`, `arena/tournament.py` (+ thin `scripts/`).
2. Skills under `.cursor/skills/` (already scaffolded): run / new-bot / evaluate / leaderboard.
3. Verification: stored game under `data/games/` from a `--mode competition` match; then ratings under `data/ratings/`.

## Roles

- bot-author: keep `bots/smoke/` green; scaffold only via `new-competition-bot`.
- evaluator: fixed seeds; store before rate.
- docs-keeper: sync `docs/arena/` and experiment protocol with the real schema.

Read `AGENTS.md` first. Put strategy in `docs/`, not here.
