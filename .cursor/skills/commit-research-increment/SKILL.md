---
name: commit-research-increment
description: >-
  Creates small ordered commits for a research step and keeps raw data/games
  JSON out of git while committing ratings and round reports. Use when
  committing a strategy spec, bot code, a measurement round, or a leaderboard
  update.
---

# Commit research increment

## Model split

- Think model: nothing
- Composer: stages and commits when the user explicitly asks

**Composer must not invent a threshold.** When a value is absent from the specification, Composer stops and asks the think model.

## Rules

- Commit after each loop step: specification, bot code, round report, ratings.
- Do not mix a specification commit with a bot code commit.
- Do **not** commit `data/games/*.json` unless the user asks for the raw games.
- Commit `data/ratings/` snapshots and round reports under `docs/research/measurements/`.
- Never commit `__pycache__/` or `.DS_Store`. Propose a `.gitignore` line when such a file appears in `git status`.
- **Never commit without an explicit user request** when repo policy requires that. State the proposed message and wait.

## Suggested commit order

1. Strategy spec (`docs/research/strategies/<bot>.md`)
2. Bot code (`bots/<bot>/`)
3. Measurement round report (`docs/research/measurements/round<N>.*`)
4. Ratings snapshot (`data/ratings/`)

Authority: [`docs/research/strategies/skills-workflow.md`](../../../docs/research/strategies/skills-workflow.md).

## Changelog

- 2026-07-31 — Initial skill from skills-workflow taxonomy (cause: skills-workflow build)
