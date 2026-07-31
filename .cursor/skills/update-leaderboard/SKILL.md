---
name: update-leaderboard
description: >-
  Refits arena ratings from stored data/games/ through arena/records/ratings/
  and publishes fit.json plus JSON and Markdown leaderboards under
  data/ratings/. Use when refreshing ratings, publishing the leaderboard, or
  closing a tournament or measurement round.
---

# Update leaderboard

## Model split

- Think model: decides when a rating snapshot is worth publishing
- Composer: refits after games are stored

**Composer must not invent a threshold.** Thresholds live in
[`docs/arena/decision-rule.md`](../../../docs/arena/decision-rule.md). When a
value is absent from both, Composer stops and asks the think model.

## Preconditions

1. Every match in the set is stored under `data/games/<round>/`.
2. Records are schema v4
   ([`docs/arena/game-record-schema.md`](../../../docs/arena/game-record-schema.md)):
   `mode == "competition"`, both content hashes present, `engine_version` and
   `round` set.
3. Both bots' hashes are registered in `data/bot_versions/`. The match runners
   do this automatically; verify with
   `python -m arena.records.registry --verify`.
4. Do **not** rate from live match stdout — store first.

## Refit

```bash
source .venv/bin/activate   # if present; prefer python3.12
python -m arena.records.ratings --print
```

There is **no incremental update**. Every write refits the whole pool from
`data/games/`; the fit is a pure function of (games, registry, policy, prior,
anchor), so a refit is the only correct operation. Per-round count caches keep
it cheap.

Typical flow after a grid:

1. `arena/tournaments/competition.py` (or `arena/matches/run_match.py`, or
   `scripts/measure_heuristics.py`) registers the roster and writes
   `data/games/<round>/`.
2. `python -m arena.records.ratings` aggregates eligible games into an integer
   count table and fits Bradley–Terry + Davidson draws + a shared seat term.
3. `data/ratings/` gets `fit.json`, `leaderboard.json`, `leaderboard.md`.

## Reading the output

- The anchor `cm_expander@<hash>` is pinned at exactly 1500.0. Every rating is
  relative to it.
- Rated identity is `bot_id@content_hash`, not the bot id.
- Entities below 30 games are **provisional**: listed after the ranked block
  and not eligible as a decision baseline. They are still in the fit.
- `excluded` counts every rejected game by reason. A non-zero
  `engine_mismatch` means the submodule moved and the leaderboard needs a
  regeneration round.
- **Never report a rank as a result.** Report the contrast:
  `python -m arena.records.ratings --lineage <bot>`.

## Outputs

| Path | Contents |
| --- | --- |
| `data/ratings/fit.json` | estimates, covariance, policy, prior, counts digest |
| `data/ratings/leaderboard.json` | ranked snapshot |
| `data/ratings/leaderboard.md` | same snapshot as Markdown |
| `data/ratings/cache/` | per-round count tables |

## Git hygiene

Everything under `data/ratings/` is derived and **gitignored** — keep it local.
`data/bot_versions/` is **committed**: it is the reviewable record of what each
version was and what it scored, and `git log -p data/bot_versions/<bot>.json`
is a bot's improvement history. Commit round reports under
`docs/research/measurements/`. Do **not** commit `data/games/*.json` unless the
user asks for the raw games.

`fit.json` carries no timestamp, so two refits over the same games produce a
byte-identical file. A diff means the games or the policy changed.

## Rules

- Store then rate — see the root `AGENTS.md` verification gate.
- Ratings from two engine eras never pool. Use `--era <sha>` to refit a past
  one; do not use `--all-eras` to make a number look better.
- Experiment reporting:
  [`docs/research/experiment-protocol.md`](../../../docs/research/experiment-protocol.md).

## Changelog

- 2026-07-31 — Batch BTDS refit replaces sequential Elo; registry preconditions,
  provisional/anchor/excluded semantics, no incremental path
- 2026-07-31 — Git hygiene line for data/games vs data/ratings (cause: skills-workflow build)
