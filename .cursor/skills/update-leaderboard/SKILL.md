---
name: update-leaderboard
description: >-
  Refits arena ratings from stored data/games/ through arena/records/ratings/ —
  one independent fit per round — and publishes fits/<round>.json plus the JSON
  and Markdown leaderboards under data/ratings/. Use when refreshing ratings,
  publishing the leaderboard, or closing a tournament or measurement round.
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

There is **no incremental update**, and there is **no pooled fit**. Every write
refits **every round** under `data/games/`, each one independently over its own
games. A round's fit is a pure function of (that round's games, registry, policy,
prior, anchor), so a refit is the only correct operation. Per-round count caches
keep it cheap — 13 solves total 0.077 s.

Refitting every round on every write is not laziness about scope: a write filter
would mean `leaderboard.json` either loses the unlisted rounds or carries them
forward from the previous file, and carrying forward is an incremental update in
everything but name. `--round <name>` is a **report** filter only.

Typical flow after a grid:

1. `arena/tournaments/competition.py` (or `arena/matches/run_match.py`, or
   `scripts/measure_heuristics.py`) registers the roster and writes
   `data/games/<round>/`.
2. `python -m arena.records.ratings` aggregates each round's eligible games into
   an integer count table and fits Bradley–Terry + Davidson draws + a shared seat
   term, once per round.
3. `data/ratings/` gets `fits/<round>.json` per rated round, `leaderboard.json`,
   `leaderboard.md`.

`--list-rounds` prints the round set — name, status, rated/stored, entities,
groups, anchor and tier — and exits without writing. Start there when a number
looks wrong.

## Reading the output

- **Never compare a rating in one table to a rating in another.** Every section is
  a separate fit on its own scale. A shared anchor fixes the additive constant, not
  the conditions the games were played under: two byte-identical programs measured
  in different rounds fitted 46 Elo apart. Each table prints a `scale` token — same
  token, comparable; different token, not, and no two rounds ever share one.
- **Rank is per-round and is never a result.** Rank numbers restart at 1 in every
  section. Report the contrast, from inside one round:
  `fits["<round>"].delta(baseline, candidate)`.
- **Anchor tiers.** The header names the anchor and its kind. `global` means
  `cm_expander@<hash>` played in that round and is pinned at 1500.0. `round_local`
  means it did not, so the round's most-played entity is pinned instead — the
  global anchor is never forced into a round it did not play, because a zero-game
  anchor is its own connectivity component and the prior then does the anchor's
  job. A local anchor changes one line of provenance, not the table.
- Rated identity is `bot_id@content_hash`, not the bot id.
- Entities below 30 games **in that round** are provisional: listed after the
  ranked block and not eligible as a decision baseline. They are still in the fit.
  One entity can be ranked in one round and provisional in another.
- **An empty ranked block is a valid readout.** A round where nobody reached 30
  games ranks nobody, says so in a sentence, and prints the full provisional table.
  A 22-game probe should read like one.
- **Unrated rounds are reported, never dropped.** `status: unrated` with a reason
  (`no_eligible_games`, `no_games_in_era`, `solver_failed`), the stored count and
  the `excluded` breakdown. A round missing from the report is indistinguishable
  from a round nobody ran. The usual fix is
  `python -m arena.records.registry --verify` then a refit.
- `excluded` counts every rejected game by reason, per round. A non-zero
  `engine_mismatch` with `era_split` true means that round spans an engine bump; it
  is rated on the current era only and the remedy is to re-run the round, not to
  split it.
- A round can be rated with `solver.converged` false. The residual is printed next
  to the table: near the 1e-9 tolerance the strengths are settled, and a round with
  no draws stalls there because `κ` runs toward −∞ against its prior alone.

## Outputs

| Path | Contents |
| --- | --- |
| `data/ratings/fits/<round>.json` | one rated round: estimates, covariance, policy, prior, counts digest, scale |
| `data/ratings/leaderboard.json` | every round's status and rows, format version 2 |
| `data/ratings/leaderboard.md` | the same, as one document of per-round sections |

Unrated rounds get **no** fit file — there is nothing to describe — and their
status lives in `leaderboard.json`. The writer prunes: a `fits/*.json` for a round
no longer under `data/games/` is deleted, because a stale published fit is a table
for a round that does not exist with nothing on it saying so.

## Git hygiene

Everything under `data/ratings/` is derived and **gitignored** — keep it local.
`data/bot_versions/` is **committed**: it is the reviewable record of what each
version was and what it scored, and `git log -p data/bot_versions/<bot>.json`
is a bot's improvement history. Commit round reports under
`docs/research/measurements/`. Do **not** commit `data/games/*.json` unless the
user asks for the raw games.

No fit file carries a timestamp, so two refits over the same games produce
byte-identical files. A diff in `fits/<round>.json` means that round's games or the
policy changed — and only that round's file moves when one round gains a game.

## Rules

- Store then rate — see the root `AGENTS.md` verification gate.
- **Never compare across rounds.** A contrast needs both arms in one round:
  `fits["<round>"].delta(a, b)`. There is nothing to read instead.
- Ratings from two engine eras never pool. Use `--era <sha>` to refit a past
  one. There is no flag that disables the era filter.
- Experiment reporting:
  [`docs/research/experiment-protocol.md`](../../../docs/research/experiment-protocol.md).

## Changelog

- 2026-07-31 — Batch BTDS refit replaces sequential Elo; registry preconditions,
  provisional/anchor/excluded semantics, no incremental path
- 2026-07-31 — Git hygiene line for data/games vs data/ratings (cause: skills-workflow build)
- 2026-08-14 — One independent fit per round replaces the pooled fit; `fit.json`
  becomes `fits/<round>.json`; anchor tiers, scale tokens, unrated rounds, empty
  ranked blocks, `--list-rounds` / `--round`; the single global ranked table is
  gone (cause: two byte-identical programs fitted 46 Elo apart across rounds)
