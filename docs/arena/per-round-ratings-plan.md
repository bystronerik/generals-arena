# Per-round ratings: refactor plan

Status: **implemented, 2026-08-14.** All 15 ordered steps landed. This document is
kept as the design record; the shipped behaviour is documented in
[ratings.md](ratings.md) and [decision-rule.md](decision-rule.md).

Three things came out differently from the plan below, and they are noted where
they occur rather than rewritten out of it:

- `round_count_tables` returns `RoundCounts` (round, table, engine eras) rather
  than `(str, CountTable)` pairs, because the schema's `engine_versions` and
  `era_split` need the round's stored eras and the cache is where they are read.
  `CACHE_FORMAT_VERSION` went to 2 to carry them.
- `solver_failed` covers a solve that **raises**, not one that merely fails to
  converge. A non-converged round still publishes, with `solver.converged` false
  and the residual printed next to its table — the same contract the pooled fit
  had. `morpheus-castle-r1` is that case today: `max|grad| 5.9e-9` against a 1e-9
  tolerance, because the round has no draws and `κ` runs toward −∞ against its
  prior alone.
- The writer also deletes a leftover top-level `data/ratings/fit.json`. A file of
  that name holding one table over every round is the claim this refactor
  withdraws, so leaving it published would keep making it.

Today `arena/records/ratings/` fits one BTDS model over every eligible stored
game and publishes one ranked table. This plan replaces that with **one
independent fit per round**, published as per-round tables in the same two
files. The single global ranked table is removed, not kept alongside.

The reason is already written down and already measured. `decision-rule.md`
records two byte-identical `macaria` programs fitting **+46.07 ± 22.01 Elo,
P = 0.982** across different rounds (2026-08-01), and the M6 morpheus-rs round
disagreeing with four re-measurements by **7.4 sigma** (2026-08-09). A pooled
fit presents numbers from rounds that drift by more than the decision
thresholds as a single comparable column. Per-round fits stop the pool from
making that claim.

## Measured starting state

Everything below was read off the current store on 2026-08-14, at engine era
`9e3b9d13cca5`. It is the evidence the four resolutions rest on.

| | |
| --- | --- |
| Round directories under `data/games/` | 16, plus the `_root` bucket |
| `_root` game records | **102** (not ~118 — that count included the 16 directories); 94 eligible, 8 `unregistered_hash`, 29 entities, **4** connectivity groups |
| Rated games today (pooled) | 9,150 over 44 entities, **2** groups (32 + 12) |
| Rated games after dropping `_root` | **9,056** over 29 entities |
| Entities that exist only in `_root` | **15** (14 `morpheus@…` smoke hashes + `smoke@30a5f8f5da4a`) |
| Rounds containing the global anchor | **8** |
| Rounds with eligible games but no anchor | **5** |
| Rounds with zero eligible games | **3** (all `morpheus-cadence-*`, 739 records, all `unregistered_hash`) |
| Rounds where no entity reaches 30 games | **3** (`m7-contention-probe`, `-loaded`, `morpheus-bootstrap`) |
| Connectivity groups per round | **1 for every non-empty round.** Only `_root` is split |

Per round, in the order the loader walks them:

| Round | Stored | Eligible | Entities | Groups | ≥30 games | Anchor |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `_root` | 102 | 94 | 29 | 4 | 0 | yes |
| `belief-ablation-off` | 100 | 100 | 2 | 1 | 2 | **no** |
| `belief-ablation-on` | 100 | 100 | 2 | 1 | 2 | **no** |
| `joe-entry-r1` | 150 | 150 | 6 | 1 | 6 | yes |
| `joe-r2` | 440 | 440 | 7 | 1 | 7 | yes |
| `joe-r3` | 640 | 640 | 5 | 1 | 5 | yes |
| `joe-r4` | 560 | 560 | 4 | 1 | 4 | yes |
| `joe-rs-ab-r1` | 200 | 200 | 7 | 1 | 7 | yes |
| `m7-contention-probe` | 22 | 22 | 2 | 1 | **0** | **no** |
| `m7-contention-probe-loaded` | 22 | 22 | 2 | 1 | **0** | **no** |
| `morpheus-bootstrap` | 42 | 42 | 7 | 1 | **0** | yes |
| `morpheus-cadence-ckpt100-ckpt200` | 509 | **0** | 0 | — | — | — |
| `morpheus-cadence-ckpt1500-ckpt2000` | 170 | **0** | 0 | — | — | — |
| `morpheus-cadence-ckpt4600-ckpt6000` | 60 | **0** | 0 | — | — | — |
| `morpheus-castle-r1` | 60 | 60 | 4 | 1 | 4 | **no** |
| `morpheus-rs-m6-strength` | 4032 | 4032 | 7 | 1 | 7 | yes |
| `morpheus-rs-m7-knobs` | 2688 | 2688 | 8 | 1 | 8 | yes |

Cost, measured on this store:

| | |
| --- | --- |
| Per-round count tables read from cache (all 17) | 0.093 s |
| One pooled Newton solve | 0.002 s |
| **13 per-round Newton solves, total** | **0.077 s** |
| Cold read of all 9,897 game files | 0.944 s |

---

## Resolutions

### 1. Anchor and cross-round scale

**Anchoring is resolved per round, in three tiers, and the tier is recorded.**

| Tier | Condition | Anchor | `anchor_kind` |
| --- | --- | --- | --- |
| 1 | the global anchor entity played ≥ 1 eligible game in this round | `cm_expander@<latest registered hash>` at 1500.0 | `global` |
| 2 | it did not, but the round has eligible games | the round's most-played entity at 1500.0, ties broken by entity key ascending | `round_local` |
| 3 | the round has no eligible games | none; the round is not fitted | — |

Tier 2's tiebreak is `max(sorted(entities), key=games_in_round)` — the same
deterministic rule `scripts/measure_heuristics.py:round_leaderboard_snippet`
already uses for its round-local reports, so the two agree instead of being two
conventions. On today's store tier 1 covers 8 rounds and tier 2 covers 5.

**The global anchor is never forced into a round that did not play it.** This
is the load-bearing rule, and it is not cosmetic. Forcing it in via
`count_table(extra_entities=[anchor])` makes the anchor a zero-game singleton
component; the round's real entities then form a second component whose
location is set by the prior, and the prior pulls every free parameter toward
1500 symmetrically. Measured on a synthetic 10-game round (`b` beats `c` 9–1):

| | `b` | `c` | contrast `b → c` | groups |
| --- | --- | --- | --- | --- |
| anchor forced in (absent from round) | 1610.0 ± 170.7 | 1390.0 ± 170.7 | −220.0 ± 191.3 | 2 |
| round-local anchor on `b` | 1500.0 (pinned) | 1348.5 ± 157.2 | −151.5 ± 157.2 | 1 |

The forced version does not merely relabel the scale — it moves the contrast by
68 Elo and inflates its SE, because the prior is now doing work that an anchor
should be doing. `tests/test_ratings_connectivity.py:136`
(`test_an_isolated_anchor_is_flagged_rather_than_silently_compared`) already
covers the shape; this makes avoiding it a rule of the round fitter.

**A round's table when it has no anchor** shows exactly the same columns, with
its header naming the round-local anchor and its kind. Nothing is hidden. What
changes is one line of provenance, not the table.

**What stops a reader comparing across tables.** Three things, in order of how
much they actually work:

1. **There is no table that invites it.** The single ranked list is gone. A
   cross-round comparison now requires the reader to open two sections and do
   the arithmetic themselves, which is a deliberate act rather than reading a
   row.
2. **Every table carries a `scale` token.** `scale_id` is
   `sha256(round \0 anchor_entity \0 anchor_kind \0 counts_digest)[:12]`,
   printed in each section header and stored in both JSON outputs. Two rounds
   can never share one, because the round name is in the digest. This gives a
   mechanical rule — *same `scale`, comparable; different `scale`, not* —
   instead of prose a reader can talk themselves out of.
3. **The document says so once, at the top, with the evidence.** A common
   anchor fixes the additive constant, not the conditions the games were played
   under; the +46 Elo between byte-identical programs is what that difference
   costs. The banner says that and links `decision-rule.md`.

Be honest about the limit: **rendering cannot make comparison impossible.**
Any absolute number can be subtracted from any other. The alternative
considered — printing `Δ vs anchor` instead of a 1500-based rating — was
rejected because it is the same number minus 1500 and prevents nothing, while
breaking the convention every existing round report already uses. What the
refactor buys is that no *published artifact* makes the comparison for the
reader, and no rank spans two rounds.

### 2. Connectivity inside a round

Connectivity is computed on the round's own count table, with the existing
`connected_components`. Nothing new is needed for the computation — only for
the scope, which becomes the round.

- **Connected round (every round today except `_root`):** table renders as it
  does now, no `Group` column, no warning.
- **Disconnected round:** that round's section gets the existing warning block
  and `Group` column, scoped to the section. Other rounds' sections are
  untouched — a split in one round must not annotate another. The payload also
  records `anchor_component`, because when the anchor lands in a minority group
  the majority of the round's rows are prior-located, and the warning should
  say which group is the measured one.
- **A round that cannot be fitted at all** gets `status: "unrated"` and a
  reason, and is rendered as a stub section with no table:

  | `reason` | Cause | Today |
  | --- | --- | --- |
  | `no_eligible_games` | every record rejected by the policy | 3 rounds, 739 records, all `unregistered_hash` |
  | `no_games_in_era` | the era filter emptied the round | 0 rounds |
  | `solver_failed` | Newton did not converge | 0 rounds; the prior makes it near-impossible, but it is handled rather than crashing |

  An unrated round is **never silently dropped**. A round that disappears from
  the report is indistinguishable from a round nobody ran, and the actionable
  content is precisely the part that survives: the stored-game count and the
  `excluded` breakdown, which for the `morpheus-cadence-*` rounds say "register
  these hashes and refit". One unrated round never aborts the others; the CLI
  prints a per-round warning and exits 0, and exits 1 only when *every* round
  is unrated.

  A round whose eligible games involve a single entity (self-play only) does
  fit — verified: one row, 1500.0, SE 0.0, one component. It stays `rated`. The
  table is degenerate but not wrong, and the row count makes that obvious.

### 3. Provisional gate

**`min_games_display` stays at 30, is applied per round, and a round table may
have an empty ranked block.**

The 30-game floor is not an artifact of the pool's size. It descends from the
decision rule — ≥ 200 games per arm and ≥ 30 games per (arm, opponent) — and it
means "below this, an entity may not be a decision baseline". That statement is
exactly as true inside one round as it was across the pool. Lowering it per
round would keep every table full at the cost of turning the gate into
decoration.

So the gate counts an entity's games **in that round**, and three of today's
non-empty rounds rank nobody: `m7-contention-probe` (22 games), its `-loaded`
twin (22), and `morpheus-bootstrap` (12 per entity). Their sections render the
ranked block as one sentence —

> No entity in this round reached 30 games, so this round ranks nobody and
> supplies no decision baseline. Every row below is provisional.

— followed by the full provisional table. That is a true and useful readout:
`m7-contention-probe` *is* a 22-game probe and should look like one. A round
with any eligible game always has a non-empty provisional block, so no rated
section is ever entirely tableless.

Consequence to state in the docs: one entity can be ranked in one round and
provisional in another. That is correct — each row is a statement about
evidence gathered inside that round.

`--min-games` stays a single global flag applied to every round. A per-round
override (`--min-games-round <round>=<n>`) is explicitly **out of scope**: it
would let an operator dial a round into rankability after seeing the result.

### 4. Schema of the combined outputs

#### `data/ratings/leaderboard.json` — format version 2

```json
{
  "version": 2,
  "format": "per_round",
  "engine_era": "9e3b9d13cca51caa1bb07db48bb85c9e90ce0462",
  "policy": { "modes": ["competition"], "engine_version": "…",
              "require_registered": true, "include_self_play": true,
              "min_games_display": 30 },
  "prior": { "mean": 1500.0, "sigma": 200.0, "seat_sigma": 200.0, "draw_sigma": 2.0 },
  "rounds_digest": "sha256:…",
  "rounds": [
    {
      "round": "joe-r4",
      "status": "rated",
      "scale_id": "scale:1f4c9a2b7e05",
      "anchor": "cm_expander@3097ee53a533",
      "anchor_kind": "global",
      "anchor_rating": 1500.0,
      "anchor_component": 0,
      "counts_digest": "sha256:…",
      "stored_games": 560,
      "rated_games": 560,
      "excluded": {},
      "engine_versions": ["9e3b9d13cca5…"],
      "era_split": false,
      "connected": true,
      "components": [["cm_expander@…", "joe@…", "joe_base@…", "joe_prev@…"]],
      "seat_advantage": { "value": -3.61, "se": 5.47 },
      "draw_log_nu":    { "value": -1.306, "se": 0.042 },
      "solver": { "iterations": 6, "max_abs_grad": 1.2e-11, "converged": true },
      "entities": [ /* LeaderboardRow.to_dict(), unchanged shape */ ]
    },
    {
      "round": "morpheus-cadence-ckpt100-ckpt200",
      "status": "unrated",
      "reason": "no_eligible_games",
      "stored_games": 509,
      "rated_games": 0,
      "excluded": { "unregistered_hash": 509 },
      "engine_versions": ["9e3b9d13cca5…"]
    }
  ]
}
```

Decisions inside that shape:

- **`rounds` is an array, not an object keyed by round name.** An object would
  be re-sorted by `_write_json`'s `sort_keys=True` anyway, and JSON object order
  is not guaranteed to any consumer. An array is ordered by construction and
  each element self-identifies with `"round"`.
- **Order is round name ascending, byte-wise.** Deterministic, and independent
  of any clock or `mtime`. Ordering by recency was rejected: it needs a
  timestamp, which the determinism constraint forbids in this file.
- **`scale_id` is the cross-round guard in machine-readable form.** Because the
  round name is inside the digest, two rounds with byte-identical games still
  get different tokens — a consumer that joins two rows on rating without
  checking `scale_id` has no excuse.
- **`stored_games` vs `rated_games`** makes the eligibility gap visible without
  summing `excluded`.
- **`engine_versions`** lists the distinct eras among the round's *stored*
  games, before the era filter, so a round spanning an engine bump is visible
  rather than inferred; `era_split` is the boolean form.
- **`rounds_digest`** = `sha256` over `(round, status, counts_digest)` for every
  round in order. It replaces the single top-level `counts_digest` as the
  one-read "does anything need refitting" answer.
- **Removed from the top level:** `entities`, `anchor`, `rated_games`,
  `counts_digest`, `connected`, `components`, `seat_advantage`. All are now
  per-round, and there is no pooled value for any of them.

#### `fit.json` → `data/ratings/fits/<round>.json`, one per rated round

`fit.json` does not become round-keyed and does not go away. It **splits**.

The reasoning is what the file is for: it carries the covariance matrix, and
covariance is what `fit.delta` reads to make a decision. Bundling every round's
covariance into one file means `load_fit` parses 16 matrices to answer one
contrast, and one new game in one round rewrites the whole file — which
destroys the property that makes the no-timestamp rule worth having. Separate
files keep byte-identical-diff **per round**: after a new round, exactly one
fit file changes and `git diff`-style comparison of the others is meaningful.

This also honours the constraint against writing a leaderboard JSON into each
round folder: `data/ratings/fits/<round>.json` sits beside
`data/ratings/cache/<round>.counts.json`, not in `data/games/<round>/`.

Payload: today's `fit_payload` shape at `FIT_FORMAT_VERSION = 2`, plus
`round`, `anchor_kind`, `scale_id`, `status`, `engine_versions`, `era_split`.
Unrated rounds get **no** fit file — there is nothing to describe — and the
combined `leaderboard.json` is where their status lives.

API changes that follow:

- `load_fit(round=...)` is required. `load_fit()` with no round raises and
  names the available rounds; it never silently returns "the first one".
- `stored_counts_digest(round=...)` likewise, plus `stored_rounds_digest()` for
  the whole-store question.
- The writer **prunes**: any `fits/*.json` whose round is not in the current
  round set is deleted. This matters because the same staleness already exists
  one directory over — `data/ratings/cache/` currently holds ~13
  `macaria-hunt-*.counts.json` files for rounds no longer under `data/games/`.
  A stale cache is harmless; a stale *published fit* is not. Prune the cache
  directory the same way while we are there.

---

## Ordered steps

Each step leaves the repo working and testable.

1. **`cache.py`: split the round tables out of the merge.** Add
   `round_directories(games_dir, *, include_root: bool = True)` and a new
   `round_count_tables(...) -> tuple[list[tuple[str, CountTable]], CacheStats]`.
   Reimplement `cached_count_table` as `merge()` over it, so the pooled path is
   provably the same work. No behaviour change; existing cache tests pass
   untouched.
2. **`fit.py`: give a fit its provenance.** `RatingFit` and `LoadedFit` gain
   `round: str | None`, `anchor_kind: str | None`, `scale_id: str | None`,
   defaulting to `None` so every existing construction site keeps working.
3. **New `arena/records/ratings/rounds.py`.** Holds `RoundResult` (a rated fit
   or an unrated status + reason), `RoundFits` (ordered, name-keyed lookup,
   `rounds_digest`), `resolve_round_anchor()` implementing the three tiers, and
   `fit_rounds()`. This is the only new module; keeping it out of `cli.py`
   preserves the module map in `__init__.py` (`cli.py` stays the CLI).
   `fit_rounds()` calls `round_count_tables(include_root=False)` — the `_root`
   exclusion lives in exactly one line, is named, and is testable.
4. **`io.py`: the new payloads.** `round_fit_payload`, `leaderboard_payload_v2`,
   `write_round_fits` (with prune), `write_leaderboard` rewritten,
   `load_fit(round=)`, `stored_counts_digest(round=)`, `stored_rounds_digest`.
   `LeaderboardRow` and `fit_from_payload` keep their shape.
5. **`reporting.py`: one new renderer.** `leaderboard_table_lines` needs no
   change — it already takes rows plus an explicit `split`. Add
   `round_section_lines(result)` producing a section's header, warnings, ranked
   block (or the empty-ranked sentence), and provisional block.
6. **`io.leaderboard_markdown` becomes a document.** Banner, index table, then
   one section per round in name order. This is where the single ranked table
   is deleted.
7. **`cli.py`: rewire and add flags.** `refit()` returns `RoundFits`. New
   `--round` (print filter) and `--list-rounds`. Per-round warnings for
   disconnection and unrated status. Exit 1 only when every round is unrated or
   the anchor bot is unregistered.
8. **`lineage.py`: per-round.** `lineage_deltas(bot_id, round_fit, registry)`
   unchanged in shape; `lineage_table_lines` gains a round argument, and
   `cli --lineage <bot>` prints one table per round in which the bot has a
   rated entity. A step whose predecessor did not play in the same round shows
   `—` with reason `not_in_this_round` instead of a number.
9. **Update the three remaining `refit()` callers.**
   `arena/tournaments/competition.py:315`, `scripts/measure_heuristics.py:403`,
   `scripts/morpheus_cadence_evidence.py:191`. The last is the substantive one:
   it takes `fit.delta(prior, later)` off the pooled fit and must take it off
   `fits["<its own round>"]`, which is the contrast the decision rule actually
   licenses. (`arena/matches/run_match.py:122` is deleted rather than updated —
   step 10.)
10. **Remove the per-game refit entirely** — see
    [Removing `--update-ratings`](#removing---update-ratings) below. Delete
    `run_and_store`'s `update_ratings` parameter and the refit block at
    `arena/matches/run_match.py:116-123`, the `--update-ratings` flag on
    `arena/matches/run_match.py:160` and `scripts/smoke_match.py:34`, and the
    now-dead `update_ratings` parameter on
    `scripts/measure_heuristics.py:run_one`.
11. **Fix `run_match`'s storage directory.** Companion to step 10, and needed
    on its own: `run_and_store` defaults `games_dir` to `GAMES_DIR`, so
    `--round X` sets the record's `round` field while the file lands flat in
    `data/games/`. That is how `_root` grew to 102 records carrying 49 distinct
    `round` values, and removing the rating flag does not change where the file
    goes. Default `games_dir` to `round_games_dir(round_name)` so an ad-hoc
    match lands in a round the rating layer can actually see.
12. **De-duplicate `scripts/measure_heuristics.py:round_leaderboard_snippet`.**
    It already does a round-local fit, but keyed on `bot_id` rather than
    `bot_id@content_hash` — a second, weaker identity convention. Re-point it
    at `rounds.fit_rounds()` for the round it just wrote, so there is one
    round-local fit in the repo.
13. **Tests** (below).
14. **Docs and skills** (below).
15. **Verify.** Run the `AGENTS.md` competition gate, then a full refit, then a
    second refit, and diff `data/ratings/fits/` byte-for-byte.

---

## Removing `--update-ratings`

The per-game refit goes away rather than being repaired. It was already the
wrong operation, and per-round fitting makes it strictly worse: a single
ad-hoc game would trigger a refit of **every** round — 16 solves and a full
rewrite of `data/ratings/` — to publish a game that lands in `_root` and
therefore contributes nothing to any number. Maximum work, zero effect. Keeping
it and adding a guard would leave a flag whose only remaining behaviour is to
refuse.

What is removed:

| Site | What goes |
| --- | --- |
| `arena/matches/run_match.py:62,116-123,160,191` | the `update_ratings` parameter, the refit block, the `--update-ratings` flag |
| `scripts/smoke_match.py:34,45` | the `--update-ratings` flag and its forward |
| `scripts/measure_heuristics.py:182,197` | `run_one`'s `update_ratings` parameter — its only caller already passes `False` (line 392), so it is dead |
| `README.md:85` | `python scripts/smoke_match.py --seed 0 --update-ratings` |
| `docs/arena/match-runner.md:24` | the sentence describing the flag |

What is **not** removed — these refit once after a whole round that stored into
`data/games/<round>/`, which is the correct granularity and stays exactly as it
is:

- `arena/tournaments/competition.py:217,311,432` — `run_tournament(update_ratings=)`
  and its `--no-ratings` inverse.
- `scripts/measure_heuristics.py:399-404` — the post-loop refit in
  `_run_legacy_grid`, and `--no-ratings` at lines 513/541.
- `training/morpheus/corpus/record.py:75,118` and `scripts/morpheus_corpus.py:69`
  — these forward to `run_tournament`, not to `run_and_store`, so they are
  untouched.

The replacement is one command, and it is already what `update-leaderboard`
prescribes ("Store then rate"):

```bash
python -m arena.records.ratings
```

Nothing decision-grade is lost. A one-game round cannot clear the 30-game gate,
so even a correctly-filed ad-hoc match publishes only a provisional row.

---

## CLI surface

`python -m arena.records.ratings` keeps every existing flag —
`--games-dir --ratings-dir --anchor --era --all-eras --sigma --min-games
--lineage --print --dry-run --no-cache` — with unchanged meaning. Two are added:

| Flag | Meaning |
| --- | --- |
| `--round <name>` (repeatable) | **Report filter only.** Restricts what `--print`, `--lineage` and the console summary show. It does **not** restrict what is written: every write still refits every round. |
| `--list-rounds` | Print the round set — name, status, stored/rated games, entities, groups, anchor and kind — then exit without writing. |

`--round` being a read filter and not a write filter is deliberate. A write
filter would mean `leaderboard.json` either loses the unlisted rounds or
carries them forward from the previous file, and carrying forward is an
incremental update in everything but name — the one thing this system refuses
to have. Refitting all 16 rounds costs 0.077 s of solve time, so there is
nothing to buy.

Optional, low priority: `--fail-on-unrated` for a non-zero exit when any round
is unrated. Useful for catching an unregistered roster, but `--list-rounds`
already answers the question.

Era flags are unchanged in semantics. The era is a policy value applied
identically to every round, so one publish is always one era; `engine_era` sits
at the top level of `leaderboard.json` and each round records its own observed
`engine_versions`.

**A round spanning an engine bump** is rated on the current era's subset. Its
other games land in that round's `excluded.engine_mismatch`, `era_split` is
true, and its section prints a line naming the count. It is **not** split into
two per-era tables: a round is defined by its manifest — one roster, one seed
list, one seat policy — and halving it by era produces two broken designs with
neither seat balance nor pair coverage, under round names no manifest and no
`data/games/` directory backs. The remedy the report names is to re-run the
round. Under `--all-eras` such a round pools across the bump, which stays
documented as unsound.

---

## `leaderboard.md` layout

```markdown
# Arena leaderboard

Updated: 2026-08-14T06:07:31Z
Engine era: `9e3b9d13cca5`
Rounds: 13 rated, 3 unrated · 9,056 rated games

Each round below is an **independent** Bradley-Terry + Davidson + seat fit over
that round's games only. No number here is fitted across rounds.

> **Ratings in two different tables are on two different scales. Do not
> compare them.** Each table is anchored inside its own round and carries its
> own `scale` token; two tables share a scale only if the tokens match, and no
> two rounds ever do. A shared anchor fixes the additive constant, not the
> conditions the games were played under: two byte-identical programs measured
> in different rounds fitted **46 Elo apart** in this repo. Rank is per-round
> and is never a result — decide from a pairwise contrast **inside one round**.
> See [decision-rule.md](decision-rule.md).

## Rounds

| Round | Status | Rated / stored | Entities | Ranked | Groups | Anchor | Scale |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| [belief-ablation-off](#belief-ablation-off) | rated | 100 / 100 | 2 | 2 | 1 | `macaria@5760b307` (round-local) | `4a1c…` |
| [joe-r4](#joe-r4) | rated | 560 / 560 | 4 | 4 | 1 | `cm_expander@3097ee53` (global) | `1f4c…` |
| [m7-contention-probe](#m7-contention-probe) | rated | 22 / 22 | 2 | **0** | 1 | `morpheus@73967d21` (round-local) | `c07b…` |
| [morpheus-cadence-ckpt100-ckpt200](#…) | **unrated** | 0 / 509 | — | — | — | — | — |

---

## joe-r4

Scale `1f4c9a2b7e05` · Anchor `cm_expander@3097ee53a533` (global) pinned at 1500.0
Rated 560 of 560 stored · engine era `9e3b9d13cca5` · one connectivity group
Seat-A advantage +1.2 ± 6.8 Elo · Draw log-nu −1.412 ± 0.061

| Rank | Entity | Rating | 95% CI | Games | W | L | D | Prov. |
| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | `joe@3307e9a05191` | … | … | 360 | 305 | 52 | 3 | |
| … |

---

## m7-contention-probe

Scale `c07b…` · Anchor `morpheus@73967d2125cc` (round-local) pinned at 1500.0
Rated 22 of 22 stored · engine era `9e3b9d13cca5` · one connectivity group

No entity in this round reached 30 games, so this round ranks nobody and
supplies no decision baseline. Every row below is provisional.

### Provisional in this round (< 30 games)

| Rank | Entity | Rating | 95% CI | Games | W | L | D | Prov. |
| … |

---

## morpheus-cadence-ckpt100-ckpt200 — unrated

Stored 509 games, rated 0. **Not rated: no eligible games.**
Excluded: `unregistered_hash` 509.

Register the roster (`python -m arena.records.registry --verify`) and refit.
```

Notes on the layout:

- The index table carries no rating and no cross-round rank, only provenance
  and counts.
- Rank numbers restart at 1 in every section and the banner says they are
  per-round.
- A disconnected round's section keeps the existing warning block and `Group`
  column, plus a line naming which group holds the anchor.
- `Updated:` in the header is the only clock in any output.

---

## Tests

Suite reality check: `pytest -q` over every `testpaths` entry is **13.94 s warm
/ 15.93 s cold**, 709 passed. The 15 s ceiling in `AGENTS.md` has about a
second of headroom warm and is already breached cold. The whole ratings suite
is 82 tests in **1.30 s**, so the rule for everything below is: synthetic count
tables and `tmp_path` only, never `data/games/`, at most 3 rounds of ≤ 8
records each. Target for the whole new set: **≤ 0.3 s**. The byte-identical
test does two full write passes and now writes N fit files per pass, so cap its
fixture at 2 rounds.

### `tests/test_ratings.py` — changed

| Test | Change |
| --- | --- |
| `test_two_fits_over_the_same_games_write_a_byte_identical_fit_json` | compare `fits/<round>.json` per round **and** `leaderboard.json` |
| `test_stored_digest_answers_whether_a_refit_is_needed` | round-scoped `stored_counts_digest(round=)`, plus `stored_rounds_digest()` |
| `test_fit_round_trips_through_its_own_payload` | payload carries `round`, `anchor_kind`, `scale_id` |
| `test_leaderboard_lists_provisional_entities_below_the_ranked_block` | per round |

### `tests/test_ratings.py` — new

- `test_each_round_is_fitted_only_from_its_own_games` — the core invariant:
  adding a game to round B leaves `fits/A.json` byte-identical.
- `test_root_games_influence_no_published_number` — a `_root` record with a
  registered hash appears in no round, no row, no digest.
- `test_round_order_is_name_ascending_and_stable_under_shuffled_discovery`.
- `test_a_round_without_the_global_anchor_uses_its_most_played_entity`, with a
  tied-appearance fixture asserting the name tiebreak is deterministic.
- `test_the_global_anchor_is_never_forced_into_a_round_that_did_not_play_it` —
  assert the round's table has no zero-game anchor entity and stays one
  component.
- `test_a_round_with_no_eligible_games_is_reported_unrated_with_its_exclusions`.
- `test_a_round_where_nobody_reaches_the_gate_ranks_nobody_and_still_renders`.
- `test_two_rounds_with_identical_games_get_different_scale_ids` — the
  anti-comparison invariant, in code.
- `test_a_round_split_by_the_era_filter_rates_one_era_and_reports_era_split`.
- `test_the_markdown_has_no_pooled_ranked_table` — guards the removal.
- `test_no_per_game_refit_path_exists` — `run_and_store` has no `update_ratings`
  parameter and `run_match`'s parser rejects `--update-ratings`. Same shape and
  same reason as the existing `tests/test_ratings.py:144`
  `test_no_incremental_rating_api_exists_to_diverge_from_the_rebuild`: a removed
  path that nothing asserts is removed grows back.
- `test_leaderboard_json_has_no_top_level_entities_key` — guards the schema
  break so a partial revert fails loudly.

### `tests/test_ratings_connectivity.py`

- `test_the_leaderboard_warns_and_groups_when_the_pool_is_split` → rescoped to
  one round's section.
- New `test_a_disconnected_round_annotates_only_its_own_section`.
- New `test_a_round_records_which_component_holds_its_anchor`.
- The `LoadedFit` refusal tests keep their shape; the payload fixture gains
  `round`.

### `tests/test_ratings_cache.py`

- `test_round_directories_include_loose_root_games` unchanged (default
  `include_root=True`); add `test_round_directories_can_exclude_root`.
- New `test_the_per_round_refit_reads_each_round_cache_exactly_once`.
- New `test_a_fit_file_for_a_removed_round_is_pruned`.

### `tests/test_ratings_lineage.py`

- Steps whose predecessor is absent from the round render `—`, not a number.

---

## Docs and skills

| File | Change |
| --- | --- |
| `.cursor/skills/update-leaderboard/SKILL.md` | Outputs table (`fits/<round>.json`, per-round `leaderboard.json`/`.md`); "Refit" section says every write refits **every round**; "Reading the output" gains anchor tiers, scale tokens, unrated rounds, empty ranked blocks; new rule "never compare a rating in one table to a rating in another"; changelog entry |
| `docs/arena/ratings.md` | New "Per-round fits" section replacing the pooled framing; anchor tiering table; connectivity scoped to a round; Files table; the `refit()` usage snippet returns `RoundFits` |
| `docs/arena/decision-rule.md` | "Verdict" changes from "Refit the whole pool, then read the contrast" to reading it inside the round both arms played; gate 1 gains "both arms are in the same round"; note that "cross-round baselines are not comparators" is now enforced by the data model rather than by discipline |
| `docs/arena/tournament.md:100,107` | The refit no longer scans `data/games/` as one pool; `leaderboard.md` is per round |
| `docs/research/experiment-protocol.md:29-30,42` | "the whole pool is refitted once after the round" → per round |
| `.cursor/skills/evaluate-bot-change/SKILL.md:31-36,74` | The `refit(); fit.delta(...)` snippet, and `--lineage` now being per-round |
| `.cursor/skills/run-measurement-round/SKILL.md:75` | Its banner says round-local ratings "are not comparable to `data/ratings/leaderboard.md`" — that file is now itself round-local, so the sentence must be rewritten, not deleted |
| `scripts/measure_heuristics.py:227-231` | `ROUND_LOCAL_BANNER`, same rewrite, and the snippet re-pointed at the shared fitter (step 11) |
| `README.md:85,94-108` | Drop `--update-ratings` from the smoke-match example; rewrite the leaderboard section |
| `docs/arena/match-runner.md:24` | Delete the `--update-ratings` sentence; say that rating is a separate `python -m arena.records.ratings` step, and that `--round <name>` now decides the storage directory |
| `docs/index.md` | Link this plan next to `ratings-refactor-plan.md` |
| `docs/arena/ratings-refactor-plan.md` | Historical record of the pooled design — add a pointer here, do not rewrite it |

The 28 `docs/research/measurements/*.md` reports carrying the old
"not comparable to `data/ratings/leaderboard.md`" banner are **not** edited.
They are dated records of what was published at the time; the template that
generates the sentence is what changes.

---

## Backward-incompatible changes, and who reads them

| # | Break | Who reads it | What they get |
| --- | --- | --- | --- |
| 1 | **`leaderboard.md` loses the single ranked table** | humans; `docs/arena/tournament.md:107`; `.cursor/skills/run-measurement-round/SKILL.md:75`; 28 historical measurement reports | No pooled ranking exists any more. Anyone who read "rank 3 overall" has no successor number, by design — that ranking pooled rounds that drift by more than the decision thresholds |
| 2 | **`leaderboard.json` loses its flat snapshot** — top-level `entities`, `anchor`, `rated_games`, `counts_digest`, `connected`, `components`, `seat_advantage` all gone | grep finds **no production reader**; only `tests/test_ratings_connectivity.py:165-177` and the docs tables | Test updated; a consumer written later against v1 fails on `version: 2` rather than silently reading a missing key |
| 3 | **`fit.json` moves to `fits/<round>.json`** | `io.load_fit`, `io.stored_counts_digest`, tests, any local tooling holding the path | `load_fit()` without a round raises and lists the rounds. The directory is derived and gitignored, so the fix is a refit |
| 4 | **`refit()` returns `RoundFits`, not `RatingFit`** | `arena/tournaments/competition.py:315`, `scripts/measure_heuristics.py:403`, `scripts/morpheus_cadence_evidence.py:191`, `.cursor/skills/evaluate-bot-change/SKILL.md:31` | `fit.delta(a, b)` becomes `fits["<round>"].delta(a, b)`. **A contrast whose two arms did not both play in one round now has no answer at all** — that is the intended behaviour change, and it is the same rule `decision-rule.md` already states in prose |
| 5 | **`--lineage` becomes per-round and refuses cross-round step deltas** | `cli.py`, `lineage.py`, the evaluate-bot-change skill, `tests/test_ratings_lineage.py` | The most visible loss of output: many steps that printed a delta now print `—`. Those deltas were the +46 Elo trap in its purest form — a step measured in round N against a predecessor measured in round N−1 |
| 6 | **`_root` games leave every published number** | anything citing a rating for one of the **15 entities that exist only in `_root`** (14 `morpheus@…` smoke hashes and `smoke@30a5f8f5da4a`) | Those entities vanish from all output. Both rows in the current provisional block are among them. 94 rated games and 4 connectivity groups leave the pool |
| 7 | **`--update-ratings` is removed** from `run_match` and `smoke_match` | anyone with `python -m arena.matches.run_match … --update-ratings` or `python scripts/smoke_match.py --update-ratings` in a script or in muscle memory; `README.md:85`; `docs/arena/match-runner.md:24` | The flag errors as unrecognised. Replacement is `python -m arena.records.ratings` after the game is stored. `run_tournament`'s `--no-ratings` is untouched — see [Removing `--update-ratings`](#removing---update-ratings) |
| 8 | **`run_match` stores into `data/games/<round>/`, not flat** | `arena/matches/run_match.py`, and anything reading `data/games/*.json` directly | An ad-hoc match now creates a round directory instead of adding to the excluded `_root` bucket. Existing `_root` records are left where they are |
| 9 | **`MissingAnchor` becomes nearly unreachable** | `arena/tournaments/competition.py:312-320` catches it to avoid losing a round's results | The tier-2 fallback means it can only fire when the anchor **bot** is unregistered. Keep the catch; note in the code why it is now rare |

## Out of scope

- Per-round `--min-games` overrides.
- Splitting an era-spanning round into two tables.
- Any cross-round aggregate — a meta-analysis over per-round fits is a
  different piece of statistics and needs its own design and its own evidence.
- Rewriting historical measurement reports.
- Migrating `_root` records into round directories. They stay where they are,
  excluded and documented.
