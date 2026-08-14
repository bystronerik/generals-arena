# Ratings

`arena/records/ratings/` fits a **batch** rating model — Bradley–Terry, plus
Davidson draws, plus one shared seat term, estimated jointly as a MAP fit with a
covariance matrix — **once per round**, over that round's games only.

It replaced sequential Elo (`elote.EloCompetitor`, K=32) because the system
exists to answer one question — *did this bot change make the bot stronger?* —
and sequential Elo could not. Replaying the same 7261 stored games in five
different orders moved ratings by up to **250 Elo** and changed the rank of 16
of 22 bots, so the leaderboard was partly a readout of worker scheduling.

## Per-round fits

**There is no pooled fit and no global ranked table.** Each round under
`data/games/<round>/` gets its own independent solve, published as its own
section and its own file.

A round is a design: one roster, one seed list, one seat policy, played in one
sitting. Pooling rounds treats their games as exchangeable, and measurement says
they are not:

- Two **byte-identical** `macaria` programs measured in different rounds fitted
  **+46.07 ± 22.01 Elo, P = 0.982** apart (2026-08-01).
- The M6 `morpheus-rs` round disagreed with four re-measurements by **7.4 sigma**
  (2026-08-09).

Both are larger than the thresholds [decision-rule.md](decision-rule.md) decides
on, so a pooled column presented drift as strength. Per-round fitting removes the
column instead of annotating it.

What follows from that:

| | |
| --- | --- |
| Scope of a fit | one round |
| Contrasts | both arms must be in the same round, or there is no answer |
| Rank | restarts at 1 in every round, and is never a result |
| Cross-round arithmetic | possible with a subtraction, licensed by nothing |
| Cost | 13 Newton solves = 0.077 s, against 0.002 s pooled |

Each table carries a **`scale_id`** —
`sha256(round \0 anchor \0 anchor_kind \0 counts_digest)[:12]`. The round name is
inside the digest, so two rounds with byte-identical games still get different
tokens. That gives a mechanical rule — *same scale, comparable; different scale,
not* — in place of prose a reader can talk themselves out of.

Be honest about the limit: rendering cannot make comparison impossible. Any
absolute number can be subtracted from any other. What the design buys is that no
*published artifact* makes the comparison for the reader, and no rank spans two
rounds.

**The `_root` bucket is excluded.** Loose `data/games/*.json` carry a `round`
field no directory backs — 49 distinct values over 102 files — so they are not a
round and cannot be fitted as one. `run_and_store` now defaults to
`data/games/<round>/` so new ad-hoc matches land somewhere the rating layer can
see.

### Anchoring, per round

The anchor is resolved in three tiers, and the tier is published as
`anchor_kind`:

| Tier | Condition | Anchor | `anchor_kind` |
| --- | --- | --- | --- |
| 1 | the global anchor played ≥ 1 eligible game in this round | `cm_expander@<latest>` at 1500.0 | `global` |
| 2 | it did not, but the round has eligible games | the round's most-played entity, ties by entity key ascending | `round_local` |
| 3 | the round has no eligible games | none; the round is not fitted | — |

**The global anchor is never forced into a round that did not play it.** Forcing
it in via `count_table(extra_entities=[anchor])` makes it a zero-game singleton
component; the round's real entities then form a second component located by the
prior alone. Measured on a 10-game round where `b` beat `c` 9–1, that moved the
`b → c` contrast from **−151.5 ± 157.2** to **−220.0 ± 191.3** — the prior doing
work an anchor should be doing.

A round-local anchor changes one line of provenance, not the table.

### Unrated rounds

A round that cannot be fitted gets `status: "unrated"`, a reason, and a stub
section with no table. It is **never silently dropped**: a round missing from the
report is indistinguishable from a round nobody ran, and the actionable content
is exactly what survives — the stored count and the `excluded` breakdown.

| `reason` | Cause | Today |
| --- | --- | --- |
| `no_eligible_games` | every record rejected by the policy | 3 rounds, 739 records, all `unregistered_hash` |
| `no_games_in_era` | the era filter emptied the round | 0 rounds |
| `solver_failed` | the solve raised | 0 rounds |

A non-converged solve is *not* `solver_failed`: it publishes, with
`solver.converged` false and the residual printed next to the table. Today
`morpheus-castle-r1` stalls at `max|grad| 5.9e-9` against a 1e-9 tolerance,
because that round has no draws and `κ` runs toward −∞ against its prior alone.

One unrated round never aborts the others. The CLI exits 1 only when *every*
round is unrated, or the anchor bot is unregistered.

## The model

Every parameter is in Elo points on the standard scale `s = 400 / ln 10 ≈ 173.72`.

| Parameter | Meaning |
| --- | --- |
| `θ_e` | strength of rated entity `e = bot_id@content_hash` |
| `β` | shared seat-A advantage, one scalar per round |
| `κ = log ν` | shared draw propensity, one scalar per round |

For a game with entity `i` in seat A and `j` in seat B, let
`d = (θ_i − θ_j + β) / s`:

```
u_a = exp(d/2)      u_b = exp(−d/2)      u_draw = exp(κ)
Z   = u_a + u_b + u_draw
P(a) = u_a / Z      P(b) = u_b / Z       P(draw) = u_draw / Z
```

**Prior (MAP, not MLE):** `θ_e ~ N(1500, 200²)`, `β ~ N(0, 200²)`,
`κ ~ N(0, 2²)`. `σ₀ = 200` is worth about three pseudo-games against the
anchor — 1.5% weight against a 200-game arm, but enough to keep an undefeated
entity finite rather than running off to infinity.

**Anchor:** one entity per round is removed from the free parameters and held at
exactly **1500.0**. [`bots/cm_expander/`](../../bots/cm_expander/agent.py) is the
global anchor: it wraps the same upstream `ExpanderAgent` the competition module
ships, lives in the repo, hashes cleanly, and carries a "do not retune" comment.
Rounds it played are statements relative to that one program; rounds it did not
are anchored locally — see [Anchoring, per round](#anchoring-per-round).

**Fit:** damped Newton with an analytic gradient and Hessian.
**Uncertainty:** Laplace — `Σ = H⁻¹` at the optimum.

### Why each piece is there

- **Batch, not sequential.** The likelihood reads the games only through
  integer counts over ordered pairs, and the penalized objective is strictly
  convex, so the fit is a pure function of the *set* of games and has a unique
  optimum. Order cannot touch it.
- **Content-hash identity.** Two revisions of one bot are as unrelated to the
  fit as two different bots, which is what makes a lineage delta a measurement
  rather than a restatement of the prior.
- **Fitted draws.** 35% of games are 1200-turn truncations. Splitting them
  half-and-half compresses the draw-heavy tail of the table about **3×**,
  exactly among the bots that draw with each other — fatal for detecting
  ~25-Elo changes.
- **Fitted seat term.** Seat A wins about 55% of decisive games. The estimate
  ranges over 30–99 Elo depending on the draw model it is fitted alongside,
  which is precisely why a fixed post-hoc offset cannot stand in for it.

## Identity

The rated entity is `bot_id@content_hash`, never the bot id alone. A hash is a
12-hex SHA-256 over the bot's source closure
([game-record-schema.md](game-record-schema.md)); the registry maps it back to
files, a commit, and a diffable git ref
([bot-version-registry.md](bot-version-registry.md)).

A hash that reappears after a revert is the **same entity** — its games pool,
which is correct and is why the estimate keeps improving instead of restarting.

There is no lineage table. Registry-step lineage was removed on 2026-08-14: it
needs one `bot_id` with two registered hashes **in one round**, and no round in
the store has ever had that. The decision workflow freezes the baseline under a
separate bot id (`joe_prev`, `joe_base`), exactly as
[decision-rule.md](decision-rule.md) prescribes, so the two arms are two bot ids
and a lineage table renders every step as `not_in_this_round`. Read the contrast
between the two frozen entities instead.

## Eligibility

A game enters its round's fit only if **all** of these hold. Every rejection is
counted in that round's `excluded`, never dropped silently.

| Rule | Rejection reason |
| --- | --- |
| `mode == "competition"` | `mode_not_eligible` |
| `schema_version >= 5` | `schema_too_old` |
| both content hashes present and not `"unknown"` | `unknown_content_hash` |
| both hashes registered in `data/bot_versions/` | `unregistered_hash` |
| `engine_version` matches the era being fitted | `engine_mismatch` |

Specifically:

- **Classic and remote games are excluded**, as
  [`AGENTS.md`](../../AGENTS.md) requires — now by the mode filter and a loader
  that only ever scans `data/games/`, not by directory convention.
- **Truncated games are included, as draws.** A 1200-turn truncation *is* a
  draw under [`RULES.md`](../../RULES.md). Excluding them would delete a third
  of the data and bias the pool toward matchups that happen to resolve.
- **Self-play is included.** It contributes nothing to any `θ` (the strength
  terms cancel) but is a clean, strength-free estimator of `β` and `κ`.
- **Engine eras never pool.** A submodule bump moves win probabilities, so
  pooling across one would be a silent correctness bug. `--era` refits a past
  one. There is deliberately no flag that disables the filter.

The era is a policy value applied identically to every round, so one publish is
always one era. A round whose stored games **span** a bump is rated on the current
era's subset, reports `era_split` with the excluded count, and is *not* split into
two per-era tables — halving a round by era produces two broken designs with
neither seat balance nor pair coverage, under round names no manifest backs. The
remedy the report names is to re-run the round.

Entities with fewer than **30 games in that round** are `provisional`: listed
below the ranked block and not eligible as a decision baseline. They still
participate in the fit — dropping them would change every other rating.

The gate counts games **per round**, which has two visible consequences. One
entity can be ranked in one round and provisional in another; that is correct,
because each row is a statement about evidence gathered inside that round. And a
round where nobody clears 30 ranks nobody — today `m7-contention-probe` (22
games), its `-loaded` twin (22), and `morpheus-bootstrap` (12 per entity). Their
sections say so in a sentence and print the full provisional table. A 22-game
probe *is* a probe and should read like one.

`--min-games` stays a single global flag applied to every round. A per-round
override is deliberately absent: it would let an operator dial a round into
rankability after seeing the result.

## Connectivity

Eligibility decides which games enter. **Connectivity decides which resulting
ratings may be compared**, and it is a separate question. It is computed on each
round's own count table, so a split in one round annotates that round's section
and no other. Every non-empty round in the store today is connected.

The likelihood reads only differences `θ_i − θ_j`, for pairs that played. So it
is flat along "add a constant to everybody" once per connected component of the
co-play graph. One component and the anchor pins that constant; two components
and the second one is set by the prior alone. The prior also makes the Hessian
invertible, so a split pool converges cleanly and hands back ordinary-looking
intervals for comparisons that rest on no evidence whatsoever.

`fit.connected` and `fit.components` expose the grouping. Across groups
`fit.delta` returns `comparable=False` with an infinite SE and `P = 0.50`, the
CLI warns, and that round's section grows a `Group` column plus a warning block —
because a single ranked list asserts that every row is comparable to every
other. The warning also names `anchor_component`: when the anchor lands in a
minority group, most rows are prior-located rather than measured, and a reader has
to be told which group is the measured one. The remedy is games between the
groups; `fit.games_to_resolve` says how many. See
[decision-rule.md](decision-rule.md).

The grouping is stored in the round's fit file, and a fit read back with
`load_fit(round=...)` applies the same refusal — the guard survives the file. A
payload written before the field existed cannot say which entities share games
(the games are not in the payload), so a reloaded legacy fit refuses **every**
cross-entity contrast until a refit rewrites the file with the grouping in it.

The usual cause is a *whole roster* forking at once — a change to a file every
bot's closure contains — followed by a round that plays only the new hashes.
A normal decision arm shares an opponent panel and stays connected by
construction.

## Flow

1. Store every match under `data/games/<round>/<game_id>.json`.
2. Refit. **There is no incremental update**; the old incremental path and the
   rebuild silently disagreed, because a path-dependent estimator has no single
   right answer. Every write refits **every** round.

```bash
python -m arena.records.ratings --print
```

```bash
python scripts/leaderboard.py --print
```

Useful flags: `--list-rounds` prints the round set and exits, `--round <name>`
(repeatable) filters what is *reported*, `--dry-run` fits without writing,
`--era <sha>` refits a past engine era.

`--round` is a read filter, never a write filter. A write filter would mean
`leaderboard.json` either loses the unlisted rounds or carries them forward from
the previous file, and carrying forward is an incremental update in everything but
name. Refitting all 16 rounds costs 0.077 s of solve time, so there is nothing to
buy.

## Files

| Path | Contents |
| --- | --- |
| `data/bot_versions/<bot>.json` | version registry — **committed** |
| `data/ratings/fits/<round>.json` | one rated round: estimates, covariance, policy, prior, counts digest, scale |
| `data/ratings/leaderboard.json` | every round's status and rows, format version 2 |
| `data/ratings/leaderboard.md` | the same, as one document of per-round sections |

Everything under `data/ratings/` is derived and gitignored. **No fit file contains
a timestamp**, so identical games produce byte-identical files and two rebuilds can
be diffed. The `Updated:` line in `leaderboard.md` is the only clock in any output.

Fits are one file per round rather than one keyed file, because a fit file carries
the covariance matrix: bundling every round would make one new game rewrite every
round's covariance, and `load_fit` parse 16 matrices to answer one contrast. Split,
a new round changes exactly one file.

The writer **prunes**. Any `fits/*.json` whose round is no longer under
`data/games/` is deleted: a stale *published fit* is a table for a round that no
longer exists with nothing on it saying so.

There is **no count cache**. Rounds were cached to `data/ratings/cache/` until
2026-08-14; the cache saved about one second on a full refit and cost a file
signature, a rules digest, a prune step and a staleness API that could disagree
with the games on disk. Reading all 9,000 games costs 1.3 s and cannot go stale.
Reinstate a cache when a cold refit costs seconds.

`stored_counts_digest(round=...)` tells a caller whether one round needs refitting;
`stored_rounds_digest()` answers it for the whole store in one read.

## Using it

```python
from arena.records.ratings.cli import refit

fits = refit()                       # RoundFits: one independent fit per round
fits.names                           # every round, name ascending
fits.result("joe-r4").anchor_kind    # 'global' or 'round_local'

fit = fits["joe-r4"]                 # KeyError if that round is unrated
delta = fit.delta("joe@ab12cd34ef56", "joe@cd34ef5678ab")
delta.value, delta.se, delta.ci, delta.p_stronger
delta.comparable          # False when no chain of games links the two
fit.games_to_resolve(a, b, target_se=12.75)
```

**A contrast whose two arms did not both play in one round has no answer at all.**
That is the intended behaviour, and it is the rule
[decision-rule.md](decision-rule.md) already stated in prose.

**Never decide from a rank; always from the pairwise contrast.** The thresholds
live in one place: [decision-rule.md](decision-rule.md).

## Cost

The fit itself is milliseconds — 13 per-round solves total 0.077 s on a
9,056-game store. Reading the games dominates (0.944 s cold for 9,897 files), and
per-round count caches mean a refit re-aggregates only the rounds whose files
changed: 0.093 s to read all 17 tables from cache.

## Related

- [decision-rule.md](decision-rule.md) — the keep/revert thresholds
- [bot-version-registry.md](bot-version-registry.md) — hash → source
- [game-record-schema.md](game-record-schema.md) — schema v4
- [tournament.md](tournament.md) — seat policy and round layout
- [ratings-refactor-plan.md](ratings-refactor-plan.md) — the audit that produced
  the batch fit; describes the pooled design it replaced sequential Elo with
- [per-round-ratings-plan.md](per-round-ratings-plan.md) — why the pool itself
  went away, with the measurements
