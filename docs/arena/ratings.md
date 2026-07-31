# Ratings

`arena/records/ratings/` fits a **batch** rating model over every eligible
stored game: Bradley–Terry, plus Davidson draws, plus one shared seat term,
estimated jointly as a MAP fit with a covariance matrix.

It replaced sequential Elo (`elote.EloCompetitor`, K=32) because the system
exists to answer one question — *did this bot change make the bot stronger?* —
and sequential Elo could not. Replaying the same 7261 stored games in five
different orders moved ratings by up to **250 Elo** and changed the rank of 16
of 22 bots, so the leaderboard was partly a readout of worker scheduling.

## The model

Every parameter is in Elo points on the standard scale `s = 400 / ln 10 ≈ 173.72`.

| Parameter | Meaning |
| --- | --- |
| `θ_e` | strength of rated entity `e = bot_id@content_hash` |
| `β` | shared seat-A advantage, one scalar for the whole pool |
| `κ = log ν` | shared draw propensity, one scalar |

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

**Anchor:** [`bots/cm_expander/`](../../bots/cm_expander/agent.py) is removed
from the free parameters and held at exactly **1500.0**. It wraps the same
upstream `ExpanderAgent` the competition module ships, lives in the repo,
hashes cleanly, and carries a "do not retune" comment. Every published rating
is a statement relative to that one program.

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
which is correct and is why the estimate keeps improving instead of restarting
— but a **new lineage step**. See `arena/records/ratings/lineage.py`.

## Eligibility

A game enters the fit only if **all** of these hold. Every rejection is counted
in `fit.json`'s `excluded`, never dropped silently.

| Rule | Rejection reason |
| --- | --- |
| `mode == "competition"` | `mode_not_eligible` |
| `schema_version >= 4` | `schema_too_old` |
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
  one; `--all-eras` disables the filter and is unsound.

Entities with fewer than **30 games** are `provisional`: listed below the
ranked block and not eligible as a decision baseline. They still participate in
the fit — dropping them would change every other rating.

## Flow

1. Store every match under `data/games/<round>/<game_id>.json`.
2. Refit. **There is no incremental update**; the old incremental path and the
   rebuild silently disagreed, because a path-dependent estimator has no single
   right answer.

```bash
python -m arena.records.ratings --print
```

```bash
python scripts/leaderboard.py --print
```

Useful flags: `--lineage <bot>` prints one bot's improvement history,
`--dry-run` fits without writing, `--era <sha>` refits a past engine era,
`--no-cache` re-reads every game.

## Files

| Path | Contents |
| --- | --- |
| `data/bot_versions/<bot>.json` | version registry — **committed** |
| `data/ratings/fit.json` | estimates, covariance, policy, prior, counts digest |
| `data/ratings/leaderboard.json` | ranked snapshot |
| `data/ratings/leaderboard.md` | same snapshot as Markdown |
| `data/ratings/cache/<round>.counts.json` | per-round integer count tables |

Everything under `data/ratings/` is derived and gitignored. **`fit.json`
contains no timestamp**, so identical games produce a byte-identical file and
two rebuilds can be diffed. `counts_digest` lets a caller tell whether a refit
is needed without doing one.

## Using it

```python
from arena.records.ratings.cli import refit

fit = refit()
delta = fit.delta("expand_plus@ab12cd34ef56", "expand_plus@cd34ef5678ab")
delta.value, delta.se, delta.ci, delta.p_stronger
fit.games_to_resolve(a, b, target_se=12.75)
```

**Never decide from leaderboard rank; always from the pairwise contrast.** The
thresholds live in one place: [decision-rule.md](decision-rule.md).

## Cost

The fit itself is milliseconds. Reading the games dominates, and per-round
count caches mean a refit re-aggregates only the rounds whose files changed.

## Related

- [decision-rule.md](decision-rule.md) — the keep/revert thresholds
- [bot-version-registry.md](bot-version-registry.md) — hash → source
- [game-record-schema.md](game-record-schema.md) — schema v4
- [tournament.md](tournament.md) — seat policy and round layout
- [ratings-refactor-plan.md](ratings-refactor-plan.md) — the audit and design
  rationale behind all of the above
