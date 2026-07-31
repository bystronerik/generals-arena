# Ratings refactor plan

Replace sequential elote Elo with a batch-fit rating model that can answer one
question: **did this bot change make the bot stronger?**

Status: **plan only, no code written.** Nothing in this document has been
implemented. Numbers marked *(measured)* come from probes against the current
`data/games/` (7261 records, 2026-07-31); probe scripts were throwaway and are
not committed.

---

## 0. Summary

| | Today | Target |
| --- | --- | --- |
| Model | sequential Elo per match (elote `EloCompetitor`, K=32) | Bradley–Terry + Davidson draws + shared seat term, MAP fit |
| Identity | `bot_id` | `(bot_id, content_hash)` |
| Order dependence | up to **250 Elo** spread over shuffles *(measured)* | none — integer sufficient statistics, unique convex optimum |
| Uncertainty | none | covariance matrix; SE, CI, `P(B > A)`, `games_to_resolve` |
| Seat effect | unmodeled, uncancelled (**+59 Elo** *(measured)*) | fitted shared parameter `β` |
| Draws | `tied()` no-op; 35.2% of games *(measured)* | fitted Davidson parameter `ν` |
| Anchor | none — scale floats | one entity pinned at exactly 1500.0 |
| Version traceability | none | committed registry + `refs/bot-versions/<hash>` |
| Incremental update | yes, diverges from rebuild | removed; every write refits |
| Dependency | `elote` (using `EloCompetitor`) | `numpy` at runtime; `elote` demoted to a test-only oracle |

---

## 1. Current-state audit

Each defect, with what it breaks for the "was this change good?" use case.

### Order dependence

**A1 — Sequential Elo is order-dependent by construction.**
[`arena/records/ratings.py:63`](../../arena/records/ratings.py:63) (`apply_game`) applies
`beat`/`lost_to`/`tied` one match at a time; each update depends on the ratings
that earlier matches produced.
*Measured:* replaying the same 7261 stored games in 5 different shuffles moves
ratings by up to **250.5 Elo** (`army_convey`: 1687.0 → 1937.5) and changes the
rank of **16 of 22** bots. Breaks hard requirement 1 outright — the leaderboard
is partly a readout of worker scheduling.

This was avoidable with the dependency already in `requirements.txt`: elote also
ships `BradleyTerryCompetitor`, a batch MLE fit that is order-independent to
**2.27e-13 Elo** *(measured)*. A1 is a usage choice, not a limitation of the
library. See §2 for why that class still does not carry the whole design.

**A2 — "Chronological rebuild" is a random order.**
[`arena/records/store.py:261`](../../arena/records/store.py:261) sorts by
`(finished_at, game_id)`. `finished_at` has one-second granularity
([`store.py:190`](../../arena/records/store.py:190)) and *(measured)* **90.4%** of stored
games share their second with at least one other game (max 11 in one second),
so ties are broken by the `uuid4` suffix in `game_id`
([`store.py:204`](../../arena/records/store.py:204)). The "deterministic chronological
rebuild" in [`ratings.py:226`](../../arena/records/ratings.py:226) is a random draw from
A1's distribution.

**A3 — Incremental and rebuild paths silently disagree.**
[`ratings.py:211`](../../arena/records/ratings.py:211) (`rate_stored_game`) loads the
persisted book, applies one match, and writes it back;
[`ratings.py:226`](../../arena/records/ratings.py:226) (`rebuild_from_games`) throws the
book away and replays everything. Path-dependent Elo means the two produce
different numbers from the same games, and nothing reconciles them.
[`run_match.py:73`](../../arena/matches/run_match.py:73) uses the first,
[`tournaments/competition.py:240`](../../arena/tournaments/competition.py:240) the second.

**A4 — Round reports carry a third, different Elo.**
[`scripts/measure_heuristics.py:216`](../../scripts/measure_heuristics.py:216)
(`round_leaderboard_snippet`) builds a fresh book with `skip_if_rated=False`
in list order. Every published round report under
`docs/research/measurements/` therefore shows Elo that disagrees with the
global leaderboard *and* with any other run of the same round.

### Identity

**A5 — Ratings key on `bot_id`, so every revision collapses into one number.**
[`ratings.py:57`](../../arena/records/ratings.py:57) (`get`). `bot_a_content_hash` is
written by [`worker.py:37`](../../arena/tournaments/worker.py:37) and
[`run_match.py:44`](../../arena/matches/run_match.py:44) but is never read by any rating
code. Directly breaks requirements 2 and 3: "did expand_plus improve?" has no
representable answer.

**A6 — `"unknown"` is a valid stored identity.**
[`fingerprint.py:187`](../../arena/records/fingerprint.py:187) returns the string
`"unknown"` on any `OSError`; both writers store it verbatim. Under a
hash-keyed model that silently pools unrelated programs into one rated entity.

**A7 — Hash cache can go stale inside a long-lived worker.**
[`fingerprint.py:182`](../../arena/records/fingerprint.py:182) is an unbounded
`lru_cache` keyed by directory. A pool worker that outlives an edit keeps
labelling matches with the pre-edit hash.

**A8 — No engine version anywhere.**
[`store.py:35`](../../arena/records/store.py:35) (`REQUIRED_FIELDS`). A
`competition-module` submodule bump (currently `9e3b9d1`) changes win
probabilities; nothing records it, so ratings from before and after a rules or
engine change pool into one number.

**A9 — The repo pin is not a version and does not detect a dirty tree.**
[`store.py:269`](../../arena/records/store.py:269) (`git_head_sha`) returns a short SHA
with no `--porcelain` check, so `bot_a_commit_or_tag` can name a commit whose
tree is not what ran. Already documented as not-a-bot-version in
[`game-record-schema.md:16`](game-record-schema.md).

**A10 — 0 of 7261 stored games carry a content hash** *(measured — all records
are schema 1 or 2)*. The identity field exists and has never been populated,
which is why wholesale regeneration is cheap.

### Statistics

**A11 — No uncertainty at all.**
[`ratings.py:31`](../../arena/records/ratings.py:31) (`LeaderboardRow`) and
[`ratings.py:93`](../../arena/records/ratings.py:93) (`leaderboard`) carry a point rating
and W/L/D. Breaks requirement 4: a keep/revert call needs an interval, and
a 20-Elo gap over 30 games is indistinguishable from noise with no way to say so.

**A12 — Seat advantage is real, unmodeled, and not cancelled by the grid.**
[`tournaments/competition.py:60`](../../arena/tournaments/competition.py:60) emits each
unordered pair once; `swap_sides` defaults to `False`
([`competition.py:164`](../../arena/tournaments/competition.py:164)) and
[`measure_heuristics.py:497`](../../scripts/measure_heuristics.py:497) hardcodes
`swap_sides=False`.
*Measured:* seat A wins **2575** of **4704** decisive games (54.7%, ≈6.5σ);
a joint fit that includes a seat term puts it at **+58.6 Elo**. Sequential Elo
credits that to whichever bot happened to be listed first.

**A13 — Draws are 35% of the data and carry no model.**
[`ratings.py:77`](../../arena/records/ratings.py:77) calls `a.tied(b)`, a no-op at equal
ratings. *(Measured)* **2557 / 7261 (35.2%)** games are draws and **every one of
them is a 1200-turn truncation** (draw count equals truncated count exactly;
cap per [`RULES.md:147`](../../RULES.md)). The model cannot distinguish "evenly
matched" from "both bots stall", and a third of all compute is informationally
half-wasted.

**A14 — Nothing bounds a perfect or near-empty record.**
No prior, no minimum-games gate: [`ratings.py:93`](../../arena/records/ratings.py:93)
lists every bot that ever played. *(Measured)* `cm_hunter` reaches the middle
of the table on **6 games**. Under a batch fit with only a weak prior the same
data pushes `blitz` to 2716 — the fix is the prior, not the model.

**A15 — K = 32 is an unjustified, invisible knob.**
Set by elote's class default; `initial_rating` is overridden to 1500 at
[`ratings.py:28`](../../arena/records/ratings.py:28) but K never is. It alone controls the
responsiveness/noise trade-off and never appears in any output.

### Pooling and artifacts

**A16 — No eligibility filter of any kind.**
[`ratings.py:234`](../../arena/records/ratings.py:234) rates whatever
[`store.list_game_paths:244`](../../arena/records/store.py:244) rglobs out of
`data/games/`, including `legacy/` and `parallel-smoke/`. `record.mode` is
**never checked** anywhere in the rating path. The classic/remote exclusion that
[`AGENTS.md:64,113`](../../AGENTS.md) mandates is enforced only by directory
convention and reviewer discipline.

**A17 — Outputs embed a timestamp, so identical inputs never produce identical
files.** [`ratings.py:118`](../../arena/records/ratings.py:118) and
[`ratings.py:177`](../../arena/records/ratings.py:177) both write `utc_now_iso()` into the
payload. Two rebuilds of the same games cannot be diffed.

**A18 — No committed record of what any version scored.**
[`.gitignore:13-14`](../../.gitignore) excludes all of `data/ratings/`. Correct for a
derived artifact, but combined with A5 it means the repo has never held a
reviewable statement of a bot version's strength.

**A19 — The test suite asserts the wrong property.**
[`tests/test_ratings.py:26`](../../tests/test_ratings.py:26) tests idempotence by
`game_id` — a much weaker property than order invariance, and one that A1 passes
while being badly broken.

---

## 2. Model choice

### Recommendation: Bradley–Terry + Davidson draws + seat term, MAP fit ("BTDS")

Parameters, all in Elo points on the standard scale `s = 400 / ln 10 ≈ 173.72`:

- `θ_e` — strength of rated entity `e = (bot_id, content_hash)`
- `β` — shared seat-A advantage (one scalar for the whole pool)
- `κ = log ν` — shared draw propensity (one scalar)

For a game with entity `i` in seat A and `j` in seat B, let
`d = (θ_i − θ_j + β) / s` and

```
u_a = exp(d/2)      u_b = exp(−d/2)      u_draw = exp(κ)
Z   = u_a + u_b + u_draw
P(a) = u_a / Z      P(b) = u_b / Z       P(draw) = u_draw / Z
```

This is Davidson's draw model (`ν·√(p_a p_b)`), which collapses to `exp(κ)`
because `√(u_a u_b) = 1` under this parameterization. It is exactly a
three-category multinomial logit with linear predictors `(d/2, −d/2, κ)`.

**Prior (MAP, not MLE):** `θ_e ~ N(1500, σ₀²)` with `σ₀ = 200`,
`β ~ N(0, 200²)`, `κ ~ N(0, 2²)`. In interpretable units, `σ₀ = 200` is worth
about **3 pseudo-games against the anchor** — negligible against a 200-game arm
(1.5% weight) but enough to keep an undefeated entity finite.

**Anchor:** one entity's `θ` is removed from the free parameter vector and held
at exactly **1500.0**. See §5 for which entity.

**Fit:** damped Newton with analytic gradient and Hessian.

**Uncertainty:** Laplace approximation — `Σ = H⁻¹` at the optimum, where `H` is
the Hessian of the penalized negative log-likelihood. Per-entity `SE = √Σ_ee`;
for a contrast, `Var(θ_B − θ_A) = Σ_AA + Σ_BB − 2Σ_AB`.

### Why this satisfies the five hard requirements

**R1 (order independence) — structurally, not incidentally.** Two properties
combine:

1. The likelihood depends on the games *only* through integer counts
   `n[i, j, outcome]` over **ordered** pairs (seat matters). Building that table
   is integer addition — exact, commutative, associative. The count table is
   therefore **bit-identical** for any input order, and a digest of it is
   included in the output for verification.
2. The penalized objective is strictly convex (log-partition of an exponential
   family plus a positive-definite quadratic), so it has a **unique** global
   optimum. The optimizer path cannot change the answer, only the last few bits.

Sequential Elo can satisfy neither; elote's `BradleyTerryCompetitor` satisfies
this requirement too *(measured: 2.27e-13 Elo)* — see the evaluation below. See
§8 C2 for the exact floating-point contract.

**R2 (identity is the hash).** The entity index is keyed on
`f"{bot_id}@{content_hash}"`. Nothing in the model knows about `bot_id` except
as a label; two hashes of one bot are as unrelated to the fit as two different
bots. This is what makes the lineage delta a *measurement* rather than a
restatement of the prior.

**R3 (ordered across hashes).** Lineage order comes from the committed registry
(§4), never from the fit and never from timestamps. The fit produces `θ` and
`Σ`; `lineage_deltas()` walks the registry's ordered `steps` and reports
`Δ = θ_{n+1} − θ_n` with its own SE from the covariance. Because lineage enters
only the *reporting* layer, R3 cannot contaminate R1.

**R4 (uncertainty first-class).** `Σ` is a first-class output, persisted with
the fit. `P(B stronger than A) = Φ(Δ / SE(Δ))` and `games_to_resolve()` fall
straight out of it (§5).

**R5 (git-traceable hashes).** Orthogonal to the model; see §4.

### Rejected: sequential Elo / Glicko-2 (status quo family)

Fails R1 by construction (A1: 250 Elo of order noise on real data). Glicko-2 and
TrueSkill do carry an RD/σ, so they would satisfy R4, but both are *filters*:
output depends on how games are bucketed into rating periods and, for TrueSkill,
on the message-passing schedule. Forcing everything into a single period to
recover order independence degrades Glicko-2 into a one-step approximation of
the very fit proposed here — strictly worse than performing the fit. Neither
provides a fitted shared seat term, and TrueSkill's draw margin is a fixed
hyperparameter rather than an estimate, which is unacceptable at a 35% draw rate
(A13).

### Rejected: Whole-History Rating

WHR *is* a batch joint MAP fit and would satisfy R1 given a fixed time index —
it is the strongest alternative. It is rejected because its entire purpose is
modeling **strength drift within one player over time**, and requirement 2
defines that away: a content hash is a frozen program whose true strength is
constant by construction. WHR would spend a hyperparameter (the Wiener drift
rate `w`) smearing each hash's games across a time axis, and it would reintroduce
sensitivity to timestamps that are 90.4% tied to the second (A2), in exchange for
modeling a phenomenon we have eliminated. It also lacks a seat term and Davidson
draws out of the box.

Put precisely: **BTDS is WHR with the drift set to zero, plus the two nuisance
parameters we actually need.** If engine-era drift later needs modeling (§7
open question 2), WHR's machinery is the right thing to revisit.

### Evaluated: elote's `BradleyTerryCompetitor`

`elote` 1.2.0 ships a real Bradley–Terry implementation
(`elote/competitors/bradley_terry.py`, 406 lines): a minorization–maximization
fit (Hunter 2004) over the whole connected component, re-seeded flat on every
call, geometric-mean normalized, with a phantom-opponent regularizer. It is a
genuine batch MLE, not a sequential updater, and it deserves a proper evaluation
rather than dismissal.

**What it gets right** *(all measured on the 7261 stored games)*:

| Property | Result |
| --- | --- |
| Order independence (R1) | **2.27e-13 Elo** max drift over 3 shuffles — satisfied |
| Undefeated entities finite | yes, via `reg` (virtual win+loss vs a unit-strength phantom) |
| Elo-comparable scale | yes, `400/ln 10` by default |
| Unique optimum | yes — concave likelihood, flat seed, no warm start |
| Cost of one batch fit | **8 ms** on the full 22-entity graph |
| Agreement with our solver | **0.004 Elo** under matched conditions (no draws, `reg→0`) |

That last row matters: under matched conditions the two fits are the same
estimator, which makes elote a usable **independent oracle** (see below).

**Where it cannot carry the design:**

1. **No uncertainty — the disqualifying gap (R4).** `_export_current_state`
   returns rating and W/L/T; there is no variance, no covariance, no standard
   error, and MM produces no Hessian as a by-product. The docs
   (`docs/source/bradley_terry.rst` in the upstream toctree) document none
   either. Every decision this system exists to make —
   `P(B > A)`, `CI₉₅(Δ)`, `games_to_resolve` — needs
   `Σ_AA + Σ_BB − 2Σ_AB`, i.e. the inverse observed information over the whole
   entity set. Obtaining it means writing the likelihood's Hessian ourselves,
   which is the bulk of `model.py`. Once that exists, the MM iteration is ~40
   lines we no longer need.
2. **Half-win draws only, on a pool that is 35% draws — and it compresses the
   scale.** `tied()` adds 0.5 to each side's head-to-head. *(Measured, our
   solver, identical prior, varying only the draw model and seat term:)*

   | Variant | Full spread | Bottom-9 spread | `garrison` → `expand_plus` |
   | --- | ---: | ---: | ---: |
   | half-win, no seat | 791 | 69 | 38 |
   | half-win, + seat | 820 | 89 | 71 |
   | Davidson, no seat | 1980 | 196 | 111 |
   | Davidson, + seat | **1994** | **222** | **168** |

   Half-win collapses the nine draw-heavy bots into a **69-Elo** band that
   Davidson resolves into **196**. For a system whose job is detecting ~25-Elo
   changes, a ~3× scale compression concentrated exactly among the bots that
   draw with each other is disqualifying. Under elote's own fit the same nine sit
   between 1462.6 and 1533.4 — a 71-Elo band containing most of the roster.
3. **No seat term.** Hunter's MM has a home-advantage variant; elote implements
   the plain model. The measured effect is **+50 to +59 Elo** depending on
   specification — real, and not something a post-hoc correction can absorb,
   because it must be estimated jointly with the strengths.
4. **Mean-centred scale, mostly fixable.** `_recalculate_ratings` normalizes to
   mean log-strength zero, so adding a 40-game punching bag shifts every existing
   rating by *(measured)* **+53.6 to +56.5 Elo**. Re-anchoring on
   `expander_python` afterwards is an exact translation and reduces that to
   **−2.2 to +0.7 Elo**. So this is *nearly* a non-issue — the residual ±2 Elo is
   the regularizer pulling toward a population mean that moved, which is the
   argument for centring our prior on the anchor instead.
5. **Serialization drops the match graph.** `_import_current_state` documents
   that opponent references cannot be restored, so a round-tripped competitor
   carries a frozen rating and **cannot refit**. Harmless for us — we always
   rebuild from `data/games/` — but it makes `to_state`/`from_state` a trap that
   the current `RatingBook` persistence pattern would walk straight into.
6. **Per-result refit is the wrong shape for hash-keyed identity.** The public
   API refits on every `beat()`/`tied()`: *(measured)* 4.6 ms/game, ~0.6 min for
   the current 7261 games. But the cost is `O(games × n² × iterations)` and under
   requirement 2 the entity count `n` grows with **every content hash forever**,
   so it degrades superlinearly precisely as the identity scheme does its job.
   One batch fit costs 8 ms; the count-table design pays that once.

**Verdict: adopt the model, not the class.** elote's BT settles requirement 1 —
I was wrong to imply a batch fit had to be written from scratch for that — but it
cannot express requirement 4 at all, and requirements 4, the seat term, and the
draw model all modify the same likelihood. Writing that likelihood once, with
its gradient and Hessian, yields all three plus the MM fit's guarantees.

### Dependency: `elote` becomes a test-only oracle

- **Runtime:** remove `elote` from `requirements.txt`; add **`numpy`** (already
  present at 2.5.1 via the submodule — make it explicit). SciPy is *not*
  required: the solver needs only `np.linalg.cholesky` / `solve` / `inv` on a
  matrix of a few hundred rows.
- **Tests:** keep `elote` as a dev dependency for **T15**, an independent
  cross-check of our solver against `BradleyTerryCompetitor` under matched
  conditions (draws disabled, seat term forced to 0, both regularizers → 0).
  *(Measured: agreement to 0.004 Elo on a synthetic 8-entity, 3360-game set —
  see T15.)* This is a genuinely valuable test: it validates our Newton
  implementation against a separately-written MM implementation of the same
  estimator, which no self-consistency test can do.

If you would rather not carry a dependency purely for one test, the fallback is
a closed-form two-entity check (BT with one pair has the analytic solution
`Δ = s·ln(w/l)`); say the word and T15 changes accordingly.

---

## 3. Target design

### Module layout

```
arena/records/
  fingerprint.py            unchanged hashing; "unknown" becomes a hard error
  registry.py               NEW  — bot version registry (§4)
  store.py                  GameRecord v4
  reporting.py              leaderboard table gains CI + provisional columns
  ratings/                  NEW package, replaces ratings.py
    __init__.py             public API re-exports
    policy.py               eligibility rules + Prior/Policy dataclasses
    counts.py               GameRecord[] -> canonical CountTable (integers)
    model.py                BTDS log-likelihood, gradient, Hessian (pure math)
    fit.py                  Newton solver -> RatingFit (estimates + covariance)
    lineage.py              registry -> ordered steps -> step deltas
    io.py                   fit.json / leaderboard.{json,md} read+write
    cli.py                  python -m arena.records.ratings
```

`model.py` has no IO and no `arena` imports — it is testable against synthetic
count tables alone, which keeps the recovery tests fast (§6).

### Public API

```python
# policy.py
@dataclass(frozen=True) class Policy:
    modes: tuple[str, ...] = ("competition",)
    engine_version: str | None = None      # None = current era from HEAD submodule
    require_registered: bool = True
    include_self_play: bool = True         # informs β and κ only
    min_games_display: int = 30

@dataclass(frozen=True) class Prior:
    mean: float = 1500.0
    sigma: float = 200.0
    seat_sigma: float = 200.0
    draw_sigma: float = 2.0

def eligible(record: GameRecord, policy: Policy, registry: Registry) -> bool
def rejection_reason(record, policy, registry) -> str | None

# counts.py
@dataclass(frozen=True) class CountTable:
    entities: tuple[str, ...]                    # canonical sorted order
    cells: tuple[Cell, ...]                      # (i, j, wins_a, wins_b, draws)
    digest: str                                  # sha256 over the canonical form
    excluded: dict[str, int]                     # reason -> count

def count_table(games, *, policy, registry) -> CountTable

# fit.py
def fit_ratings(counts: CountTable, *, prior: Prior, anchor: str) -> RatingFit

class RatingFit:
    entities: list[str]
    anchor: str
    seat_advantage: Estimate          # .value, .se
    draw_log_nu: Estimate
    solver: SolverReport              # iterations, max|grad|, converged

    def rating(entity) -> float
    def se(entity) -> float
    def interval(entity, level=0.95) -> tuple[float, float]
    def record(entity) -> tuple[int, int, int]        # W, L, D
    def games(entity) -> int
    def provisional(entity) -> bool                   # games < policy.min_games_display

    def delta(a, b) -> Delta                          # .value .se .ci .p_stronger
    def p_stronger(b, a) -> float                     # Φ(Δ / SE(Δ))
    def games_to_resolve(a, b, *, target_se: float) -> int
    def expected(a, b, *, seat_a: bool = True) -> tuple[float, float, float]

    def to_state() -> dict ;  @classmethod from_state(dict) -> RatingFit

# lineage.py
def steps(bot_id, registry) -> list[Step]             # ordered, may repeat a hash
def lineage_deltas(bot_id, fit, registry) -> list[StepDelta]
def lineage_table_lines(bot_id, fit, registry) -> list[str]
```

`games_to_resolve` uses the model's own per-game Fisher information for the
contrast, which for a direct head-to-head is

```
I(d) = ¼ · [ (p_a + p_b) − (p_a − p_b)² ] / s²        per game
SE(Δ) after n games ≈ s / √( n · ¼ · [(p_a + p_b) − (p_a − p_b)²] )
```

Sanity values: evenly matched, no draws → `SE(Δ) = 347.4/√n` (n=100 → 35 Elo).
Evenly matched at the observed 35% draw rate → `SE(Δ) = 430.9/√n`
(n=100 → 43 Elo; n=1150 → 12.7 Elo). Draws cost about **24% more games** for the
same precision — not catastrophic, and cheap to buy (matches average 3.11 s
*(measured)*, so 1150 games ≈ **5.4 min** wall clock at 11 jobs).

### On-disk state

`data/ratings/fit.json` (gitignored, derived) — **no timestamp in the payload**,
so identical inputs give a byte-identical file (fixes A17):

```json
{
  "version": 1,
  "anchor": "expander_python@0f3a91cc21de",
  "anchor_rating": 1500.0,
  "prior": {"mean": 1500.0, "sigma": 200.0, "seat_sigma": 200.0, "draw_sigma": 2.0},
  "policy": {"modes": ["competition"], "engine_version": "9e3b9d1…",
             "include_self_play": true, "min_games_display": 30},
  "counts_digest": "sha256:…",
  "solver": {"iterations": 7, "max_abs_grad": 3.1e-12, "converged": true},
  "excluded": {"mode_not_competition": 0, "unregistered_hash": 0, "engine_mismatch": 0},
  "seat_advantage": {"value": 58.61, "se": 4.12},
  "draw_log_nu": {"value": 1.5408, "se": 0.031},
  "entities": [
    {"entity": "blitz@ab12cd34ef56", "bot_id": "blitz", "content_hash": "ab12cd34ef56",
     "rating": 1873.44, "se": 21.03, "games": 640,
     "wins": 310, "losses": 90, "draws": 240, "provisional": false}
  ],
  "covariance": {"order": ["…"], "lower_triangle": [ … ]}
}
```

Covariance is stored as a flat lower triangle in `entities` order; at 300
entities that is ~45k floats (~1 MB JSON), acceptable. `counts_digest` lets any
caller check whether a refit is needed without redoing it.

`data/ratings/leaderboard.{json,md}` — presentation only. Markdown columns:
`Rank | Entity | Rating | 95% CI | Games | W | L | D | Prov.`
Provisional entities (below `min_games_display`) are listed **below** the
ranked block, not interleaved. `updated_at` may appear in the Markdown header
only, never in `fit.json`.

`data/ratings/cache/<round>.counts.json` — per-round integer count tables keyed
by a digest of `(filename, size, mtime)` over that round dir. Count tables are
exactly additive, so merging caches is order-independent by construction. This
makes refit `O(#rounds)` rather than `O(#games)`.

### GameRecord changes (schema v4)

| Field | Change | Why |
| --- | --- | --- |
| `schema_version` | `3` → `4` | new required fields |
| `bot_a_content_hash` / `_b` | optional → **required**, non-empty, never `"unknown"` | A6: identity must not silently degrade |
| `engine_version` | **added** (required) — `competition-module` submodule SHA | A8: era separation |
| `round` | **added** (required) — round name | A16: eligibility must not depend on parsing a path |
| `bot_a_commit_or_tag` / `_b` | **removed** | A9: not a version; the registry now holds the real commit per hash |
| everything else | unchanged | |

Writers that must change: [`telemetry.record_from_match_result`](../../arena/records/telemetry.py),
[`worker.run_one_worker:50`](../../arena/tournaments/worker.py:50),
[`run_match.run_and_store:53`](../../arena/matches/run_match.py:53).

### Removed API

- `RatingBook`, `LeaderboardRow`, `rate_stored_game`, `rebuild_from_games`,
  `load_book`, `save_book` — all of `arena/records/ratings.py`.
- `run_match.py --update-ratings` keeps its name but now means "refit after
  storing", not "apply one Elo update".
- `measure_heuristics.round_leaderboard_snippet` is rewritten to call
  `fit_ratings` on the round's own count table, and its output is explicitly
  labelled *round-local, anchor-free* to stop it being confused with the global
  leaderboard (A4).

---

## 4. Hash registry design

### Path and gitignore

**`data/bot_versions/<bot_id>.json`**, one file per bot.

*Verified:* `git check-ignore data/bot_versions/x.json` exits non-zero today.
The existing rules ([`.gitignore:11-14`](../../.gitignore)) are **file globs scoped to
`data/games/` and `data/ratings/`**, not directory excludes, so a sibling
directory under `data/` needs **no carve-out and no negation pattern**. The
`.gitignore` change is therefore one comment line documenting that
`data/bot_versions/` is deliberately committed, plus a test that asserts it
stays that way (§6 T11) — a negation entry without a preceding ignore would be a
no-op and worse than the test.

One file per bot rather than one per hash, so that `git log -p
data/bot_versions/expand_plus.json` *is* the bot's improvement history.

### Format

```json
{
  "bot_id": "expand_plus",
  "versions": [
    {
      "content_hash": "ab12cd34ef56",
      "first_seen_at": "2026-07-31T20:14:03Z",
      "git_commit": "b67125f1c0…",
      "git_dirty": true,
      "closure_ref": "refs/bot-versions/ab12cd34ef56",
      "closure_tree": "4b825dc642…",
      "files": [
        {"path": "bots/_common/wire.py", "sha256": "…", "blob": "…"},
        {"path": "bots/expand_plus/agent.py", "sha256": "…", "blob": "…"}
      ]
    }
  ],
  "steps": [
    {"seq": 1, "content_hash": "ab12cd34ef56", "at": "2026-07-31T20:14:03Z"},
    {"seq": 2, "content_hash": "cd34ef5678ab", "at": "2026-08-01T09:02:11Z"},
    {"seq": 3, "content_hash": "ab12cd34ef56", "at": "2026-08-01T13:40:55Z",
     "note": "revert to seq 1"}
  ]
}
```

`files` carries the `bot_source_closure` list with both the per-file SHA-256 the
hash is built from and the git blob SHA, so a registry entry alone proves which
bytes were rated.

### Entities vs steps — and the revert case

This is the resolution of a genuine tension in the requirements (§8, C1).
Requirement 2 says each hash is one improvement step; requirement 3 asks what
happens when a hash reappears after a revert. Those cannot both be a bijection.

- **`versions`** is a *set*, unique by hash, append-only, never mutated. This is
  what the fit rates. A reverted hash is the **same program**, so its games from
  before and after the revert **pool into one entity** — which is correct, and
  the reason the model gets stronger over time rather than restarting.
- **`steps`** is an ordered *sequence* that may repeat a hash. This is the
  lineage. Step 3 above reports as "returns to seq 1's entity;
  Δ vs seq 2 = −34 ± 19 Elo, P(seq 2 stronger) = 0.96".

Lineage order comes from `steps` — an append-only log written when a program
actually ran — never from `finished_at`, never from git. It is therefore
independent of everything R1 cares about.

### Write path

**The match runner registers at hash time**, in the parent process only:

- [`tournaments/competition.py:189`](../../arena/tournaments/competition.py:189) already
  hashes the whole roster once before the pool starts — registration hooks in
  there, exactly once per round.
- [`run_match.run_and_store:44`](../../arena/matches/run_match.py:44) registers for
  single matches.
- Pool workers **never write** the registry; they assert the hash is registered
  and fail the match if not. This avoids concurrent writes entirely.
- Manual CLI for out-of-band use:
  `python -m arena.records.registry --register <bot…> | --verify | --diff A B | --files H`.

Registration is idempotent: an existing hash is a no-op, and an existing entry is
never mutated.

**Pre-commit hook rejected.** Matches routinely run from dirty trees, so a
commit hook would register versions that never played a game while missing the
ones that did — exactly backwards from what the registry is for.

### Dirty working trees

The content hash is over the **working tree**, so a hash can exist that lives in
no commit. Handled in two parts:

1. **Anchor the closure in git's object database under a ref.** At registration:
   `git hash-object -w` each closure file → build a tree → `git commit-tree` →
   `git update-ref refs/bot-versions/<hash>`. The ref makes the objects
   reachable (gc-safe) and gives every hash a permanently diffable name. This
   also survives the repo's rebase-onto-main workflow, which rewrites the SHAs
   recorded in `git_commit` but leaves these independent commit objects intact.
2. **Record the context:** `git_commit` (HEAD at registration) and
   `git_dirty: true`, plus a stdout warning. A `--strict` mode (recommended
   default for published rounds) refuses to register a dirty closure, forcing a
   commit first.

Sharing the refs:

```bash
git push origin 'refs/bot-versions/*:refs/bot-versions/*'
```

```bash
git fetch origin 'refs/bot-versions/*:refs/bot-versions/*'
```

### "What changed between hash X and hash Y"

Primary (works for dirty-tree hashes, survives rebases):

```bash
git diff refs/bot-versions/<hash_a> refs/bot-versions/<hash_b>
```

```bash
git ls-tree -r --name-only refs/bot-versions/<hash_a>
```

Wrapper that resolves either mechanism and prints the closure delta first:

```bash
python -m arena.records.registry --diff <hash_a> <hash_b>
```

Fallback when refs were not fetched, using the recorded commits and closure:

```bash
git diff <commit_a>..<commit_b> -- $(python -m arena.records.registry --files <hash_a> <hash_b>)
```

### Lineage vs git order when they disagree

**`steps` is authoritative for lineage; git is authoritative only for diffs.**
They disagree when (a) a hash was registered from a dirty tree and committed
later or never, (b) history was rewritten — live here, since the workflow is
rebase-onto-main — leaving `git_commit` unreachable, or (c) work registered on a
side branch before its parent landed.

`--verify` reports entries whose `git_commit` is unreachable or whose
`closure_ref` is missing as `git-unresolvable`, and **never changes ratings**.
A hash whose diff cannot be resolved is still a perfectly valid rated entity —
its identity comes from content, not from git.

---

## 5. Decision procedure for `evaluate-bot-change`

**Setup.** Baseline hash `A` (the bot's current lineage head), candidate hash
`B`, a fixed **opponent panel** `P` of ≥5 bots spanning the rating range and
including the anchor, a pinned `--round-seed`, and identical map seeds for both
arms.

**Seat balance is required for decision arms.** Because `β` is fitted, large
exploratory rounds do *not* need `--swap-sides` (which saves 50% of the compute —
a change from the current protocol's implicit assumption). But if the candidate
always sits in seat A and the baseline in seat B, `β` and the contrast are
aliased within that subset. Decision arms must play **both seat orders equally**.

**Gate (any failure → `unproven`, with the reason).**

1. `A` and `B` both registered; every game `mode == "competition"`; identical
   `engine_version` across both arms; identical opponent panel and seed set.
2. **≥ 200 games per arm**, and **≥ 30 games per (arm, opponent)**.
3. **≥ 60 decisive (non-draw) games per arm.** Below that → `unproven — no
   signal`, and the report names which opponents were 100% draws.

**Verdict.** Refit the whole pool, then compute `Δ = θ_B − θ_A`, `SE(Δ)` from
the covariance, `CI₉₅ = Δ ± 1.96·SE`, and `P(B > A) = Φ(Δ/SE)`.

| Verdict | Rule |
| --- | --- |
| **improvement** | `P(B > A) ≥ 0.95` **and** `CI₉₅.low > +10` |
| **regression** | `P(B > A) ≤ 0.05` **and** `CI₉₅.high < −10` |
| **no change (proven flat)** | `CI₉₅` lies entirely within `±25` |
| **unproven** | anything else — report `games_to_resolve(A, B, target_se=12.75)` |

Never decide from leaderboard rank; always from the pairwise contrast.

**Why these numbers.** 10 Elo ≈ 1.4 percentage points of winrate at parity —
below the useful floor for bot iteration. 25 Elo ≈ 3.6 pp. Reaching a `±25` CI
needs `SE = 12.75`, i.e. **≈1150 games per arm** at the observed 35% draw rate —
about **5.4 minutes** of wall clock at 11 jobs *(measured: 3.11 s/game)*. So
200 games is the floor for *any* verdict (`SE ≈ 30`), and "proven flat" is a
verdict you can actually afford to buy.

**Replaces** the current blanket rule at
[`evaluate-bot-change/SKILL.md:54`](../../.cursor/skills/evaluate-bot-change/SKILL.md)
("both arms draw every game → unproven"). Under Davidson, draws are *not*
uninformative — they pin a rating band — so the new rule is quantitative
(gate 3) rather than all-or-nothing.

### Pooling policy

Eligible for the fit iff **all** hold:

| Rule | Enforcement |
| --- | --- |
| `mode == "competition"` | explicit check in `policy.eligible` (fixes A16 — never directory-implied) |
| both content hashes present, non-empty, `!= "unknown"` | schema v4 makes them required; `fingerprint` raises instead of returning `"unknown"` |
| both hashes registered in `data/bot_versions/` | `policy.require_registered` |
| `engine_version` equals the fit's era | mismatches are counted in `excluded`, never silently pooled |
| `schema_version >= 4` | older records were deleted; a stray one is rejected loudly |

Specific answers:

- **Classic and remote games: still excluded**, as [`AGENTS.md:64,113`](../../AGENTS.md)
  requires. Now enforced three ways: the mode filter, a loader that only ever
  scans `data/games/`, and a test (§6 T9). Today it is enforced by convention only.
- **Truncated games: included, as draws.** A 1200-turn truncation *is* a draw
  under [`RULES.md:147`](../../RULES.md), and *(measured)* 100% of current draws are
  truncations. Excluding them would delete 35% of all data and bias the pool
  toward matchups that happen to resolve.
- **Self-play (same hash both seats): included.** Such games contribute nothing
  to any `θ` (the strength term cancels) but are a *clean* estimator of `β` and
  `κ`, uncontaminated by strength. This makes `--include-self` rounds genuinely
  useful for the first time.
- **Competition mode only** — no other mode is eligible, and no other mode
  currently exists in `data/games/` *(measured: 7261/7261 are `competition`)*.

### Minimum-games gate

Entities with `games < 30` are marked `provisional`, listed below the ranked
block, and are **not** eligible as a decision baseline. They still **participate
in the fit** — dropping them would change every other entity's rating and break
the "pure function of the game set" contract.

### Incremental vs rebuild

**Incremental update does not survive.** `rate_stored_game` is deleted (A3).
Every write path refits.

Cost, *(measured)* today: loading all 7261 records takes **0.92 s**; the fit
itself over 22 entities is milliseconds. At **100×** (≈726k games) the JSON
load dominates at roughly 90 s plus inode pressure. Mitigations, in order:

1. **Per-round counts cache** (§3). Refit re-aggregates only rounds whose digest
   changed, so a refit after one new round stays under ~1 s regardless of total
   history.
2. Optional round compaction: fold a finished round dir into one
   `games.jsonl`. Deferred — per-file writes are what makes the process pool
   safe; compaction would be a separate end-of-round step (migration step 8).

---

## 6. Migration steps

Ordered; each is independently committable. Work on `main`, no branches.
No backwards compatibility: `data/games/` and `data/ratings/` are deleted
outright, and no code path reads schema < 4 or hashless records.

| # | Step | Deletes | Commit contains |
| --- | --- | --- | --- |
| 1 | **Clear the decks.** Delete `data/games/**` and `data/ratings/**` (keep `.gitkeep`). Create `data/bot_versions/`. Add the `.gitignore` comment documenting that it is deliberately committed. | 7261 game records, all rating snapshots | gitignore comment, `.gitkeep`s |
| 2 | **Registry module.** `arena/records/registry.py` + CLI (`--register/--verify/--diff/--files`), git-ref anchoring, `--strict`. No consumers yet. | — | module + tests T10–T12 |
| 3 | **GameRecord v4.** Required content hashes, add `engine_version` + `round`, drop `*_commit_or_tag`; `fingerprint` raises instead of returning `"unknown"`; fix the stale-cache window (A7). Update `telemetry.py`, `worker.py`, `run_match.py`, and `docs/arena/game-record-schema.md`. | `"unknown"` fallback | schema + writers + doc |
| 4 | **Wire registration in.** Parent-side registration in `run_tournament` and `run_and_store`; workers assert-only. | — | runner changes |
| 5 | **New ratings package.** `arena/records/ratings/{policy,counts,model,fit,lineage,io,cli}.py`; delete `arena/records/ratings.py`; move `elote` from `requirements.txt` to a test-only dependency and add `numpy`; update `reporting.leaderboard_table_lines` for CI/provisional columns; rewrite `measure_heuristics.round_leaderboard_snippet`. | `RatingBook`, `rate_stored_game`, `rebuild_from_games`, runtime use of `elote` | model + tests T1–T9, T13, T15 |
| 6 | **Counts cache.** Per-round cache + digest invalidation. Optional at current scale; required before 100×. | — | cache layer |
| 7 | **Regenerate.** Register every roster bot, then run one full anchored round. Cost below. | — | registry entries + round report only (games stay gitignored) |
| 8 | **Docs and skills sync.** Rewrite `docs/arena/ratings.md`; new `docs/arena/bot-version-registry.md`; update `docs/arena/game-record-schema.md`, `docs/research/experiment-protocol.md`, `docs/index.md`, `AGENTS.md`, `.cursor/skills/evaluate-bot-change/`, `.cursor/skills/update-leaderboard/`. | stale elote references | docs |

### Regeneration cost

*(measured: 3.11 s mean per match, 11 physical cores)*

| Round | Games | Serial bot-hours | Wall clock @ 11 jobs |
| --- | --- | --- | --- |
| Full 23-entity round robin, 50 games/pair, no seat swap | 12,650 | 10.9 h | **~60 min** |
| Same with `--swap-sides` | 25,300 | 21.9 h | ~2.0 h |
| Reduced: 30 games/pair, no seat swap | 7,590 | 6.6 h | ~36 min |
| One decision arm (`±25` CI, seat-balanced) | 1,150 | 1.0 h | **~5.4 min** |

**Recommendation:** one ~60-minute full round without seat swap (`β` is fitted,
so balance is not needed for identifiability), then per-experiment decision arms
of ~1150 games each. Budget **≈1.5 bot-hours of wall clock** for the whole
migration, plus ~6 minutes per subsequent bot change.

---

## 7. Test plan

Suite must stay under 3 s ([`AGENTS.md:82`](../../AGENTS.md)). Recovery tests sample
**count tables** from a multinomial rather than simulating individual games, so
a "2000 games per pair" test costs microseconds.

| # | Test | Asserts |
| --- | --- | --- |
| **T1** | **Order invariance (property test)** | Build a synthetic game list; shuffle 20× with distinct seeds; the `CountTable` digest is **byte-identical** every time and every fitted rating, SE, `β`, `κ` and covariance entry matches to `< 1e-9`. Also asserts no incremental API exists to diverge. |
| **T2** | Count-table canonicality | Cells are emitted in sorted entity order; counts are exact integers; merging per-round caches in any order gives the same digest as one-shot aggregation. |
| **T3** | **Anchor stability** | Anchor rating is exactly `1500.0` in every fit; adding an entity with zero games moves no other rating by `> 1e-9`; adding a fully disconnected clique does not shift the anchored component. |
| **T4** | **Recovery of known strengths** | Generate counts from the model with known `θ` spread over ±300; recovered `θ` within 3·SE, RMSE `< 15` Elo. |
| **T5** | **Seat-advantage recovery** | Data generated with `β = +60` on a deliberately seat-imbalanced design: `β̂` within 3·SE of 60, and entity ratings unbiased — while a control fit with `β` forced to 0 shows the expected bias, proving the term earns its place. |
| **T6** | Draw model | `ν̂` reproduces the simulated draw rate; a 100%-draw pair leaves both entities at the prior mean with large SE. |
| **T7** | **Sparse / undefeated** | A 6–0 entity gets a *finite* rating bounded by the prior and `provisional = True`; a 0-game entity sits at exactly 1500.0 with `SE = σ₀`; removing a provisional entity from *display* does not change any other rating. |
| **T8** | Probability + sample size | `p_stronger` equals `Φ(Δ/SE(Δ))`; `games_to_resolve(target_se)` returns an `n` such that a refit with `n` extra simulated games hits the target SE within ±20%. |
| **T9** | **Pooling policy** | Non-competition mode, unregistered hash, `"unknown"` hash, and mismatched `engine_version` are each excluded *and counted in* `excluded`; truncated draws are included; self-play is included and moves only `β`/`κ`; no code path can reach `data/classic_games/` or `data/remote_games/`. |
| **T10** | **Registry round-trip (hash → closure → diff)** | In a temp git repo: register a bot → registry lists the closure with matching digests → `refs/bot-versions/<hash>` resolves → edit one file → new hash, new ref → `git diff refA refB` shows exactly that file → revert → hash equals the first, `versions` is unchanged, `steps` gains a third entry pointing at the first hash. |
| **T11** | Registry is committed | `git check-ignore data/bot_versions/x.json` exits non-zero (guards against a future `data/*` rule silently hiding the registry). |
| **T12** | Lineage | Steps are ordered and independent of timestamps; the revert case reports a delta against the *previous step*, not the previous hash occurrence; an unreachable `git_commit` is flagged by `--verify` **without changing any rating**. |
| **T13** | Reproducible artifacts | Two fits over the same games produce a byte-identical `fit.json` (no timestamps in the payload — the A17 regression guard). |
| **T14** | Determinism environment | The fit is unchanged with `OPENBLAS_NUM_THREADS=1` vs unset. If this proves flaky, `fit.py` pins the thread count and the test asserts the pin (see C2). |
| **T15** | **Independent solver oracle** | Under matched conditions — no draws, seat term forced to 0, `BradleyTerryCompetitor.configure_class(reg=1e-6, tol=1e-12)` and our prior `σ → ∞` — our Newton fit and elote's MM fit agree on a synthetic well-connected set to `< 0.01` Elo after mean-centring. *(Measured on an 8-entity / 3360-game set: max diff **0.004 Elo**, both at RMSE 13.6 vs the generating truth.)* Validates our solver against a separately-written implementation of the same estimator. Keeps `elote` as a dev dependency — see §9 q10. |

---

## 8. Conflicts in the requirements

Flagged rather than silently resolved.

**C1 — "each hash is one improvement step" vs. the revert case.**
Requirement 2 makes hash → step a bijection; requirement 3 asks what happens
when a hash reappears. Both cannot hold. **Proposed resolution:** split
*entity* (unique by hash, what gets rated, games pool across reappearances) from
*step* (ordered position in the lineage, may repeat an entity). §4. Say the word
if you would rather a reverted hash be a *distinct* entity with its own rating —
that is defensible if you suspect the engine or opponent pool drifted between
the two eras, but it throws away pooled games and makes "same code, same
rating" false.

**C2 — "bit-identical" is achievable for the counts, not literally for the
floats.** Integer sufficient statistics are exactly reproducible, and the
optimum is unique, but floating-point reduction order inside BLAS can differ
across thread counts. **Proposed contract:** (a) `CountTable` digest identical,
(b) fitted values equal to `< 1e-9`, (c) published values rounded to 2 dp and
therefore literally identical. Accept, or should `fit.py` pin BLAS threads to 1
and claim true bit-identity at some speed cost?

**C3 — "pure function of the set of stored games" is not quite the whole
truth.** The fit is a pure function of *(games, registry, policy, prior,
anchor)*. Editing the registry or bumping `σ₀` changes ratings without any new
game. **Proposed resolution:** all five inputs are recorded inside `fit.json`
(§3) so any rebuild is verifiable, and the order-invariance guarantee is stated
over the games argument with the rest held fixed.

**C4 — shrinkage toward the lineage parent would help, and would import
lineage into the fit.** Shrinking each hash toward its parent's rating (rather
than toward 1500) would cut the games needed per step substantially — most bot
edits are small. But then ratings become a function of the registry's lineage,
so a registry edit changes ratings, which is a soft violation of C3's spirit.
**Default: no parent shrinkage.** Worth revisiting if decision arms feel
expensive.

---

## 9. Open questions

1. **Anchor identity.** `expander_python` is the natural anchor, but it lives in
   `competition-module/…`, outside `bots/` — so
   [`bot_source_closure`](../../arena/records/fingerprint.py:131) cannot resolve its
   imports (it searches only the bot dir and `bots/`), and its real version is
   the submodule SHA. Options: **(a)** copy it to `bots/_anchor/` and freeze it
   (registry-visible and hashable, but diverges from upstream); **(b)** anchor on
   the submodule SHA and treat any anchor change as an era boundary. Which?
   The same question applies to the `cm_*` benchmark bots.
2. **Engine eras.** I propose adding `engine_version` (submodule SHA) to every
   record and refusing to pool across eras, with a `--era` flag to refit a past
   one. Confirm — this means a submodule bump invalidates the whole leaderboard
   and costs one ~60-minute regeneration round.
3. **Draws are timeouts, and they cost ~24% more games.** All 2557 draws are
   1200-turn truncations. A land-margin tiebreak at the cap would convert most
   into decisive results, but that is a **deviation from `RULES.md:147`**. Off
   the table, or worth a rating-only `--tiebreak` scoring mode (games still
   stored as draws, scored as margin wins)?
4. **Cross-bot closure coupling.** `proteus` imports `aegis`/`blitz`/`boom`/
   `metro`, so editing `aegis` forks proteus's lineage even though proteus's own
   code did not change. Behaviourally correct, but should such steps be labelled
   `inherited` in the lineage report so they are not mistaken for a proteus
   experiment?
5. **`min_games_display = 30`** — confirm. Provisional entities stay in the fit
   either way; this only controls the ranked block and baseline eligibility.
6. **Thresholds in the skill body.** `evaluate-bot-change` currently forbids
   Composer from inventing a threshold. The new rule embeds `±10`, `±25`, `200`,
   `60`, `0.95`. Do those live in the skill body, or in a
   `docs/arena/decision-rule.md` that the skill links so there is one source of
   truth?
7. **Historical reports.** `docs/research/measurements/round1–4.{json,md}`
   contain Elo tables from the deleted model. Leave them as history, add a
   banner, or delete?
8. **Prior strength `σ₀ = 200`** (≈3 pseudo-games). Weaker means less bias on
   small samples and wilder undefeated entities; stronger means a more stable
   leaderboard and shrunken deltas. Happy with 200, or should it be tuned by
   cross-validation on the regenerated round?
9. **Registry ref sharing.** `refs/bot-versions/*` needs an explicit push/fetch
   refspec (§4). Add it to `.git/config` as a documented setup step, or keep the
   refs purely local and rely on `git_commit` + `files` for anyone else?
10. **Keep `elote` as a dev dependency for T15?** It buys one genuinely strong
    test — our Newton solver checked against an independently written MM
    implementation of the same estimator (measured agreement: 0.004 Elo) — at
    the cost of carrying a dependency used nowhere in the runtime. Keep it,
    or drop it and settle for the analytic two-entity check
    (`Δ = s·ln(w/l)`), which is weaker but dependency-free?
