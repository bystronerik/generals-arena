# Decision rule

**One source of truth for the thresholds.** Skills and reports link here rather
than restating numbers; nothing else in the repo may invent one.

The question is always the same: *is candidate `B` stronger than baseline `A`?*
The answer always comes from the **pairwise contrast**, never from leaderboard
rank.

## Setup

- Baseline hash `A` — the bot's current lineage head. It must not be
  `provisional` (< 30 games).
- Candidate hash `B`.
- A fixed **opponent panel** `P` of at least 5 bots spanning the rating range
  and including the anchor (`cm_expander`).
- A pinned `--round-seed`, and identical map seeds for both arms.
- `--seat-policy alternate`. Seat balance is required for a decision arm:
  randomization balances only *in expectation* (over 1150 games the seat-A
  count is 575 ± 17), so if the candidate drifts toward one seat, `β` and the
  contrast are aliased within that subset. Alternation is exactly 50/50 by
  construction and additionally cancels map difficulty within each matched
  pair.

```bash
python -m arena.tournaments.competition \
  bots/<candidate>/run.sh bots/<panel...>/run.sh \
  --round <name> --games-per-pair 50 --round-seed 7 \
  --seat-policy alternate --strict-versions
```

## Cross-round baselines are not comparators

Both arms must be measured in the same round. **This is now enforced by the data
model rather than by discipline:** ratings are fitted one round at a time, so
`fits["<round>"].delta(A, B)` needs both arms inside that round and a contrast
across two rounds has no answer to give. See
[ratings.md](ratings.md#per-round-fits).

This is measured, not theoretical: in the macaria hunt evaluation (2026-08-01),
two **byte-identical**
macaria programs — `macaria@80f3047ac205` from rounds `macaria-r1`/`macaria-r2`
vs `macaria_base@862ac0189a1a` from rounds `macaria-hunt-*` — fitted at
**+46.07 ± 22.01 Elo, P = 0.982**, though no code differed. Round-to-round
drift of that size exceeds the 10 and 25 Elo thresholds below, so a contrast
against a baseline measured in earlier rounds measures the rounds, not the
change. Re-measure the baseline in the same rounds as the candidate, with
matched seed lists and alternated seats. The usual way to do that is a frozen
copy of the pre-change bot under `bots/`, kept only for the duration of the
comparison (`bots/macaria_base/` was that copy for the run above; it has since
been removed).

## A round is not evidence until it replicates

**Every gate below passed on a round whose result was wrong by 300 Elo.** The
M6 morpheus-rs contrast (2026-08-09) fitted `+425.06 ± 22.77, P = 1.0000` over
1,152 games per arm, 973 decisive, registered hashes, one engine version,
shared panel, alternated seats. Re-measured four ways — including a replay of
that round's own seeds under its own job count — the same two programs sit at
**0.504 ± 0.042** against the round's 0.867, a 7.4-sigma disagreement. Cause
unidentified; seeds, parallelism, external load and the programs are each
ruled out by experiment. Detail:
[morpheus-rs-m6-replication.md](../research/measurements/morpheus-rs-m6-replication.md).

Two rules follow, and they bind hardest on **deadline-driven bots** — anything
that thinks until a wall-clock budget expires, where strength is a function of
how much compute the host happened to spare:

- **Replicate before publishing.** A contrast that will be written down needs a
  second round, separately scheduled. The gates below cannot substitute for it;
  they all passed.
- **Host state is an experimental variable and is currently unrecorded.** The
  round manifest captures the roster, seeds, seat policy, engine and job count,
  and nothing about what else the machine was doing. Until it does, note it by
  hand, and treat a round whose host state is unknown as unpublishable.

Related: the same phenomenon at 1/9th the size is the "cross-round baselines
are not comparators" section above, which was measured at +46 Elo between
byte-identical programs.

## Gate

Any failure means `unproven`, with the reason named.

1. `A` and `B` are both registered; **both arms are in the same round**; every
   game is `mode == "competition"`; both arms share one `engine_version`; both
   arms use the same opponent panel and seed set.
2. **`A` and `B` are in the same connectivity group** — `fit.comparable(A, B)`.
   They need not have played each other, but some chain of games must link
   them, or their difference is prior rather than evidence (see below).
3. **≥ 200 games per arm**, and **≥ 30 games per (arm, opponent)**.
4. **≥ 60 decisive (non-draw) games per arm.** Below that the verdict is
   `unproven — no signal`, and the report names which opponents were 100%
   draws.

Gate 2 is enforced in code, not by discipline: `fit.delta` returns
`comparable=False` with an infinite SE across groups, so no threshold below
can be met and the verdict is necessarily `unproven`. Sharing an opponent
panel satisfies it automatically, which is why a decision arm cannot trip it —
the failure mode it guards against is a *whole roster* forking at once.

## Verdict

Refit, then read the contrast **inside the round both arms played**:

```python
from arena.records.ratings.cli import refit

fits = refit()                                          # one fit per round
fit = fits["<round>"]                                   # the round of the arms
delta = fit.delta(baseline_entity, candidate_entity)    # theta_B - theta_A
delta.value, delta.se, delta.ci, delta.p_stronger
```

There is no pooled fit to read instead. Asking for a round that is unrated raises
and names the rated rounds; asking for a contrast whose arms are not both in the
round raises a `KeyError` on the missing entity. Both refusals are the point:
gate 1 has no soft failure mode any more.

| Verdict | Rule |
| --- | --- |
| **improvement** | `P(B > A) ≥ 0.95` **and** `CI₉₅.low > +10` |
| **regression** | `P(B > A) ≤ 0.05` **and** `CI₉₅.high < −10` |
| **no change (proven flat)** | `CI₉₅` lies entirely within `±25` |
| **unproven** | anything else — report `fit.games_to_resolve(A, B, target_se=12.75)` |

## Why these numbers

- **10 Elo** ≈ 1.4 percentage points of winrate at parity — below the useful
  floor for bot iteration.
- **25 Elo** ≈ 3.6 pp. Reaching a `±25` confidence interval needs
  `SE(Δ) = 12.75`.
- At the observed 35% draw rate, `SE(Δ) ≈ 430.9/√n`, so `SE = 12.75` costs
  **≈1150 games per arm** — about **5.4 minutes** of wall clock at 11 jobs.
  Draws inflate the standard error 24% at fixed `n`; since `n` scales as `SE²`,
  that is ~54% more games than a draw-free pool would need.
- **200 games** is therefore the floor for *any* verdict (`SE ≈ 30`), and
  "proven flat" is a verdict you can actually afford to buy.

## Connectivity: no contrast without a chain of games

The model only ever sees differences `θ_i − θ_j`, and only for pairs that
played. The likelihood is therefore flat along "add a constant to everybody",
once per connected component of the co-play graph. One component: the anchor
pins that constant, and every contrast is measured. Two components: the second
constant is pinned by nothing but the `N(1500, 200)` prior.

The prior also makes the Hessian invertible, so a split pool **converges
cleanly and reports normal-looking intervals**. Nothing downstream can tell
that the curvature came from three pseudo-games rather than from evidence.

This has bitten once, for real. Deleting a stderr write that ran after the
last move forked every bot's content hash; the next round played only the new
hashes; the two generations shared zero games. The lineage table then reported
**+474 Elo, CI [+379, +570], P(better) = 1.00** for the anchor bot — whose only
changed file could not alter a single action.

The guard:

- `fit.connected` and `fit.components` expose the grouping;
- `fit.delta(a, b)` across groups returns `comparable=False`, `se = inf`, and
  `P = 0.50`, so no verdict above can be reached;
- the leaderboard prints a warning block and a `Group` column, because a
  single ranked list asserts that every row is comparable;
- `fit.games_to_resolve(a, b, …)` still answers, from zero precision — the
  remediation is **games between the groups**, never comparing anyway.

## Draws are not uninformative

Under the Davidson draw model a draw pins a rating band rather than saying
nothing, so the rule is quantitative (gate 3) rather than all-or-nothing. The
older blanket rule — "both arms draw every game → unproven" — is replaced by
the ≥60-decisive-games gate.

## Reporting a decision

State, in this order: the two entity keys, `Δ ± SE`, `CI₉₅`, `P(B > A)`, the
games per arm, and the verdict. If the verdict is `unproven`, state
`games_to_resolve`. Never quote a rank.

## Related

- [ratings.md](ratings.md) — the model the contrast comes from
- [bot-version-registry.md](bot-version-registry.md) — what `A` and `B` name
- [`.cursor/skills/evaluate-bot-change/`](../../.cursor/skills/evaluate-bot-change/SKILL.md)
