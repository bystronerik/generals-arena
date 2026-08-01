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

Both arms must be measured in the same rounds. This is measured, not
theoretical: in the macaria hunt evaluation (2026-08-01), two **byte-identical**
macaria programs — `macaria@80f3047ac205` from rounds `macaria-r1`/`macaria-r2`
vs `macaria_base@862ac0189a1a` from rounds `macaria-hunt-*` — fitted at
**+46.07 ± 22.01 Elo, P = 0.982**, though no code differed. Round-to-round
drift of that size exceeds the 10 and 25 Elo thresholds below, so a contrast
against a baseline measured in earlier rounds measures the rounds, not the
change. Re-measure the baseline in the same rounds as the candidate, with
matched seed lists and alternated seats; `bots/macaria_base/` exists as a
frozen copy of pre-change macaria for exactly this purpose.

## Gate

Any failure means `unproven`, with the reason named.

1. `A` and `B` are both registered; every game is `mode == "competition"`;
   both arms share one `engine_version`; both arms use the same opponent panel
   and seed set.
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

Refit the whole pool, then read the contrast:

```python
delta = fit.delta(baseline_entity, candidate_entity)   # theta_B - theta_A
delta.value, delta.se, delta.ci, delta.p_stronger
```

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
