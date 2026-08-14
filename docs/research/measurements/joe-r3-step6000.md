# Joe r3 — step 6000 vs 5000 vs 3500 (2026-08-14)

Third arena round for `bots/joe`, measuring the newly exported EMA
checkpoint (**step 6000** of `joe-M-vast-20260813-0213`) against both of
its references in one round: **step 5000**, the previous export and the
network the Rust sibling plays, and **step 3500**, the strongest
checkpoint a full round had measured (r2).

640 games, zero failures, engine `9e3b9d13cca5`, `--seat-policy alternate
--strict-versions`, explicit matched seed lists.

## Verdict: **improvement**, and it is marginal

| Contrast | Elo | CI₉₅ | P | Verdict |
| --- | ---: | --- | ---: | --- |
| **5000 → 6000** (primary) | **+62.7 ± 26.7** | **[+10.3, +115.1]** | **0.9905** | **improvement** |
| 3500 → 5000 | +279.9 ± 33.1 | [+215.0, +344.9] | 1.0000 | improvement |
| 3500 → 6000 | +342.7 ± 34.3 | [+275.5, +409.8] | 1.0000 | improvement |

The primary contrast clears the `CI₉₅.low > +10` floor **by 0.3 Elo**. It
is a pass, but one that would flip to `unproven` on a different seed
draw, and the rule's replication requirement is unmet (one round). Treat
it as "probably a real but small gain", not as a settled number.

Head-to-head, on matched maps played both ways:

| Pair | Record | Decisive win-rate |
| --- | :---: | ---: |
| 6000 vs 5000 | 93W 66L 1D | 58.5% |
| 6000 vs 3500 | 102W 17L 1D | 85.7% |
| 5000 vs 3500 | 96W 19L 5D | 83.5% |

Each fitted contrast reproduces its raw head-to-head win-rate (58.5% →
+60 raw vs +62.7 fitted; 83.5% → +281 vs +279.9), which is the check that
carries weight.

> **Correction (2026-08-14, r4).** This paragraph originally also offered
> the transitivity of the three *fitted* contrasts (+279.9 + 62.7 =
> +342.6 vs a "directly measured" +342.7) as a consistency check. It is
> not one. A Bradley-Terry fit gives each entity a single strength, so
> `Δ(a,c) = Δ(a,b) + Δ(b,c)` is an algebraic identity that holds however
> the games fell — and the "+342.7" was itself a fitted contrast, not a
> measurement. Only the raw win rates can test the chain. The verdict
> above is unaffected; the evidence offered for it was weaker than
> stated. See [joe-r4-step10000.md](joe-r4-step10000.md).

## Progress is decelerating

> **Superseded (2026-08-14, r4).** The rate stopped falling: 6000 → 10000
> came in at ~71 Elo per 1000 iterations, holding the ~66 measured here
> rather than continuing down. Read this section as "the rate fell from
> ~200 to ~70 and levelled off", not as evidence for stopping the run.
> See [joe-r4-step10000.md](joe-r4-step10000.md).

| Interval | Iterations | Elo | Elo per 1k iters |
| --- | ---: | ---: | ---: |
| 3000 → 3500 (r2) | 500 | +104.5 | ~209 |
| 3500 → 5000 | 1500 | +279.9 | ~187 |
| 5000 → 6000 | 1000 | +62.7 | ~63 |

The last 1000 iterations bought about a third of the earlier rate.
In-training eval cannot see this — vs-random was 96.7% / 97.9% / 98.8% at
3500 / 5000 / 6000, all at ceiling. If the question is when to stop the
Phase 4 run, this table is the evidence, not the eval curve.

## Do not read joe's rating or rank from the leaderboard

**The pool can no longer measure joe's absolute level.** In this round
the three joe arms went **239W–1L** against the anchor-connected pool
(80 games each vs morpheus-rs and cm_expander). That is quasi-complete
separation: the likelihood is nearly flat in "how far above morpheus-rs
is joe", so the `N(1500, 200)` prior sets the level, and the level a joe
entity gets depends mostly on **how many anchor-linked games it happened
to play**.

Two symptoms in the current fit, both artifacts rather than findings:

- Byte-identical step-3500 reads **2584.5** as r2's `joe@b7cd053f905b`
  (200 anchor-linked games) and **2045.5** as r3's
  `joe_base@8af4c88d3ec6` (80 anchor-linked games) — a 539 Elo gap for
  the same weights.
- That r3 entity sits *below* `morpheus-rs` (2310.6) in the table while
  beating it **39–1 in this very round**, which is impossible as a
  statement about strength.

The joe family is nominally in the anchored connectivity component, so
`comparable()` returns True and the machinery reports finite intervals —
the guard that catches a *split* pool does not catch a *separated* one.
What stays valid is exactly what the decision rule already says to use:
the pairwise contrast between two arms measured in the same round, which
here rests on 400 direct joe-vs-joe games and matches the raw win rates.

Consequence for future rounds: measuring joe against the existing roster
is now spending compute to learn nothing. Joe needs a **checkpoint
ladder** — archived joe checkpoints as the opponent panel, which is how
the paper's own reference-Elo eval works. The panel bots are worth
keeping only as a thin connectivity tie, and even that tie no longer
identifies a level.

## Design

- Arms, all frozen copies playing greedy argmax: `bots/joe` (6000),
  `bots/joe_prev` (5000), `bots/joe_base` (re-pointed 3000 → 3500).
  `joe_base`'s weights hash equals the program r2 rated; `joe_prev`'s
  equals the network in the joe-rs artifact.
- Head-to-head budget where the signal is (r2 showed panel games buy
  almost nothing): 160 games 6000-vs-5000, 120 each for the other two
  pairs; 40 games per arm against morpheus-rs and cm_expander for the
  connectivity tie.
- Matched maps via explicit `--seeds` (the per-pair RNG keys on bot ids,
  so a shared `--round-seed` alone would give different arms different
  maps): head-to-head `3101-3180` / `3201-3260` / `3301-3360`, panel
  `3001-3020` for every arm.
- Gate: 360/360/320 games per arm (≥ 200) ✓; 40–160 per pair (≥ 30) ✓;
  ≥ 60 decisive per arm ✓; one engine, one seed set ✓; same component ✓.
  Replication ✗ — single round.
- Host: M3 Pro, 11 jobs, no other load.

## Step 6000 started building castles

The first behavioral change visible in this lineage. Castles built:

| Arm | Games with ≥ 1 castle | Total castles |
| --- | ---: | ---: |
| 6000 | 21 / 360 | 23 |
| 5000 | 0 / 360 | 0 |
| 3500 | 0 / 320 | 0 |

Every joe checkpoint through step 5000 had built zero castles across r1,
r2 and r3 — 1,000+ games. Step 6000 builds in 5.8% of its games, and only
against the other joe arms (13 vs 5000, 10 vs 3500), never in the 80
games against morpheus-rs and cm_expander, which it wins quickly.

This is an observation, not an explanation, and the sample is small: the
games where it builds are 15W–6L, a *worse* rate than its overall record,
which is consistent either with building being mildly bad or — more
likely — with builds happening in the longer, harder games it was losing
anyway. Nothing here separates those. What it does establish is that the
build action stopped being dead code in the policy somewhere between
step 5000 and step 6000, which is worth watching in the next export.

## Standing

Step 6000 is the strongest joe measured, by a small and marginally
proven margin over step 5000 and a large one over step 3500.
