# Round report — macaria

Candidate `macaria@80f3047ac205` against baseline `blitz@b4a69aad6389`.
Spec: [`../strategies/macaria.md`](../strategies/macaria.md). Bot doc:
[`../../bots/macaria.md`](../../bots/macaria.md). Thresholds:
[`../../arena/decision-rule.md`](../../arena/decision-rule.md).

All games `--mode competition`, `--seat-policy alternate`,
`--strict-versions`, engine `9e3b9d13cca5`.

| round | roster | games/pair | seed | games |
| --- | --- | ---: | ---: | ---: |
| `macaria-r1` | macaria, blitz, cm_expander, cm_hunter, metro, aegis, boom | 50 | 7 | 1050 |
| `macaria-r2-<opp>` | macaria vs each of the six, separately | 100 | 17 | 600 |
| control (**not pooled**) | macaria with the search off, both seed sets mirrored | 50 / 100 | 7 / 17 | 900 |

`macaria-r2` is six two-bot rounds rather than one seven-bot round.
`_pair_rng` keys only on `(round_seed, sorted bot ids)`, not on the roster, so
pair `(macaria, X)` draws the same seeds either way; a seven-bot roster would
have replayed every panel-vs-panel pair to buy macaria's 600 games. r1 already
bought the panel-vs-panel games that connect the pool.

---

## 1. The decision

Refit over the whole pool, then the pairwise contrast — never a rank.

| | |
| --- | --- |
| A (baseline) | `blitz@b4a69aad6389`, 2486.2 [2454, 2519], n = 3204 (2364-814-26) |
| B (candidate) | `macaria@80f3047ac205`, 2566.0 [2519, 2613], n = 900 (755-142-3) |
| `comparable` | **True** — same connectivity group |
| **Δ (B − A)** | **+79.8 ± 20.4 Elo** |
| **CI₉₅** | **[+39.8, +119.8]** |
| **P(B > A)** | **1.0000** |
| **Verdict** | **improvement** |

Gate, item by item: both hashes registered; every game `mode == "competition"`;
one engine era; shared opponent panel and seed sets; **900 and 3204 games per
arm** (floor 200); **150 games per (macaria, opponent)** (floor 30); **897
decisive games** on the candidate arm (floor 60). 1650 rated games were
re-read and checked individually — zero anomalies, zero degenerate games, every
one reaching a normal end.

### 1.1 It was `unproven` first, and the reason was sample size

At 300 games on the candidate arm — the r1 round alone — the same contrast read:

> Δ = **+57.9 ± 33.2**, CI₉₅ **[−7.2, +122.9]**, P(B > A) = **0.9594** →
> **unproven**

`P ≥ 0.95` was met; `CI₉₅.low > +10` was not, at **−7.2**. That is a miss of
the rule as written, and it is recorded rather than rounded up. The mechanism
is not strength but precision: blitz carried 3104 games of history and macaria
300, so the contrast's SE was almost entirely the candidate's arm.
`games_to_resolve(target_se=12.75)` said **830**. r2 bought 600 more, the SE
fell 20.4 from 33.2, and the point estimate *rose* rather than regressing to
zero. No code changed between the two readings.

---

## 2. No regression against the panel

r1, the round where both arms played all five panel opponents at 50 games each:

| opponent | macaria | blitz | Δ |
| --- | --- | --- | ---: |
| aegis | 45-5-0 · 0.900 [0.79, 0.96] | 36-13-1 · 0.730 [0.59, 0.83] | **+17.0 pp** |
| cm_hunter | 46-4-0 · 0.920 [0.81, 0.97] | 43-7-0 · 0.860 [0.74, 0.93] | +6.0 pp |
| boom | 37-13-0 · 0.740 [0.60, 0.84] | 35-15-0 · 0.700 [0.56, 0.81] | +4.0 pp |
| cm_expander | 50-0-0 · 1.000 [0.93, 1.00] | 49-1-0 · 0.980 [0.90, 1.00] | +2.0 pp |
| metro | 45-4-1 · 0.910 [0.80, 0.96] | 45-4-1 · 0.910 [0.80, 0.96] | 0.0 pp |
| **pooled** | **223-26-1 · 0.894** [0.85, 0.93] | 208-40-2 · 0.836 [0.79, 0.88] | +5.8 pp |

**No opponent regresses.** macaria is at or above blitz on all five, and the
one flat cell (metro) is flat to the game.

These two columns are *not* seed-paired with each other — `_pair_rng` keys on
the bot ids, so `(macaria, aegis)` and `(blitz, aegis)` drew different maps.
They are matched on panel and count only; the seed-paired comparison is §4.

Pooled over all 900 candidate games (r1 + r2):

| opponent | macaria | score | CI₉₅ |
| --- | --- | ---: | --- |
| cm_expander | 150-0-0 | 1.000 | [0.975, 1.000] |
| cm_hunter | 139-11-0 | 0.927 | [0.873, 0.959] |
| metro | 136-13-1 | 0.910 | [0.853, 0.946] |
| aegis | 130-19-1 | 0.870 | [0.807, 0.915] |
| boom | 115-34-1 | 0.770 | [0.696, 0.830] |
| **blitz (head-to-head)** | **85-65-0** | **0.567** | **[0.487, 0.643]** |

## 3. The head-to-head, stated as what it is

**85-65-0 over 150 games: 0.567, CI₉₅ [0.487, 0.643].** The interval still
contains 0.500. The direct matchup is *directionally* consistent with the
contrast and **does not on its own establish anything** — which is exactly why
the decision rule reads the panel contrast rather than the duel. Reported here
because the brief asked for it, not because it carries the verdict.

---

## 4. Control arm — how much of this is the search?

`MACARIA_TUNE='{"mcts_enabled": false}'` is macaria with the search off, which
is the vendored core alone. Run on **the same 900 map seeds and seats** as the
live arm, so every cell is a matched triple.

Stored outside `data/games/` and rated by winrate only: it plays under
macaria's content hash while being a different program, and pooling it would
merge two programs into one rated entity.

| arm | W-L-D | score | CI₉₅ |
| --- | --- | ---: | --- |
| search **off** (the core alone) | 734-165-1 | 0.816 | [0.789, 0.840] |
| search **on** (shipped) | 755-142-3 | **0.841** | [0.815, 0.863] |

Paired over all 900 matched cells: **+73 / −51**, 776 unchanged.
Two-sided sign test on the 124 discordant pairs: **p = 0.0589**.

**So the search's own contribution is +2.5 pp, and it does not clear p < 0.05
on matched pairs.** It is suggestive, not established. Stated plainly because
the alternative reading — that the +79.8 Elo is mostly the search — is the one
the numbers do *not* support at conventional significance.

### 4.1 What the control does establish, decisively

| arm | vs blitz, 150 games | score | CI₉₅ |
| --- | --- | ---: | --- |
| search off | 75-75-0 | **0.500** | [0.421, 0.579] |
| search on | 85-65-0 | 0.567 | [0.487, 0.643] |

**The core with the search off draws dead even with blitz — 75-75.** That is
the vendoring validated end-to-end on 150 competition games, on top of the
move-for-move check over 6 seeds. macaria's core *is* blitz's program; nothing
in the copy, the constant-hoisting, or the `strategy_context` refactor changed
how it plays. Every point of separation between macaria and blitz therefore
comes from the search, and from nowhere else.

That is the cleaner claim, and it is the one the data supports: **the
attribution is certain even though the effect size is not.**

---

## 5. Latency

`Agent.act` timed per move through the probe channel, on real recorded games.
Single job, quiet machine — which is the competition's own condition (§08
grants one dedicated core).

| sample | moves | p50 | p95 | p99 | max | >100 ms | >150 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 6 games vs cm_hunter / aegis / metro | 2984 | 2 ms | 3 ms | 3 ms | **5 ms** | 0 | 0 |

**Gate: max 5 ms against a 150 ms budget.** Zero games recorded a fault, a
crash, or a malformed reply across 1650 rated games plus 900 control games.

| | |
| --- | --- |
| search firing rate | **54.7%** of turns (38.0% vs metro, 60.4% vs aegis) |
| overrides | 216 = **13.2% of searched turns**, 7.2% of all moves |
| iterations per searched turn | min 12, **median 40**, mean 45, max 143 |

### 5.1 The search is ladder-bound, not deadline-bound

The budget almost never binds. Quadrupling it — `mcts_budget_ms` 55 → 220,
cap 400 — over the same two games changed **nothing measurable**:

| `mcts_budget_ms` | searched | overrode | iters median | iters mean | iters max | max ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 55 (shipped) | 816 | 131 | 40 | 44 | 135 | 7 |
| 220 | 816 | 131 | 40 | 44 | 135 | 6 |

Identical, to the individual override. The search terminates because it
**exhausts its deepening ladder** (horizons 4/6/8/10), not because it runs out
of clock. On a dedicated core it spends about **3% of its 100 ms cap**.

The deadline machinery is therefore a safety net rather than the operating
point — but it is a *load-bearing* one, and it was observed working: measured
under three-way agent contention on this dev machine, the same search showed
whole-move p99 27 ms and max 128 ms with iterations cut to as low as 16, and
still nothing above 150. That is the deadline check inside the rollout loop
doing its job under a load the competition does not impose.

**This is unexploited headroom, and it is the obvious next experiment:** a
deeper ladder or a wider matrix is free in wall-clock terms. Untried here —
stated as a lead, not a result.

---

## 6. Negative and null results

- **§1.1** — the contrast was `unproven` at 300 games/arm, `CI₉₅.low = −7.2`
  against a +10 threshold.
- **§4** — the search's own contribution, on matched pairs, is p = 0.0589.
  Not significant.
- **§3** — the head-to-head interval contains 0.500.
- **The biggest lever in the loss data is untouched.** Instrumenting blitz over
  600 games found target acquisition to be the strongest single correlate of
  the result: the enemy general is located in **100% of wins (510/510), 45% of
  losses, 8% of draws**. Finding it is exploration policy, not search — vision
  is Chebyshev-1, you must physically walk adjacent — so macaria does not
  address it at all. Whatever this bot is doing, it is not fixing the thing the
  data points at hardest.
- **Three intermediate configurations lost and were discarded** during
  development, all the same failure: an override that is locally reasonable and
  strategically corrosive. Firing the contested-contact trigger during
  `rebuild`/`rally` bled the wave economy one marginal override at a time
  (88–216 rebuild-phase overrides in a single game; a 371-turn win stretched
  past 771). An ungated home-gather row and a progress term anchored per-row
  instead of on one common object did the same thing by making `PASS` look free
  after a spent wave. All three are why the trigger is assault-phase only.
- **Draws are not blitz's problem, contrary to the pool-wide rate.** The pool
  draws ~35%; blitz draws **1.2%** over its 2328 historical games and 2.0% over
  600 fresh ones. macaria draws 3 of 900. No draw-conversion work was done and
  none was warranted.

## 7. What is assumed, and what the search cannot see

- **Fog.** Enemy cells outside vision do not exist in the forward model. The
  observation's `opp_army`/`opp_land` are the engine's *true* totals, so hidden
  army is an exact deduction — but *where* it sits is not, and the search does
  not place it. The certainty horizon is one ply.
- **Castle builds**, ours and theirs, are not modelled; at ≤10 plies a build's
  swing is ≤5 army and it must first bank 35.
- **Off-screen army** and enemy production from castles built in fog.
- **The opponent's real policy.** Their reply set is paranoid and small; their
  continuation is one stack marching at our general. That is the most dangerous
  cheap model, not the most likely one — deliberately, since the row must
  survive the min.
- **One fabrication**, inherited from the core: a once-seen, currently-fogged
  enemy general carries the core's own `1 + turn/2` garrison estimate.
- **The 8-turn defensive gather remains a per-turn approximation.** The loss
  data says blitz gets a median 8 turns of warning and loses with 84 mobile
  army against a 33.5 killing stack, because it is spread one-per-cell. A
  search that re-decides every turn improves the visible end of that race; it
  cannot begin a gather against army it cannot yet see.

---

## 8. Reproduce

```bash
python -m arena.tournaments.competition \
  bots/macaria/run.sh bots/blitz/run.sh bots/cm_expander/run.sh \
  bots/cm_hunter/run.sh bots/metro/run.sh bots/aegis/run.sh bots/boom/run.sh \
  --round macaria-r1 --games-per-pair 50 --round-seed 7 \
  --seat-policy alternate --strict-versions
```

The control arm is `MACARIA_TUNE='{"mcts_enabled": false}'`, research-only
plumbing (`bots/macaria/params.py`). Unset — which is what every stored game
above ran — the shipped defaults are the whole program.
