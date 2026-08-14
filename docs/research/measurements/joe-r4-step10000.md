# Joe r4 — step 10000 vs 6000, and r3 replicated (2026-08-14)

Fourth arena round for `bots/joe`, measuring the newly exported EMA
checkpoint (**step 10000** of `joe-M-vast-20260813-0213`) against **step
6000**, and re-measuring the **6000 vs 5000** contrast that r3 reported so
that this round carries its own replication.

560 games, zero failures, engine `9e3b9d13cca5`, `--seat-policy alternate
--strict-versions`, explicit matched seed lists. Following r3's finding
that the roster can no longer measure joe, this was a **joe-only ladder**:
440 head-to-head games and 120 games against the anchor purely to keep the
entities in the anchored component.

## Verdict: **improvement**, unambiguously this time

| Contrast | Elo | CI₉₅ | P | Verdict |
| --- | ---: | --- | ---: | --- |
| **6000 → 10000** (primary) | **+284.8 ± 29.0** | **[+228.0, +341.7]** | **1.0000** | **improvement** |
| 5000 → 6000 (replication) | +69.9 ± 29.1 | [+12.9, +126.8] | 0.9919 | improvement |
| 5000 → 10000 | +354.7 ± 33.7 | [+288.7, +420.7] | 1.0000 | improvement |

Head-to-head on matched maps, both orientations:

| Pair | Record | Decisive win-rate |
| --- | :---: | ---: |
| 10000 vs 6000 | 163W 35L 2D | 82.3% |
| 6000 vs 5000 | 72W 47L 1D | 60.5% |
| 10000 vs 5000 | 102W 17L 1D | 85.7% |

Unlike r3's primary contrast, which cleared the `+10` floor by 0.3 Elo,
this one clears it by 218.

## r3 replicates

This is the first contrast in the joe series to be measured twice, in
separately scheduled rounds, and it holds:

| Round | 5000 → 6000 | Games |
| --- | ---: | ---: |
| r3 | +62.7 ± 26.7 | 160 |
| r4 | +69.9 ± 29.1 | 120 |

The two agree to 7.2 Elo, far inside either interval. The decision rule
demands replication because the M6 morpheus-rs contrast failed it by 7.4
sigma; this one passes, which is evidence for the **method** — frozen
copies, matched seeds, joe-only ladder — and not only for the number.

## The rate did not keep decelerating

r3 reported deceleration and suggested it as stopping evidence. With
another 4000 iterations measured, the picture changes: the rate fell from
~200 to ~70 Elo per 1000 iterations and then **held there**.

| Interval | Iterations | Elo | Elo per 1k |
| --- | ---: | ---: | ---: |
| 3000 → 3500 (r2) | 500 | +104.5 | ~209 |
| 3500 → 5000 (r3) | 1500 | +279.9 | ~187 |
| 5000 → 6000 (r3, r4) | 1000 | +62.7 / +69.9 | ~66 |
| 6000 → 10000 (r4) | 4000 | +284.8 | ~71 |

Steady ~70 Elo/1k across the last 5000 iterations, with no sign of a
plateau. On this evidence the run is still buying strength at a constant
rate and there is no measurement-based reason to stop it.

In-training eval remains blind to all of it: 97.9% / 98.8% / 98.4% vs
random at 5000 / 6000 / 10000. (The R2 state at step 10500 reports 94.1%,
a dip worth watching on the next export, though at this ceiling the
vs-random number has not tracked arena strength for three rounds.)

## Step 10000 uses the build action routinely

r3 caught step 6000 building its first castles. Step 10000 has made it
normal play:

| Arm | Games with ≥ 1 castle | Total castles |
| --- | ---: | ---: |
| 10000 | **180 / 360 (50%)** | 281 |
| 6000 | 11 / 360 (3%) | 17 |
| 5000 | 0 / 280 | 0 |

Dead code at 5000, a curiosity at 6000, half of all games at 10000 — and
the same interval carries the +284.8 Elo. The two are not shown to be
causally linked here; what is established is that the policy discovered
the competition ruleset's distinctive mechanic somewhere after step 5000
and now uses it as a matter of course.

## Reading the contrasts, and one correction

The fitted contrasts agree with the raw win rates within their
uncertainty:

| Pair | Raw rate → Elo | Fitted |
| --- | ---: | ---: |
| 10000 vs 6000 | +267.2 ± 32.4 | +284.8 |
| 6000 vs 5000 | +74.1 ± 32.6 | +69.9 |
| 10000 vs 5000 | +311.3 ± 45.5 | +354.7 |

**Correction to [joe-r3-step6000.md](joe-r3-step6000.md).** That report
offered the transitivity of the three *fitted* contrasts as a consistency
check. It is not one: a Bradley-Terry fit gives each entity a single
strength, so `Δ(a,c) = Δ(a,b) + Δ(b,c)` is an algebraic identity and holds
however the games fell. Only the raw rates can test the chain. Doing that
here: chaining the raw rates gives 10000 → 5000 = +341.3, the direct raw
measurement is +311.3 ± 45.5 — consistent, no meaningful intransitivity.
The r3 conclusion is unaffected; the evidence offered for it was weaker
than stated.

## Absolute ratings remain unreadable

r3's finding stands and this round sharpens it. Step 6000, byte-identical
weights, reads **2388.1** as r3's entity and **2019.5** as r4's — 369 Elo
apart. All three arms went 120W–0L against the anchor, so the prior sets
their level and the fit cannot say how far above the roster they are. Use
the pairwise contrasts; ignore the leaderboard for these entities.

## Design

- Arms: `bots/joe` (10000), `bots/joe_prev` (6000), `bots/joe_base` (5000),
  all frozen copies. The two references' weight hashes equal the programs
  r3 rated (`8ce6cf7d6bfa`, `d2043e48e5a5`).
- 200 games 10000-vs-6000, 120 each for the other two pairs, 40 per arm
  vs `cm_expander`. Matched maps via explicit `--seeds` (`4101-4200`,
  `4201-4260`, `4301-4360`, `4001-4020`).
- Gate: 360/360/280 games per arm (≥ 200) ✓; 40–200 per pair (≥ 30) ✓;
  ≥ 60 decisive per arm ✓; one engine, one seed set ✓; same component ✓;
  replication ✓ for 5000 → 6000, ✗ for the primary contrast.
- Host: M3 Pro, 11 jobs, no other load.
