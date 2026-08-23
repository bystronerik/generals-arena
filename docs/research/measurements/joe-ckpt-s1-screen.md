# Joe-M stage-1 checkpoint screen — 22 checkpoints, 236,544 games (2026-08-21)

> **Corrected 2026-08-22.** Every Elo here was refitted after a draw-accounting
> bug: ~1 % of games terminate with no winner (simultaneous death under
> deathtouch), the screen's counter saw only truncation draws, and the fit
> credited each unaccounted game to the pair's second entity — always the
> later checkpoint in a step-ordered round-robin. Contrasts moved by up to
> 4.6 Elo, all shrinking the later checkpoint's advantage. **No conclusion
> changed.** Detail: [joe-m7f4-growth-recruitment](joe-m7f4-growth-recruitment.md).


Stage 1 of
[joe-M-checkpoint-selection-plan](../strategies/joe-M-checkpoint-selection-plan.md).
Coarse screen over `joe-M-vast-20260813-0213`: every 2500 iterations from 2500
to 50000, plus 3000 and 6000 so the rated-round overlap sits inside the same
joint fit.

**Not arena matches.** `scripts/joe_modal_ckpt_sweep.py` on one Modal H100;
nothing here enters `data/games/`, `data/ratings/`, or any rating fit.
Instrument and its calibration:
[joe-ckpt-screen-calibration](joe-ckpt-screen-calibration.md).

## Setup

22 checkpoints, full round-robin = 231 pairs, 1,024 games per pair (512 maps ×
2 orientations, matched maps in both seats), 236,544 games in **98 min**
at 40.2 games/s. Competition preset, distance 17+, truncation 1200, both seats
greedy, `use_bf16=False` to match the exported bot. `map_seed` 20260821 shared
across every pair. Joint Bradley-Terry, mean-anchored.

## Result: step 50000 wins, and the curve is monotone

| step | Elo | ± | vs 50000 | | step | Elo | ± | vs 50000 |
| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 2500 | -1021.1 | 11.1 | -1343.5 ± 12.3 | | 25000 | +171.0 | 2.9 | -151.4 ± 4.1 |
| 3000 | -767.7 | 7.7 | -1090.1 ± 9.0 | | 27500 | +202.1 | 2.9 | -120.3 ± 4.1 |
| 5000 | -479.6 | 5.1 | -802.0 ± 6.4 | | 30000 | +227.6 | 2.9 | -94.8 ± 4.1 |
| 6000 | -414.2 | 4.7 | -736.6 ± 6.0 | | 32500 | +246.8 | 3.0 | -75.6 ± 4.1 |
| 7500 | -271.4 | 4.0 | -593.8 ± 5.2 | | 35000 | +257.8 | 3.0 | -64.6 ± 4.1 |
| 10000 | -159.9 | 3.5 | -482.3 ± 4.8 | | 37500 | +271.5 | 3.0 | -50.8 ± 4.1 |
| 12500 | -96.3 | 3.3 | -418.7 ± 4.6 | | 40000 | +285.8 | 3.0 | -36.5 ± 4.1 |
| 15000 | -18.2 | 3.1 | -340.6 ± 4.3 | | 42500 | +299.5 | 3.0 | -22.9 ± 4.1 |
| 17500 | +56.4 | 3.0 | -265.9 ± 4.2 | | 45000 | +311.6 | 3.1 | -10.8 ± 4.1 |
| 20000 | +111.1 | 2.9 | -211.3 ± 4.1 | | 47500 | +318.3 | 3.1 | -4.0 ± 4.1 |
| 22500 | +146.5 | 2.9 | -175.9 ± 4.1 | | **50000** | **+322.4** | 3.1 | — |

**Monotone at every one of the 21 steps.** Nothing degraded, no late collapse,
no local peak at this granularity. The only checkpoint the screen cannot
separate from 50000 is **47500** (−4.0 ± 4.1). 45000 is already 3σ behind.

The calibration re-check holds inside the larger fit: 5000→6000 **+65.4 ± 6.4**
(arena +69.9 ± 29.1), 6000→10000 **+254.3 ± 5.9** (arena +284.8 ± 29.0),
5000→10000 **+319.7 ± 6.3** (arena +354.7 ± 33.7). All three inside the
published intervals.

## Correction (2026-08-22): this fit is misspecified — trust the order, not the Elo

Added when the sweep script started reporting goodness of fit. **This stage-1
fit has chi2/dof = 9.7** (residual sd 14.0 Elo). Bradley-Terry
assumes one logistic scale, and a 22-checkpoint ladder spanning 1,358 Elo
breaks that assumption. The three narrow-band fits in this series are fine by
comparison: joe-M stage 2 is 0.78, the M7F4 stages are 0.66 and 0.66.

What survives and what does not:

- **The ordering survives.** A model-free Copeland ranking over the same 231
  pairs reproduces the BT order exactly, except that it swaps 45000 and 47500
  — a pair whose direct match is 0.497. Exactly **1 of 231 pairs** has the
  later checkpoint losing its direct match, and it is that same near-tie. The
  conclusion "step 50000 is strongest, and the curve is monotone" is
  model-free and stands.
- **The Elo magnitudes below do not.** The wide-gap values (step 2500 at
  −1029.6) and the per-1k rate table are the parts the misfit distorts most,
  because the residuals concentrate where the logistic curve is saturated.
  Read the rate table as a shape — fast, decelerating, flat at the end — not
  as calibrated Elo.
- **The top-of-run conclusion was never resting on this fit.** It came from
  [stage 2](joe-ckpt-s2-screen.md), chi2/dof 0.92, over a 39-Elo band.

This is also the cleanest explanation for the cross-fit gap noted in stage 2
(42500 → 50000 reads +24.9 here and +38.7 there): one of the two fits is
misspecified, and it is this one.

## The rate decayed to nothing

| Interval | Elo | per 1k iters |
| --- | ---: | ---: |
| 3000 → 5000 | +291.4 | +145.7 |
| 7500 → 10000 | +112.4 | +45.0 |
| 15000 → 17500 | +75.4 | +30.2 |
| 22500 → 25000 | +25.2 | +10.1 |
| 32500 → 35000 | +12.3 | +4.9 |
| 42500 → 45000 | +12.4 | +5.0 |
| 45000 → 47500 | +7.8 | +3.1 |
| **47500 → 50000** | **+4.6** | **+1.9** |

r4 measured +71 Elo per 1k across 6000 → 10000 and reported the rate had
stopped decelerating. Over the next 40,000 iterations it resumed decelerating
and effectively stopped: the last 2,500 iterations bought 4.6 ± 4.1 Elo, which
is not distinguishable from zero. The run converged; it was not cut short.

There is still +489.9 ± 4.8 between step 10000 — the last step the arena ever
rated — and step 50000.

## What this means for stage 3

**The top of the run is below the arena's resolution.** The whole 40000–50000
span is 38.9 ± 4.1 Elo, and 47500 → 50000 is 4.6. The decision rule calls
`CI₉₅` inside ±25 "proven flat" and needs `CI₉₅.low > +10` for an
improvement. An arena round between 47500 and 50000 would be buying ~5 Elo of
separation with an instrument whose affordable floor is ±25 — it can only
return "proven flat", at 4–6 h of dev-Mac time.

Two consequences, neither of which is a failure of the plan:

- **The answer to "which joe-M checkpoint is strongest" is step 50000.** It is
  the argmax of a monotone curve, not merely the default, and it costs nothing
  extra to use.
- Stage 2 is now a narrow question with a known-low ceiling: whether one of the
  500-step checkpoints in 44000–50000 beats 50000 by more than a few Elo. The
  monotone curve argues against it. Any winner it found would still be inside
  the arena's flat window.

**Caveat on the scale.** These are self-play Elo among joe checkpoints. The
+489.9 from 10000 to 50000 does not transfer to the arena panel, where the
late checkpoints likely saturate against aegis and macaria; panel-relative
differences compress. The screen ranks; it does not size an arena gap.

## Raw

`joe-ckpt-s1.json` — every pair record plus the joint fit and its covariance.
