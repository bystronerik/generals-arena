# Joe-M stage-1 checkpoint screen — 22 checkpoints, 236,544 games (2026-08-21)

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
| 2500 | −1029.6 | 11.2 | −1357.9 ± 12.5 | | 25000 | +171.9 | 2.9 | −156.3 ± 4.1 |
| 3000 | −775.8 | 7.8 | −1104.1 ± 9.1 | | 27500 | +203.2 | 2.9 | −125.1 ± 4.1 |
| 5000 | −484.4 | 5.2 | −812.7 ± 6.5 | | 30000 | +229.5 | 3.0 | −98.8 ± 4.1 |
| 6000 | −419.0 | 4.7 | −747.3 ± 6.0 | | 32500 | +247.9 | 3.0 | −80.3 ± 4.1 |
| 7500 | −274.1 | 4.0 | −602.3 ± 5.3 | | 35000 | +260.3 | 3.0 | −68.0 ± 4.1 |
| 10000 | −161.7 | 3.6 | −489.9 ± 4.8 | | 37500 | +274.6 | 3.0 | −53.7 ± 4.1 |
| 12500 | −97.5 | 3.4 | −425.8 ± 4.6 | | 40000 | +289.4 | 3.0 | −38.9 ± 4.1 |
| 15000 | −19.5 | 3.2 | −347.8 ± 4.4 | | 42500 | +303.4 | 3.1 | −24.9 ± 4.1 |
| 17500 | +55.9 | 3.0 | −272.4 ± 4.2 | | 45000 | +315.9 | 3.1 | −12.4 ± 4.1 |
| 20000 | +110.9 | 3.0 | −217.4 ± 4.1 | | 47500 | +323.7 | 3.1 | −4.6 ± 4.1 |
| 22500 | +146.8 | 2.9 | −181.5 ± 4.1 | | **50000** | **+328.3** | 3.1 | — |

**Monotone at every one of the 21 steps.** Nothing degraded, no late collapse,
no local peak at this granularity. The only checkpoint the screen cannot
separate from 50000 is **47500** (−4.6 ± 4.1). 45000 is already 3σ behind.

The calibration re-check holds inside the larger fit: 5000→6000 **+65.4 ± 6.5**
(arena +69.9 ± 29.1), 6000→10000 **+257.4 ± 5.9** (arena +284.8 ± 29.0),
5000→10000 **+322.8 ± 6.3** (arena +354.7 ± 33.7). All three inside the
published intervals.

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
