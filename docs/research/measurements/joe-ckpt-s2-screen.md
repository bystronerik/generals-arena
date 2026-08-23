# Joe-M stage-2 checkpoint screen — the final band at 500 granularity (2026-08-22)

> **Corrected 2026-08-22.** Every Elo in this document was refitted after a
> draw-accounting bug was found: ~1 % of games terminate with no winner
> (simultaneous death under deathtouch), the screen's draw counter saw only
> truncation draws, and the fit credited each unaccounted game to the pair's
> second entity — always the later checkpoint in a step-ordered round-robin.
> Contrasts moved by up to 4.6 Elo, all shrinking the later checkpoint's
> advantage. **No conclusion changed.** Detail:
> [joe-m7f4-growth-recruitment](joe-m7f4-growth-recruitment.md).


Stage 2 of
[joe-M-checkpoint-selection-plan](../strategies/joe-M-checkpoint-selection-plan.md),
run as the completeness check
[stage 1](joe-ckpt-s1-screen.md) scoped it to be: does any 500-step checkpoint
in the run's last 6,000 iterations beat step 50000?

**No.** Step 50000 and step 49500 tie, and nothing else comes within 7 Elo.

**Not arena matches.** `scripts/joe_modal_ckpt_sweep.py` on one Modal H100;
nothing here enters `data/games/`, `data/ratings/`, or any rating fit.

## Setup

13 checkpoints: 44000–50000 at 500 granularity (12 of them — **47000 is not
retained in R2**), plus 42500 as a positive control that stage 1 had already
placed clearly behind. Full round-robin, 78 pairs, **2,048 games per pair**
(1,024 maps × 2 orientations), 159,744 games in **64 min** at 41.9 games/s.
Settings otherwise identical to stage 1.

## Result

| step | Elo | ± | vs 50000 |
| ---: | ---: | ---: | ---: |
| **49500** | +15.6 | 2.0 | +1.2 ± 3.0 |
| **50000** | +14.4 | 2.0 | — |
| 48500 | +8.9 | 2.0 | -5.5 ± 3.0 |
| 49000 | +8.7 | 2.0 | -5.7 ± 3.0 |
| 48000 | +6.7 | 2.0 | -7.7 ± 3.0 |
| 46500 | +2.5 | 2.0 | -11.9 ± 3.0 |
| 47500 | +2.4 | 2.0 | -12.0 ± 3.0 |
| 46000 | -2.7 | 2.0 | -17.1 ± 3.0 |
| 44000 | -8.2 | 2.0 | -22.6 ± 3.0 |
| 45500 | -8.7 | 2.0 | -23.2 ± 3.0 |
| 45000 | -9.3 | 2.0 | -23.8 ± 3.0 |
| 44500 | -12.3 | 2.0 | -26.8 ± 3.0 |
| 42500 (control) | -17.9 | 2.0 | -32.3 ± 3.0 |

**No local peak.** The curve is monotone up to sub-noise wiggles: the only
inversions are 44500 below 44000 (4.1 ± 3.0) and 50000 below 49500
(1.2 ± 3.0), neither significant. The finer granularity did not reveal
structure that the 2500-step grid had hidden — it confirmed there was none.

**49500 and 50000 are one answer, not two.** 1.2 ± 3.0 Elo apart, i.e. the
ordering between them is arbitrary at this sample size and would stay
arbitrary at any sample size the arena can afford. Take 50000: it is the run's
last checkpoint, so it needs no justification beyond being tied for best.

The control behaved: 42500 is last, separated from the field.

## Cross-stage consistency

Six pairs were measured in both sweeps, on disjoint map sets (n_maps 512 vs
1024 split from the same key give different maps) at different sample sizes:

| Pair | Stage 1 | Stage 2 | diff |
| --- | ---: | ---: | ---: |
| 42500 v 45000 | −7.8 | −8.7 | −0.8 |
| 42500 v 47500 | −23.4 | −19.4 | +4.1 |
| 42500 v 50000 | −18.0 | −25.5 | −7.5 |
| 45000 v 47500 | +2.0 | −16.3 | −18.3 |
| 45000 v 50000 | −26.5 | −28.4 | −1.9 |
| 47500 v 50000 | −9.8 | −9.7 | +0.2 |

Mean |diff| 5.5 Elo against a per-pair SE of ~11 (stage 1) and ~8 (stage 2);
the largest gap is 1.4σ. The instrument repeats.

**Compare within a fit, not across fits.** The 42500 → 50000 contrast reads
+24.9 in stage 1's joint fit and +38.7 in stage 2's, while the *direct*
head-to-head readings agree (−18.0 and −25.5). A Bradley-Terry contrast
depends on the opponent field it was fitted in — stage 1's spans 1,350 Elo,
stage 2's spans 39 — so screen Elo from two different sweeps is not directly
comparable. Rankings within one sweep are.

**Draws rose with parity:** 3.9% mean here (range 2.8–5.1%) against 1.3% in
stage 1, as expected when every pair is near-equal.

## Infrastructure note

The first stage-2 attempt lost 25 finished pairs — Modal cancels a remote call
when the local client dies, and the client was killed at ~35 min. The script
now commits every finished pair to the Volume as it goes and takes a `--tag`
resume, and sweeps launch with `modal run --detach`. The relaunch reproduced
the lost run's first pairs **exactly** (42500 v 44000: 934W 999L 91D in both),
which is also a useful determinism check on the screen.

## Raw

`joe-ckpt-s2.json` — every pair record plus the joint fit and its covariance.
