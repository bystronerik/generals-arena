# joe-M7F4 checkpoint screen — the deployed lineage (2026-08-22)

Which EMA checkpoint of `joe-M7F4-vast-20260819-0207` is strongest. This is
the **deployed** lineage: `bots/joe` and `bots/joe-rs` both ship its step
16000. Same instrument and method as the
[joe-M screen](joe-ckpt-s1-screen.md); calibration in
[joe-ckpt-screen-calibration](joe-ckpt-screen-calibration.md).

**Answer: step 19000 or 20000 — they tie. The deployed step 16000 is
15.9 ± 3.0 Elo behind step 20000 (5.3σ).**

**Not arena matches.** Nothing here enters `data/games/`, `data/ratings/`, or
any rating fit.

## The run

41 EMA checkpoints, steps 0–20000 every 500, all retained. depth 7, ff×4,
13,581,658 params — 1.59× joe-M's parameter count, which costs 1.56× the
screen time (26.5 vs 41.9 games/s). Single curriculum stage at distance 17+;
`num_iters: 20000`. Step 0 is the function-preserving ff×4 graft seed, so it
is the M7 parent's endpoint and serves as a free control.

**This run logged its own strength curve, and joe-M did not.** It ran with
`eval_ref_checkpoint: ref/joe-M_ema_final.eqx` — the reference is joe-M step
50000, the checkpoint the [joe-M screen](joe-ckpt-s1-screen.md) proved
strongest. 376 eval points, 512 games each, EMA greedy vs that reference:

| step bin | Elo vs joe-M 50000 | | step bin | Elo vs joe-M 50000 |
| ---: | ---: | --- | ---: | ---: |
| 1000 | −4.3 | | 11000 | +22.7 |
| 4000 | −18.7 | | 14000 | +25.4 |
| 7000 | −3.6 | | 17000 | +31.2 |
| 8000 | +9.7 | | 18000 | +35.1 |
| 10000 | +20.9 | | 19000 | +29.9 |

M7F4 does not pass its own joe-M reference until about step 8000, and ends
roughly 30 Elo above it. That is the whole return on the depth-7 + ff×4 growth
at this budget. The figure is bf16 (the training forward) against a fixed
external opponent, so treat it as context, not as this screen's result.

## Stage 1 — the whole run, every 2000 iterations

11 checkpoints, 55 pairs, 1024 games/pair, 56,320 games, 35 min.
chi2/dof 0.81.

| step | Elo | | step | Elo |
| ---: | ---: | --- | ---: | ---: |
| **20000** | **+29.7 ± 3.1** | | 8000 | −11.6 ± 3.1 |
| 18000 | +21.4 ± 3.1 | | 4000 | −22.7 ± 3.1 |
| 16000 | +20.0 ± 3.1 | | 6000 | −24.0 ± 3.1 |
| 14000 | +19.3 ± 3.1 | | 0 (graft seed) | −25.0 ± 3.1 |
| 12000 | +14.9 ± 3.1 | | 2000 | −35.8 ± 3.1 |

**The graft cost strength before it paid.** Step 2000 sits 10.8 Elo *below*
the step-0 seed: the ff×4 columns are zero-initialised and function-preserving
at init, but PPO recruiting them disturbs the policy first. The run passes its
own seed by step 8000.

**The whole run spans 65 Elo**, against joe-M's 1,358. This is an adaptation
run on an already-trained network, not a from-scratch run.

## Stage 2 — the top band at 500 granularity

13 checkpoints (14000–20000), 78 pairs, 2048 games/pair, 159,744 games, 96
min. chi2/dof 0.72.

| step | Elo | vs best | vs 16000 (deployed) |
| ---: | ---: | ---: | ---: |
| **19000** | **+10.9 ± 2.0** | — | **+16.4 ± 3.0** |
| **20000** | **+10.3 ± 2.0** | −0.5 ± 3.0 | **+15.9 ± 3.0** |
| 19500 | +5.8 ± 2.0 | −5.0 ± 3.0 | +11.4 ± 3.0 |
| 17500 | +2.4 ± 2.0 | −8.4 ± 3.0 | +8.0 ± 3.0 |
| 18500 | +2.0 ± 2.0 | −8.8 ± 3.0 | +7.6 ± 3.0 |
| 18000 | +0.1 ± 2.0 | −10.7 ± 3.0 | +5.7 ± 3.0 |
| 17000 | −1.4 ± 2.0 | −12.3 ± 3.0 | +4.1 ± 3.0 |
| 15000 | −1.5 ± 2.0 | −12.3 ± 3.0 | +4.1 ± 3.0 |
| 16500 | −2.4 ± 2.0 | −13.3 ± 3.0 | +3.2 ± 3.0 |
| 15500 | −4.7 ± 2.0 | −15.6 ± 3.0 | +0.9 ± 3.0 |
| **16000 (deployed)** | −5.6 ± 2.0 | −16.4 ± 3.0 | — |
| 14500 | −7.6 ± 2.0 | −18.5 ± 3.0 | −2.0 ± 3.0 |
| 14000 | −8.4 ± 2.0 | −19.2 ± 3.0 | −2.8 ± 3.0 |

19000, 20000 and 19500 are the band; nothing else comes within 8 Elo.

**The fine structure is noisy, unlike joe-M's.** Adjacent checkpoints wobble
by up to 5 Elo against a trend of about 1 Elo per 500 iterations — 19000
(+10.9) sits above 19500 (+5.8), and 17500 (+2.4) above 18000 (+0.1). At this
granularity the EMA jitter is the same size as the drift, which is why the
answer is a three-checkpoint band and not a single step.

Six pairs measured in both stages agree to a mean 5.2 Elo, inside per-pair
noise.

## The deployed checkpoint is not the best one

`bots/joe` and `bots/joe-rs` ship step 16000, which the screen puts
**15.9 ± 3.0 Elo behind step 20000** — 5.3σ, the clearest separation this
screen has produced. Step 16000 is not a bad pick: it is mid-band, and its
`last_eval_wr` of 0.992 gave no reason to look further. But 20000 is the run's
own last checkpoint and measurably stronger.

Two things temper acting on it:

- **15.9 Elo is still under the arena's floor.** The decision rule needs
  `CI₉₅.low > +10` and treats anything inside ±25 as flat. A rated round on
  this contrast is not a good use of 4–6 h.
- **A re-export is not free.** It forks the bot's rating identity and
  staleens the joe-rs weights, the parity corpus, the committed fixture, and
  the joe-rs self-goldens.

The recommendation is to switch on the next export that happens anyway,
rather than to spend a round proving 16 Elo.

## Cost

Stage 1 + stage 2 = 216,064 games in ~2.2 H100-hours. The two joe-M stages
were 396,288 games in ~2.7 h; this run is smaller but its network is 1.59×
larger.

## Raw

`joe-ckpt-m7f4-s1.json`, `joe-ckpt-m7f4-s2.json` — pair records, joint fits,
covariances, fit diagnostics.
