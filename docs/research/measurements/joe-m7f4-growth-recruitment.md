# Why joe-M7F4 is only ~30 Elo above joe-M (2026-08-22)

Two growth surgeries — depth 5→7, then ff ×3→×4 — took joe-M from 8.56 M to
13.58 M parameters and bought about **30 Elo**. This asks why, and answers it
by ablation rather than by argument.

**The added capacity is not installed.** Zeroing the entire ff ×4 graft out of
the final checkpoint costs **−1.1 ± 3.8 Elo** over 8,192 games: nothing. The
depth splice does carry weight (+15.2 ± 3.8), but its blocks are still at a
fraction of mature scale. The network that plays as M7F4 is, functionally,
joe-M's policy with two partly-attached extensions.

## 1. The lineage is far shorter than the parameter count suggests

| Run | Surgery | Iterations | LR cap | ent_coef start |
| --- | --- | ---: | ---: | ---: |
| joe-M | none (from scratch) | 50,000 | 1e-4 | 0.05 |
| joe-M7 | depth 5 → 7 | **2,000** | 2e-5 | 0.006 |
| joe-M7F4 | ff ×3 → ×4 | 20,000 | 2e-5 | 0.0013 |

joe-M7 was stopped at step 2,000 — four checkpoints exist — and its endpoint
became the ff ×4 seed. Total post-surgery training is 22,000 iterations
against joe-M's 50,000, and the [stage-1
screen](joe-ckpt-m7f4-screen.md) shows M7F4 spending its first ~8,000
recovering to its own graft seed. Net productive adaptation is roughly 14,000
iterations for 59 % more parameters.

Both growth runs were configured as **continuations of a converged schedule**,
not as training a larger model. `lr_power_law_max` is 2e-5 against joe-M's
1e-4 — the M7F4 config says so explicitly: "M7 is stopped inside its flat 2e-5
phase, so a plain continuation keeps the same cap." Entropy went the same way:
joe-M started at `ent_coef` 0.05 with measured entropy 0.778 and annealed to
0.381; M7F4 started at 0.0013 with entropy **0.343 and ended at 0.302**. It
inherited a sharp, converged policy and was given no pressure to explore away
from it.

## 2. The grafted parameters were never recruited

Both surgeries are function-preserving by zero-init, so the RMS of the zeroed
tensors is pure learned recruitment. Measured on the M7F4 EMA checkpoints:

| step | ff ×4 columns RMS | pre-existing columns | ratio |
| ---: | ---: | ---: | ---: |
| 0 | 0.000000 | 0.1442 | 0 % |
| 5000 | 0.007777 | 0.1462 | 5.3 % |
| 10000 | 0.011473 | 0.1478 | 7.8 % |
| 15000 | 0.014668 | 0.1494 | 9.8 % |
| **20000** | **0.017398** | 0.1509 | **11.5 %** |

`attn.out_proj` RMS for the two spliced depth blocks at step 20000: 0.0290
(blk2) and 0.0463 (blk5) against a mature-block mean of 0.1338 — **22 % and
35 %**. The five mature blocks moved 3–4 % across the whole run.

Recruitment is decelerating but nowhere near saturated: the ff ×4 columns
gained 0.00055 RMS per 1k iterations over the last 5,000 steps. **At that rate
they would need roughly 250,000 further iterations to reach the scale of the
columns beside them** — five times the entire joe-M run. Growth-then-adapt at
this schedule cannot deliver the capacity on any affordable budget.

## 3. Ablation: what the grafts actually contribute

Weights zeroed in the step-20000 EMA, then played against the unmodified
checkpoint. 8,192 games per contrast, same instrument as the screens.

| Contrast | Record | Elo | |
| --- | --- | ---: | :-: |
| full vs **ff ×4 zeroed** | 3919W 3944L 329D | **−1.1 ± 3.8** | 0.3σ |
| full vs **depth splice zeroed** | 4112W 3754L 326D | **+15.2 ± 3.8** | 4.0σ |

Zeroing the ff ×4 graft restores the function the graft started from. It costs
nothing measurable — **the entire ff ×4 surgery, 2.07 M parameters and 20,000
H100-iterations, is worth 0 Elo at its own final checkpoint.** The depth
splice is worth about 15, on blocks that are themselves only a quarter grown.

## 4. Capacity is not the binding constraint yet

At the end of each run the 59 %-larger network matches joe-M on every internal
measure:

| metric | joe-M 45–50k | M7F4 15–20k |
| --- | ---: | ---: |
| explained_variance | 0.9741 | 0.9758 |
| value_loss | 3.3960 | 3.3789 |
| mean_ep_length | 513.9 | 507.4 |
| mean_owned_castles | 1.414 | 1.499 |
| draw_rate | 0.0286 | 0.0267 |

A bigger network that fits the value function no better, and plays games of
the same length and shape, is not a network straining against its capacity.

Raw records: `joe-ckpt-m7f4-ablate.json` (2,048 games/pair, three-way) and
`joe-ckpt-m7f4-ablate2.json` (8,192 games/pair, the numbers quoted above).
The 2,048-game pass read the ff ×4 contrast at +12.0 ± 7.7; four times the
sample size moved it to −1.1 ± 3.8. A 1.6σ reading that reverses under more
games is why the claim waited for the larger run.

## 5. What this does and does not establish

It establishes that **"does ff ×4 help?" has not been tested**, because ff ×4
is only 11 % present and contributes 0 Elo. It does not establish that a
properly trained ff ×4 net would be better.

The [update-lever bench](joe-m7f4-update-lever-bench.md) concluded the run is
"signal-limited, not update-limited" from a 1.5× LR cap and a second PPO epoch
over ~1,500 steps. That result stands on its own terms and is not contradicted
here: it tested a nudge, and this measures the end state. Both are consistent
with recruitment being slow for a structural reason — the gradient into a
zero-output column is proportional to its own vanishing contribution — rather
than for want of a slightly larger step.

What would settle it: a growth run given joe-M's actual schedule (LR cap 1e-4,
`ent_coef` 0.05) rather than a decayed continuation, long enough for the
recruitment gauges above to approach mature scale. The gauges make that
cheap to monitor — if new-unit RMS is not tracking toward the mature columns
within a few thousand iterations, the schedule is still wrong.

The cheaper conclusion available today: **the ff ×4 lineage carries 2.07 M
parameters and 5.6 MB of weights that do nothing.** Reverting to the M7
architecture would cost 0 Elo and buy back inference time.
