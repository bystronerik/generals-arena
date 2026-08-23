# Joe checkpoint screen — calibration against the rated rounds (2026-08-21)

> **Corrected 2026-08-22.** Every Elo here was refitted after a draw-accounting
> bug: ~1 % of games terminate with no winner (simultaneous death under
> deathtouch), the screen's counter saw only truncation draws, and the fit
> credited each unaccounted game to the pair's second entity — always the
> later checkpoint in a step-ordered round-robin. Contrasts moved by up to
> 4.6 Elo, all shrinking the later checkpoint's advantage. **No conclusion
> changed.** Detail: [joe-m7f4-growth-recruitment](joe-m7f4-growth-recruitment.md).


Stage-1 pilot of
[joe-M-checkpoint-selection-plan](../strategies/joe-M-checkpoint-selection-plan.md).
Before the screen is allowed to narrow 98 checkpoints, it has to reproduce
contrasts the arena already measured. It does, on all three.

**These are not arena matches.** The screen plays inside the JAX env under
`scripts/joe_modal_ckpt_sweep.py`; no game here enters `data/games/`,
`data/ratings/`, or any rating fit (root `AGENTS.md`).

## Setup

- Modal, one H100. Script `scripts/joe_modal_ckpt_sweep.py`, weights staged
  from R2 into the `joe-M-ckpts` Volume.
- Checkpoints: EMA steps 5000, 6000, 10000 of `joe-M-vast-20260813-0213`
  (8,556,250 params), the three steps with published head-to-head records.
- Competition preset, distance 17+, engine truncation 1200, scan 1202 steps.
  All 6,144 games finished; none hit the scan ceiling.
- Both seats greedy. Each pair meets on **one shared map set in both
  orientations** — the same key drives both seat assignments, which is the
  arena's `--seat-policy alternate`: seat balance by construction, and map
  difficulty cancels inside the matched pair. `map_seed` 20260821, shared
  across every pair.
- **`use_bf16=False`.** Training ran bf16, but `joe_export_bot.py` does not
  forward that field, so the *rated* bot runs an f32 forward. The screen
  matches the bot, not the trainer.
- 2,048 games per pair (1,024 maps × 2 orientations), 6,144 games total.

## The calibration

Joint Bradley-Terry over the pair matrix, anchored at step 5000, against the
fitted arena numbers from [joe-r3](joe-r3-step6000.md) and
[joe-r4](joe-r4-step10000.md):

| Contrast | Screen | Arena (r4) | Arena CI₉₅ | Inside |
| --- | ---: | ---: | --- | :---: |
| 5000 → 6000 | **+49.2** | +69.9 ± 29.1 | [+12.9, +126.8] | ✅ |
| 6000 → 10000 (r4 primary) | **+313.2** | +284.8 ± 29.0 | [+228.0, +341.7] | ✅ |
| 5000 → 10000 | **+362.4** | +354.7 ± 33.7 | [+288.7, +420.7] | ✅ |

The widest span — 5000 → 10000, where a broken instrument has the most room
to disagree — matches to **7.7 Elo**. Screen SEs are 4.3–5.4 Elo against the
arena's ~30, on 2,048 games per pair against the arena's 120–200.

Head-to-head records, screen vs arena, as decisive win-rates:

| Pair | Screen (2,048 games) | Arena (r4) |
| --- | --- | --- |
| 6000 v 5000 | 1115W 899L 16D — 55.4% | 72W 47L 1D — 60.5% |
| 10000 v 6000 | 1717W 321L 5D — 84.3% | 163W 35L 2D — 82.3% |
| 10000 v 5000 | 1854W 186L 5D — 90.9% | 102W 17L 1D — 85.7% |

**Gate: passed.** Stage 2 may proceed.

## Per-pair Elo does not chain — fit jointly

At 512 games the per-pair Elos read +43.0 (6000>5000) and +269.6
(10000>6000), which sum to +312.6 against a **directly measured +402.3** for
10000>5000. A 73-Elo transitivity gap. The arena's numbers add up exactly
(69.9 + 284.8 = 354.7) because they come from one joint fit, and the screen's
did not because they came from three independent score rates.

Quantified once the script started reporting goodness of fit: this
three-node fit has **chi2/dof 14.4 on a single degree of freedom** — the
triangle does not close. That is the same defect seen from the other side,
and it is why the calibration verdict rests on the *model-free* head-to-head
score rates above as much as on the fitted contrasts. A 3-checkpoint fit has
no slack to absorb a strained triangle; the narrow-band production fits
(0.66–0.78) do.

The script now fits Bradley-Terry over the whole matrix
(`bradley_terry()`, also re-runnable on a saved JSON via the `fit` entry
point). Nothing downstream may chain per-pair Elo across hops — over a
20-checkpoint ladder that error compounds at every step.

## Throughput — measured, and 2.2× worse than the plan estimated

| Shape | Games/pair | Steady-state |
| --- | ---: | ---: |
| n_maps 256 | 512 | 42.8 games/s |
| n_maps 1024 | 2,048 | 41.9 games/s |

**~42 games/s on one H100, flat in batch size from 256 up** — the GPU is
already saturated at the smaller shape, so there is nothing to buy by
batching harder. The plan's pre-measurement estimate of ~93 games/s
(extrapolated from the phase-1 M-tier rollout figure) was optimistic by
2.2×; every cost line in the plan has been corrected to the measured number.

Fixed costs, per job: pool generation 12–45 s (JIT cache on the Volume warms
it), ~1.6 s per checkpoint load, ~20 s to compile a new scan shape.

For scale: the arena runs 14–18 games/min on the dev Mac. The screen is
**~150× faster** and answers a different, weaker question.

## Raw

`joe-ckpt-pilot.json` (512 games/pair) and `joe-ckpt-calib.json`
(2,048 games/pair, with the BT fit).
