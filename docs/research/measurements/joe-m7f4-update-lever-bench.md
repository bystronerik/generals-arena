# joe-M7F4 — update-lever bench: num_epochs 2 and LR cap 3e-5 (2026-08-19)

One live run, three phases, two levers measured and reverted. The question:
can the M7F4 growth-adaptation run learn faster per wall-clock hour on the
same H100 by extracting more update from each rollout? Answer: **no** — the
run is signal-limited, not update-limited. Both cheap levers are closed.

Provenance:

- Run: `joe-M7F4-vast-20260819-0207` (1×H100 interruptible, vast.ai
  instance 47615917), engine `9e3b9d13cca5`.
- Lineage: joe-M (depth 5, 50k iters, LR annealed to 5e-6) → depth-7
  splice, ~2k iters (M7) → ff ×3→×4 growth (M7F4). Both surgeries are
  young: at step 2000 the spliced blocks 2/5 sit at ~10 % of mature
  attn.out scale, the new FF units at ~5 %.
- Config commits: `afecafd` (num_epochs 2), `e9683fb` (revert + cap 3e-5),
  and the revert commit that adds this file.
- Instruments: the frozen-reference eval (EMA greedy vs joe-M final EMA,
  512 games / 50 iters; EMA decay 0.999 lags ~700 iters — discount early
  post-change evals) and a checkpoint weight gauge over the R2 full saves
  (every 500 steps). The gauge exploits the zero-init surgery leaves:
  blocks 2/5 `attn.out_proj` + `ff_linear2` (M7 splice) and all blocks'
  `ff_linear2[:, 1152:]` (ff ×4) start at exact zero, so their RMS is pure
  learned recruitment. Script: session scratchpad `inspect_blocks.py`
  (rebuild from this table's definitions if needed).

## Phases

| Steps | Config | s/iter | mean KL | ref-eval mean |
| --- | --- | --- | --- | --- |
| ≤1500 | 1 epoch, cap 2e-5 (baseline) | 12.86 | ~0.010 | 47.5 % (n=6) |
| 1500–2500 | 2 epochs, cap 2e-5 | 17.8 eff. (+38 %) | ~0.019 | 46.9 % (n=16) |
| 2500–4000 | 1 epoch, cap 3e-5 | 12.87 | ~0.019 | 47.5 % (n=13 logged) |

The `target_kl 0.02` guard (live only at num_epochs 2) trimmed the second
epoch on 13–20 % of iterations. Entropy stayed at 0.34–0.39 with no trend
through all phases; the draw rate oscillated 4–12 % without direction.

## Result 1 — a second PPO epoch adds √2 diffusion, not learning

Per-500-step relative drift ‖ΔW‖/‖W‖ (network split by block age; young =
spliced blocks 2, 5):

| Window | Phase | young | mature |
| --- | --- | --- | --- |
| 0–500 | 1 ep @ 2e-5 | 0.192 | 0.0300 |
| 500–1000 | 1 ep @ 2e-5 | 0.195 | 0.0303 |
| 1000–1500 | 1 ep @ 2e-5 | 0.195 | 0.0308 |
| 1500–2000 | 2 ep @ 2e-5 | 0.261 | 0.0413 |
| 2000–2500 | 2 ep @ 2e-5 | 0.275 | 0.0441 |
| 2500–3000 | 1 ep @ 3e-5 | 0.231 | 0.0403 |
| 3000–3500 | 1 ep @ 3e-5 | 0.234 | 0.0410 |

Two epochs doubled per-iteration KL (0.010 → 0.019) but moved weights only
~1.35–1.45× — the √2 signature of uncorrelated steps, in every parameter
group. The second pass over a batch diffuses; it does not learn. The ref
eval stayed flat while iterations cost +38 %. Reverted.

## Result 2 — a 1.5× LR cap does not speed recruitment

KL scales ~quadratically with step size (0.010 → 0.019 at 1.5× LR), but
total drift rose only ~1.2–1.35× (overshoot partly cancels), and the
recruitment gauges did not move at all. Per-500-step increments of the
young blocks' `attn.out_proj` RMS:

| Window | Phase | blk2 | blk5 |
| --- | --- | --- | --- |
| 0–1500 (3 windows) | 1 ep @ 2e-5 | +0.0008…+0.0010 | +0.0013…+0.0016 |
| 2500–3000 | 1 ep @ 3e-5 | +0.0008 | +0.0014 |
| 3000–3500 | 1 ep @ 3e-5 | +0.0011 | +0.0011 |

(The apparent 2-epoch bump in 1500–2500, +0.0015/+0.0026, is diffusion
inflation of near-zero weights, consistent with Result 1.) The ff ×4
columns tell the same story: ~5.2 % of old-unit scale at step 2000,
growing ~+0.001 RMS per 500 steps regardless of cap. No harm appeared at
3e-5 over 1500 iterations (entropy, draws, ref eval all flat with the EMA
~64 % post-change by step 3500), but no speed appeared either. Reverted:
equal pace at higher drift on the annealed mature weights is risk with no
return.

## Conclusion

Adam already gives the tiny-gradient young units full-size normalized
steps. Their slow growth means the gradient direction on them is
inconsistent batch to batch — recruitment is limited by signal coherence
in fresh self-play data, not by step size or update count. Global levers
(epochs, LR) cannot add coherence; they add diffusion. This also predicts
a per-group LR on young parameters would mostly diffuse as well.

Decision: run the ff4 plan's 20k-iteration budget at the plan's settings
(1 epoch, cap 2e-5). Track adaptation with the recruitment gauge at
~2k-step intervals (read-only, no relaunch) and judge strength by arena
export contrasts at the r-checkpoints, not by the in-run eval. Blocks 2/5
roughly doubled their recruitment gauge every ~3k steps so far; if that
compounding holds, meaningful scale lands inside the budget.
