# Morpheus online path: would a Rust rewrite fix the 150 ms budget?

> Verdict: **not worth it.** At the real deployed configuration
> (`deployment.json`: 8 particles, target 16 simulations) a *complete* warm
> move costs 203 ms p50 / 278 ms p99 on the measurement host, of which
> 143 ms p50 / 202 ms p99 is TorchScript forward passes that survive any host
> language. Zeroing every Python-attributable millisecond still leaves the
> p99 above the 140 ms internal deadline. At the Phase 6 qualification floor
> (32 particles) the NN-only p99 is ~520 ms — worse. The binding constraint
> is forward count × forward cost, and the levers that move it (fewer/cheaper
> forwards) are language-independent.

Date: 2026-08-07. Commit: `b4a2770`. Analysis only — no rewrite plan.

## Question

`RULES.md` §08 gives 150 ms per move; `deployment.json` targets
`normal_deadline_ms: 140` with a 10 ms reserve. Phase 6 re-qualification
([`morpheus-online-runtime.md`](morpheus-online-runtime.md)) rejected every
configuration (36 trials, 0 survivors) and the submission harness rejected the
bundle with `normal_reply_timeout` (8 faults). Would rewriting the online path
under `bots/morpheus/` in Rust make the p99 move land under 140 ms while still
completing belief + root on every warm move and reaching `min_simulations: 8`?

## New measurement (gap fill)

Prior stage timings (`morpheus-online-runtime.json`,
`morpheus-complete-turn-cost-pass-leaf8.json`) date from the int8-qnnpack era
and never separated interpreter time from forward-pass time inside a stage.
This run fills that gap on the **current float32 deployment artifacts**
(bit-exact with `ckpt-00100000-e5b415184274`, per
[`morpheus-float32-deployment.md`](morpheus-float32-deployment.md)):

- Host: **Apple M3 Pro**, macOS 26.5.2, Python 3.12.11, torch 2.13.0,
  `torch.set_num_threads(1)`. Not the judge host; the x86 transfer argument
  is below.
- Method: `measure_online`-style pass-vs-pass scenario, side 21, 99 warm
  moves; `InferenceSession.forward_policy` / `forward_policy_wdl` wrapped
  with timers tagged by evaluator entry point, so
  `python_ms = move_total − nn_ms`.
- Primary config: **`deployment.json` verbatim** — 8 particles, target 16
  simulations, min 8, leaf batch 4, proposal batch 8, 140 ms deadline,
  guard 0 — plus an uncapped-deadline variant that measures the full cost of
  a complete 16-simulation move. A secondary pair at the Phase 6
  qualification floor (32 particles, target 8, proposal batch 32) covers the
  configuration the invariants actually require for acceptance.
- Raw data:
  [`morpheus-rust-rewrite-analysis.json`](morpheus-rust-rewrite-analysis.json).

### What the deployed configuration actually does (verbatim run)

| Regime | sims mean (min) | below 8-sim min | total p50 |
| --- | ---: | ---: | ---: |
| turns 1–64 | 0.1 (0) | 63/64 | 24.5 ms |
| turns 65–99 | 10.7 (8) | 0/35 | 128.7 ms |

The first regime is a **config defect, not a latency problem**:
`deployment.json` ships `offline_p99_ms.enemy_prior_batch: NaN`, and
`can_admit` compares `remaining_ms >= NaN + guard`, which is always False —
enemy priors are never admitted, and every search path that needs one is
discarded, until the estimator's 64-sample window fills with the 0.0 padding
samples appended each turn. So the deployed bot replies fast early in the
game by doing almost no search. **Fixed in `ce6643e`** (finite measured
value + non-finite rejection in `deployment.py` / `runtime.py`); post-fix
the instrumented harness admits enemy priors on turn 1 and early-turn sims
average 7.4 instead of 0.1. The rows above describe the pre-fix state and
are kept as the measured record of the defect.

The second regime is the honest deployed behavior: admission sheds the
16-sim target down to ~10, the 8-sim minimum holds on every move, and 2 of
99 warm moves exceeded 140 ms. Missing the deadline is a *tail* event at
these sizes — but only because admission is shedding roughly a third of the
target search.

### The cost of a complete move at deployed sizes (uncapped run)

All 99 warm moves complete belief + root + 16 simulations:

| Metric | p50 | p99 |
| --- | ---: | ---: |
| total warm move | 202.6 ms | 278.4 ms |
| NN forward time | 142.7 ms | 202.0 ms |
| Python (total − NN) | 60.4 ms | 101.9 ms |

The cheapest complete move observed was 162 ms — no complete 16-sim move fit
inside 140 ms.

## Three-bucket split of the p99 move (deployed config)

**1. Python-interpreter-attributable — 60 ms p50, 102 ms p99.** Selection
12.7, backup 3.4, particle transitions 1.3 ms mean (pure-Python stages), plus
the Python remainder inside NN-bearing stages: leaf batch 113.9 − 84.2 NN =
29.7, enemy priors 46.8 − 41.1 = 5.7, proposal 16.4 − 13.2 = 3.2, root
9.4 − 6.3 = 3.1 ms (uncapped `component_mean_ms` vs `stage_nn_mean_ms`).
~30% of the mean move. This — and only this — is what Rust removes.

**2. TorchScript forwards that survive any host language — 143 ms p50,
202 ms p99.** At 8 particles the mix is search-dominated: leaf batches 84.2
ms mean (16 sims ÷ 4 per batch × ~21 ms per batch-of-4), enemy priors 41.1
ms (~4.3 forwards/turn), belief proposal 13.2 ms (unique inputs ≤ 8 by
construction), root 6.3 ms. Judge-side per-forward costs from the Modal x86
1-thread preflight
([`morpheus-export-preflight-baseline-1thread.json`](morpheus-export-preflight-baseline-1thread.json),
float32 candidate) cross-check the M3 within ~±15% at these shapes: root
(b=1) 6.05 vs 6.3 ms, leaf (b=4) 18.5 vs ~21 ms. Scaling the M3 NN mix by
that ratio puts the judge-side NN-only move at roughly 125–145 ms p50 and
175–205 ms p99.

**3. Fixed overhead — ≤ ~1 ms.** Reply component ≈ 0.0003 ms; non-component
share of a move ≈ 0.4% (`component_total_over_move_mean` 0.9957 in
[`morpheus-complete-turn-cost-pass-leaf8.json`](morpheus-complete-turn-cost-pass-leaf8.json));
stdio serialization is one short line. Model load and warmup sit inside the
10 s first-move grace. Not a factor.

## Amdahl ceiling

Set bucket 1 to zero at the deployed configuration. The complete-move p50
becomes ≈ 143 ms on the M3 (≈ 125–145 ms judge-projected) — at or above the
140 ms deadline — and the p99 becomes ≈ 202 ms (≈ 175–205 ms judge-projected),
1.3–1.5× over. **No — a Rust rewrite does not land the p99 under 140 ms
while completing the deployed 16-simulation target.** What a Rust host
*could* hold is a minimum-profile move (belief + root + exactly 8 sims:
~80–90 ms NN + ~0 Python) — but the post-warmup verbatim run shows Python
already achieves nearly that same profile through admission shedding (p50
128.7 ms, 8-sim minimum kept on every move, 2/99 over 140 ms). Rust would
convert those residual tail faults into margin; it would not change what the
bot can compute per move.

At the **qualification floor** the answer is unconditional: Phase 6 requires
≥ 32 particles, and there the uncapped complete move is 141 ms p50 / 575 ms
p99 with an NN-only share of 94.5 ms p50 / **519.8 ms p99** — the tail is
25–30 unique belief-proposal forwards on high-diversity turns (correlation
0.94 between unique proposal inputs and move time; the worst move spent
424.6 ms in proposal forwards alone). No host language pays less than the
forward count. Rust cannot qualify Part 09 either.

## What a Rust inference runtime would actually cost

- **tch-rs (libtorch bindings).** The only path that loads the TorchScript
  artifacts directly (`CModule::load`) and stays bit-exact — same libtorch
  kernels. By the same token it makes bucket 2 cost exactly what it costs
  today. Ships libtorch (~200 MB) inside the 2 GB judge cap; fine but heavy.
- **ONNX Runtime (`ort` crate).** Needs an ONNX re-export; kernels and
  accumulation order differ, so bit-exactness with the checkpoint is lost
  (the float32 deployment's headline property — export MAE exactly 0.0 —
  would need re-derived parity limits). Any speedup ORT offers is equally
  available from Python via `onnxruntime`, so it is not an argument for Rust.
- **candle / burn.** No TorchScript import; manual weight port; single-thread
  x86 conv performance unproven against libtorch; not bit-exact. Highest
  effort, least evidence.
- **Boundary.** Training, export, and checkpoints stay Python (out of scope
  by definition); the natural boundary is the whole online path behind the
  stdio protocol — ~8.8k lines across search, belief, tree, proposal,
  transition, tensor, tactics, runtime. A partial rewrite (PyO3 module for
  the tree loops) buys only the ~17 ms of pure-Python stage time
  (selection + backup + transitions).

## Cheaper alternatives against the same budget

Ordered by expected effect at the deployed configuration, where the cost is
search forwards (leaf + enemy priors), then for the 32-particle
qualification path, where it is proposal forwards:

| Lever | Bucket | Expected effect | Evidence / cost |
| --- | --- | --- | --- |
| Fix the `enemy_prior_batch: NaN` admission lockout — **done, `ce6643e`** | — | Restored search on turns 1–64 (0.1 → 7.4 sims mean, verified post-fix) | Verbatim run, this measurement; config fix + guards + tests |
| Smaller / distilled net (leaf batch-of-4 is ~21 ms; 16 sims cost ~84 ms alone) | 2 | Halving forward cost puts a complete 16-sim move at ~130–160 ms; a ¼-cost net puts it well under | Width scaling in [`morpheus-float32-deployment.md`](morpheus-float32-deployment.md); needs training + quality baseline per 09b |
| Reduce effective sims per move (admission already does this) or larger leaf batches only if per-sample cost holds | 2 | Post-warmup verbatim run: ~10 sims, p50 128.7 ms, min-sims kept — the deployed shedding is close to the best achievable at this net size | This measurement; bounded by the 8-sim invariant |
| Cap unique proposal forwards per turn (reuse prior logits across turns) | 2 (32p tail) | Turns the 425 ms qualification tail into K × 5–10 ms; the only lever that directly removes the 32-particle p99 driver | 32p runs, corr 0.94; algorithmic; belief-quality effect must be measured |
| Enforce `max_proposal_batch` ≤ 8 in measurement configs | 2 (32p tail) | Per-sample 14.2 ms at batch 30 vs 5.3 ms at small batches (M3); up to ~2.7× tail cut for free | This measurement; x86 preflight same shape (b=4 4.6 vs b=64 9.6 ms/sample). `deployment.json` already says 8; the Phase 6 sweep ran 32 |
| `onnxruntime` under Python | 2 | Plausible 1.3–2× on small convs; unmeasured here | A day with the preflight harness; costs bit-exactness, same as any non-libtorch runtime |
| `torch.compile` (inductor CPU) at `build.sh` time | 2 | Uncertain on 0.25M-param convs; compile must happen offline (no network at match time) | Unmeasured; cheap to probe |
| Targeted C/Cython for selection/backup/transitions | 1 | Removes ~17 ms of pure-Python stage time | Bounded by bucket 1; does not change the verdict |
| Fewer particles / lower sim minimum / zero guard | — | — | Forbidden invariants, below |

## Invariants any rewrite must still honour

From [`morpheus-complete-turn-cost.md`](morpheus-complete-turn-cost.md) and
the Phase 6 gates: do not lower the 8-simulation minimum, do not select a
zero admission guard, do not promote 8 particles — the deployed
`deployment.json` is explicitly a best-effort local play file, not an
accepted deployment, so the 8-particle numbers above describe reality but
cannot become the qualification story. Belief + root must complete on every
warm move; first-move setup stays inside the 10 s grace; 2 GB memory cap
(float32 artifacts are ~1.1 MB — no pressure). The float32 deployment's
bit-exactness with the trained checkpoint is a stated property of the
current artifact trio; only a libtorch-backed runtime preserves it.

## Relation to the plan of record

[09b (2026-08-05)](../../morpheus-implementation/09b-defer-normal-deadline.md)
already defers the normal-latency gate in favour of getting complete belief +
search working and measuring how much reply delay costs in real competition.
This analysis supports that ordering: at deployed sizes the shortfall is
simulations shed (16 → ~10), not replies missed (2/99 over 140 ms after the
NaN fix regime), and the useful speed work is cheaper forwards — not a
rewrite. The submission-harness `normal_reply_timeout` verdict predates the
float32 artifact swap and should be re-run before being cited as current.

## Hosts

- Apple M3 Pro, 1 torch thread (this run,
  `morpheus-rust-rewrite-analysis.json`): the bucket split, the NaN lockout,
  and both configurations' complete-move costs.
- Modal x86 Linux, 1 torch thread
  (`morpheus-export-preflight-baseline-1thread.json`): judge-side per-forward
  costs used for the transfer argument. No judge CPU model is published; the
  x86 numbers are the closest available proxy.
