# Joe training loop — per-phase wall-clock profile (Modal, 2026-08-27)

Where the time goes inside one `training/joe` PPO iteration, measured on the
real loop rather than a re-implementation.

Provenance:

- Script: [`scripts/joe_modal_profile.py`](../../../scripts/joe_modal_profile.py).
- Engine: `competition-module` @ `9e3b9d13cca5`.
- Stack: Modal `debian_slim` py3.12, `jax[cuda12]==0.11.0`, equinox, optax.
- Network: frozen **S tier** (depth 4, embed 352, ff 2, patch 3),
  5,087,930 parameters — identical to the S row in
  [joe-phase1-throughput.md](joe-phase1-throughput.md).
- Shape: 256 envs x 64 steps x 2 seats = **32,768 samples/iter**,
  minibatch 1024, `adv_top_frac` 0.25, 1 epoch, `pool_size` 200,000,
  single curriculum stage at the competition preset (distance 17+).
- Raw JSON: `joe-train-phase-profile-{t4,l4}.json`,
  `joe-train-phase-drilldown-t4.json`, `joe-train-bf16-ab-t4.json`.

## Method

`ppo.train` is **not modified**. The profiler attaches to its code object
through `sys.monitoring` local LINE events (PEP 669) and timestamps every
source line, tagging each with the iteration it ran in. Sixteen named phases
are mapped from anchor strings in the source, so the mapping survives edits.

Validation: on a CPU dry run the line clock agreed with the loop's own
built-in timers — line 490 measured 1.672 s against its self-reported
`rollout 1.73 s`, line 574 measured 0.871 s against `ppo 0.93 s`.

Reading caveat: JAX dispatch is async, so a line's wall clock is the time
until the next blocking point. The loop blocks after the rollout (line 492)
and after the PPO step (line 579), and every `float(...)` in the diagnostics
block is an implicit barrier, so the phases are separated by real barriers.

## 1. The headline: one clean iteration

A "clean" iteration is one with no eval, no pool refresh and no checkpoint.
On L4 those are iterations 2, 4, 6 and 9; they agree to within 3 %.

| Phase | L4 s/iter | share | T4 s/iter | share |
| --- | --- | --- | --- | --- |
| Rollout (self-play) | 1.04 | 45 % | 96.4 | 96.1 % |
| PPO update | 1.09 | 47 % | 3.81 | 3.8 % |
| GAE + advantage normalize + diagnostics | 0.02 | 1 % | 0.02 | 0.0 % |
| EMA, metrics log, checkpoint check | 0.15 | 7 % | 0.05 | 0.1 % |
| **Total** | **2.30** | | **100.3** | |

**On a bf16-capable GPU the loop is ~50/50 rollout and PPO, and everything
else is noise.** Both halves are the same transformer: forward only in the
rollout, forward plus backward on 25 % of the samples in PPO. The env, the
observation pipeline, GAE, the diagnostics and the logging together cost
under 10 %.

L4 throughput: 15,197 samples/s. T4: 328 samples/s — a **46x** gap between
two cards whose raw specs are ~2x apart. Section 4 explains it.

## 2. Rollout drill-down: it is entirely the network

Cumulative stacks of the real rollout pipeline, same 64 x 512 shape, T4:

| Stage | s | share of rollout |
| --- | --- | --- |
| `env.step` only | 0.031 | 0.0 % |
| + observations, build cost, move/build masks | 0.043 | 0.0 % |
| + 39-channel augmentation | 0.102 | 0.1 % |
| + network forward and sampling (real `collect_rollout`) | 94.94 | **99.9 %** |


> **Scoped by a later measurement.** See
> [joe-transformer-kernel-probe.md](joe-transformer-kernel-probe.md): on an
> RTX PRO 6000 at the X16 production shape, removing the entire depth-16
> transformer trunk leaves the rollout time unchanged (16.078 s -> 16.064 s).
> The 99.9 % above is real for *this* T4 / S-tier / 256x64 configuration --
> inflated by the 17x Turing bf16 penalty and a 32x smaller batch -- and does
> not generalize. Which part dominates is a property of (card, tier, shape).
>
> **The 0.1 % non-network figure in this table is also an artifact.** The S1-S3
> stacks above reduce each stage into a scalar accumulator, so XLA elides most
> of the observation and augmentation work. A bisection that keeps `obs_aug`
> feeding the network and `env.step` measures the same pipeline at **~99 % of
> the rollout**, with `env.step` itself at 0.016 s. Do not cite the 0.1 %.

Everything that is not the network costs 0.10 s. The Phase 1 finding that
"the env itself will never be the bottleneck" holds with a large margin —
the observation and augmentation code is free too, so there is nothing to
win there.

## 3. Periodic and one-time costs

Incremental cost over the 2.30 s clean L4 iteration:

| Event | L4 cost | Cadence in `S.yaml` | Amortized at this shape |
| --- | --- | --- | --- |
| Eval vs random (cached compile) | 2.06 s | `eval_every` 50 | 0.04 s/iter |
| Map-pool refresh, 200k | 4.59 s | `reset_pool_every` 10 | 0.46 s/iter (**+20 %**) |
| Checkpoint save | 0.46 s | `save_every` 500 | ~0 |

One-time, per run (L4, 448 s total for 10 iterations):

| Cost | s | note |
| --- | --- | --- |
| Cold map-pool generation, 200k | 185.9 | 96 % of all setup; T4 130.0 |
| Iteration 0 — every cold JIT compile | 118.8 | eval 44.2, rollout 36.8, PPO 30.3 |
| Iteration 1 — a second wave of compiles | 55.9 | see below |
| Iteration 3 — the second eval recompiles | 39.4 | see below |
| **Total one-time** | **~400** | **89 % of this 10-iteration run** |

Two of those are not obvious. Iteration 1 costs 58.2 s against a 2.30 s
steady iteration, and the *second* eval (iteration 3) costs 43.8 s while the
third (iteration 7) costs 4.4 s. So a second compile wave happens after
values have made one trip through `pmap`. The likely cause is an array
layout change between the freshly replicated parameters and the ones
`p_ppo_step` returns, which would force `_evaluate_side` and the rollout to
recompile once; that mechanism is a hypothesis, the extra ~95 s is measured.

## 4. `use_bf16: true` costs 17x on Turing — and not for the obvious reason

Every joe config sets `use_bf16: true`. The T4 is Turing (sm_75). Same-host
A/B on the real `collect_rollout`, identical weights, only the static
`use_bf16` flag differing, interleaved:

| Round | bf16 | f32 | ratio |
| --- | --- | --- | --- |
| 0 | 95.138 s | 5.581 s | 17.05x |
| 1 | 95.025 s | 5.631 s | 16.88x |

The obvious explanation — "Turing has no bf16 tensor cores" — is **wrong**.
A bare GEMM at the network's hidden width, same container:

| dtype | 4096x352 @ 352x1056 |
| --- | --- |
| f32 | 0.589 ms |
| bf16 | 0.587 ms |
| f16 | 0.152 ms |

bf16 matmul on a T4 is exactly as fast as f32; XLA simply upcasts. So the
17x does not come from the matmuls — it comes from the rest of the bf16
path (the `_to_bf16` parameter cast, LayerNorm, softmax and the elementwise
ops) failing to lower well on sm_75. The specific kernel is not identified
here.

Consequences:

- **Do not train joe on Turing.** T4 with `use_bf16: false` is 17x faster
  than T4 with the shipped config, and an L4 is another 5x faster again.
- The f16 row shows the T4's tensor cores are reachable through float16, not
  bfloat16 — a lever if a Turing card is ever the only option.
- Ampere and later are unaffected: A10G, L4, A100 and H100 all have native
  bf16, and the L4 numbers above are the ones to plan against.

## 5. What is worth optimizing

1. **The transformer forward and backward.** ~92 % of a clean iteration.
   Nothing else is close. Levers: precision, kernel/batch shape,
   `adv_top_frac` (0.25 today) which scales PPO cost linearly, and
   `num_steps` / `num_envs` which scale the rollout.
2. **Nothing in the env or observation pipeline.** Measured at 0.1 % of the
   rollout. Any effort there is wasted.
3. **The map-pool refresh** is a real 20 % tax *at this batch shape*
   (`reset_pool_every` 10, 4.59 s). At the production 2048 x 256 shape the
   base iteration is ~32x longer and the same 4.59 s falls under 2 %. It is
   only worth touching for small-batch runs.
4. **`_evaluate_side` scans the full `truncation` (1200) steps twice per
   eval with no early exit.** The `finished` mask is accumulated but never
   used to stop the scan. At the final curriculum stage most games run long
   so the waste is small, but at the early stages (distance 2–6) games end
   in a few dozen turns and almost the whole scan is wasted work.
5. **Cold start is ~400 s** and is dominated by pool generation (186 s) plus
   compiles. A curriculum stage change pays a fresh cold pool generation, and
   `S.yaml` has five stages. The Volume-backed
   `jax_compilation_cache_dir` in `scripts/joe_modal_train.py` addresses the
   compile half on restarts only.

## 6. Caveats

- **Shape.** 256 x 64 = 32,768 samples/iter against the production
  2048 x 256 = 1,048,576. Every *fixed* per-iteration cost (pool refresh,
  eval, EMA, logging, diagnostics) is ~32x more prominent here than in
  production. The rollout/PPO split is shape-stable; the overhead shares are
  not.
- **Tier.** S tier only. M and L shift the split toward the network further,
  never away from it.
- Ten iterations per GPU, one run each. The clean iterations agree to within
  3 %, and the bf16 A/B was interleaved same-host, but no run-to-run
  variance across containers was measured. The T4 vs L4 comparison is
  therefore cross-host and carries that risk; the 46x gap is far larger
  than any plausible host spread, and the bf16 conclusion rests on the
  same-host A/B, not on it. See
  [unclejoe forward steps](unclejoe-forward-steps.md) on why only
  same-host interleaved contrasts settle small differences.
