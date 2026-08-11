# Joe Phase 1 — env throughput benchmark (Modal, 2026-08-11/12)

Phase 1 of
[`averagejoe-competition-plan.md`](../strategies/averagejoe-competition-plan.md):
the paper's Table-I equivalent for our fork, measured in **competition mode**
(build_castles + deathtouch, boards 18–21 padded to 21, truncation 1200).
These numbers re-anchor the cost estimates for Phases 2–6.

Provenance:

- Script: `scripts/joe_modal_bench.py` (single run per GPU; timings are means
  of 3 iterations after a compile warmup).
- Engine: `competition-module` @ `9e3b9d13cca5`
  (competition-engine-2026-15-g9e3b9d1).
- Stack: Modal `debian_slim` py3.12, `jax[cuda12]==0.11.0`, equinox 0.13,
  optax 0.2. GPUs: 1×A10G (24 GB) and 1×H100 (80 GB) on 2026-08-11;
  1×A100-80GB added 2026-08-12 to check whether a cheaper 80 GB card
  matches the H100 on cost per sample (dashboard utilization: all three
  cards peaked at 95–100 % — every GPU is compute-saturated at the
  1024-env bench shape).
- Raw JSON: appended at the bottom of this file.
- Actual spend across all four bench jobs: ≈ $2–3, well inside the plan's
  <$10 Phase 1 budget.

## 1. Raw env stepping (no net)

`vmap(env.step)` under `lax.scan`, random *valid* actions (categorical over
the move mask), auto-reset from a pool sized to the env count. One env step
advances one game tick for both players.

| num_envs | A10G steps/s | A100-80G steps/s | H100 steps/s |
| --- | --- | --- | --- |
| 1,024 | 2.78 M | 3.08 M | 4.42 M |
| 4,096 | 3.20 M | 6.08 M | 8.67 M |
| 16,384 | 3.20 M | 7.63 M | 12.17 M |

A10G saturates at ~3.2 M steps/s by 4k envs; A100 and H100 are still
scaling at 16k. Env state memory is trivial (< 0.4 GB peak at 16k envs).
The env itself will never be the bottleneck — with the net in the loop the
rollout runs two orders of magnitude slower (§3).

## 2. Map-pool generation, pool_size = 200,000

`env.reset()` at the frozen configs' pool size. Cold includes compiling the
16 (h, w) kernels; warm is a re-reset with a new key (what
`reset_pool_every` actually pays).

| Window | GPU | cold s | warm s | pool GB | peak GB |
| --- | --- | --- | --- | --- | --- |
| distance 17+ (preset) | A10G | 52.7 | 2.7 | 0.91 | 4.16 |
| distance 2–6 (early) | A10G | 54.5 | 2.7 | 0.91 | 4.16 |
| distance 17+ (preset) | A100 | 78.9 | 1.3 | 0.91 | 4.16 |
| distance 2–6 (early) | A100 | 81.8 | 1.3 | 0.91 | 4.16 |
| distance 17+ (preset) | H100 | 38.2 | 0.8 | 0.91 | 4.15 |
| distance 2–6 (early) | H100 | 37.9 | 0.8 | 0.91 | 4.16 |

Pool refresh is a non-issue: 0.8–2.7 s warm every 10–20 iters (≈ 0.1–0.3 %
of iteration time at S-tier). The distance window does not affect cost.
Curriculum stage changes pay one cold generation each (~40–55 s), a handful
of times per run.

## 3. Net-in-the-loop: rollout + PPO update (the real cost)

Randomly initialized nets at the frozen tier shapes (39-channel augmented
obs, 10-action head with build mask, patch 3, HL-Gauss 128-bin value CE,
|adv| top-25 % filter, minibatch 1024, 1 epoch, bf16). Bench shape:
1,024 envs × 64 steps × 2 seats = 131,072 samples/iter. Parameter counts
came out at 5.09 M / 8.56 M / 15.35 M — L matches the paper's 15.35 M
exactly, independently confirming the depth-7 reading of D1.

| Tier | params | GPU | rollout s | PPO s | samples/s |
| --- | --- | --- | --- | --- | --- |
| S | 5.1 M | A10G | 2.50 | 2.59 | 25,783 |
| M | 8.6 M | A10G | 3.69 | 3.74 | 17,644 |
| L | 15.4 M | A10G | 6.32 | 6.65 | 10,105 |
| S | 5.1 M | A100 | 0.94 | 0.87 | 72,399 |
| M | 8.6 M | A100 | 1.32 | 1.20 | 52,021 |
| L | 15.4 M | A100 | 2.14 | 2.07 | 31,165 |
| S | 5.1 M | H100 | 0.45 | 0.40 | 154,821 |
| M | 8.6 M | H100 | 0.60 | 0.54 | 114,290 |
| L | 15.4 M | H100 | 0.93 | 0.88 | 72,114 |

Rollout and PPO update split roughly 50/50 at every tier on every GPU.
Peak device memory at the bench shape: ~11.4 GB (all GPUs, all tiers —
dominated by rollout storage, not the net).

**Config-shape batch (2,048 envs × 64 steps, A100 only).** To check whether
a bigger batch changes the picture, the A100 also ran the net loop at the
frozen configs' env count: S 69,421 samples/s, M 49,354 samples/s — within
5 % of its 1024-env numbers. The A100 is already saturated at 1,024 envs
(dashboard confirmed ~100 % utilization), so larger batches buy nothing
on that card. Peak memory doubled to
~22 GB, as expected (rollout storage scales with samples). The H100 was
not re-run at this shape; its 1024-env numbers are used as-is below.

## 4. What this means for the plan

**GPU choice: H100.** The A100-80GB question ("cheaper card, same 80 GB —
better value?") is answered by measurement: no. At Modal on-demand rates
(~$1.10 A10G, ~$2.50 A100-80G, ~$3.95 H100 per hour), cost per billion
samples at each tier:

| Tier | A10G | A100-80G | H100 |
| --- | --- | --- | --- |
| S | $11.9 | $9.6 | **$7.1** |
| M | $17.3 | $13.4 | **$9.6** |
| L | $30.2 | $22.3 | **$15.2** |

The H100 is ~2.1–2.3× faster than the A100 at ~1.6× the price — ~25–35 %
cheaper per sample at every tier, and it halves wall-clock time on top.
The A100's saturation at 1,024 envs (§3) closes the "just feed it a bigger
batch" escape hatch. All Phase 3/6 estimates below assume 1×H100 (~$4/h).

**Iteration time at the frozen configs** (2,048 envs × 256 steps × 2 seats
= 1.05 M samples/iter, extrapolated linearly from measured samples/s):

| Tier | s/iter (H100) | iters/hour | 50k iters |
| --- | --- | --- | --- |
| S | ~6.8 | ~530 | ~94 h ≈ $380 |
| M | ~9.2 | ~390 | ~128 h ≈ $510 |

- **Phase 2 smoke** (~30 min at distance 2–6): ~250 S-tier iters ≈ 260 M
  samples — plenty to see loss movement and >50 % vs random. Within budget.
- **Phase 3** ($50–100 = 12–24 h) buys ~6.5k–13k S-tier iterations, **not**
  the released config's 50k. The released S run's 50k iters would cost
  ~$380. Whether 6.5k–13k reaches the final curriculum stage and ≥90 % vs
  random is exactly what Phase 3's gates test; if it falls short, the
  honest options are more hours or a cut-down `num_iters` — the curriculum
  gate, not the iteration counter, is the success criterion.
- **Phase 6** M-tier at 50k iters lands at ~$510 — consistent with the
  plan's $300–500 for a 3–5 day run (fewer iters or spot pricing covers
  the gap).

**Memory bounds the config shape, not the pool.** Measured: 131k
samples/iter peaks at 11.4 GB; 262k samples/iter peaks at 22 GB (A100,
config-shape batch). Rollout storage is ~42 GB/M samples (bf16 obs +
masks + float32 temporal windows), so the frozen configs' 1.05 M
samples/iter extrapolate to ~55–60 GB peak — fits an 80 GB card,
**definitively rules out A10G (24 GB) and A100-40G**, and leaves no room
for careless extra copies in the Phase 2 port. If the port hits allocator
pressure, halving `num_steps` per rollout (and doubling rollouts per pool
refresh) is the lever; the temporal window (4 KB/sample, float32) is the
second-largest term and could store bf16. Note the Modal dashboard's
"60 GB used" during these runs was XLA's default preallocation (75 % of
VRAM), not demand — the numbers here are `memory_stats()` peaks.

## 5. Caveats

- The bench net is a compact, shape-faithful stand-in written for this
  benchmark, not the Phase 2 port: fused QKV projection, no EMA update, no
  eval games, no checkpointing, no `target_kl` early stop. Real iterations
  will be a few percent slower; eval every 50 iters adds its own cost that
  this bench did not measure.
- Raw-env numbers (§1) use uniform-random valid moves; a trained policy's
  action distribution does not change env step cost, but the §1 numbers
  exclude observation building and are an upper bound on env throughput,
  not a training-relevant figure.
- Single run per GPU; iteration timings varied by a few percent across the
  3 measured repeats.
- Samples/s at the configs' 2,048-env batch was measured on the A100 only
  (within 5 % of its 1024-env figure). The H100's 2048-env behavior is
  extrapolated linearly from its 1024-env numbers; since the H100 also
  peaked at ~100 % utilization, the batch-size change should be as neutral
  there as it measured on the A100.

## Raw results

```json
{
  "A10G": {
    "raw_env_steps": [
      {"num_envs": 1024, "env_steps_per_s": 2782806, "compile_s": 5.0},
      {"num_envs": 4096, "env_steps_per_s": 3202829, "compile_s": 5.9},
      {"num_envs": 16384, "env_steps_per_s": 3202701, "compile_s": 7.7}
    ],
    "pool_gen": [
      {"window": "distance-17-preset", "pool_size": 200000, "cold_s": 52.7, "warm_s": 2.7, "pool_gb": 0.91, "peak_gb": 4.156},
      {"window": "distance-2-6-early", "pool_size": 200000, "cold_s": 54.5, "warm_s": 2.7, "pool_gb": 0.91, "peak_gb": 4.156}
    ],
    "net_in_loop": [
      {"tier": "S", "params_m": 5.09, "rollout_s": 2.5, "ppo_s": 2.59, "samples_per_s": 25783, "peak_gb": 11.197},
      {"tier": "M", "params_m": 8.56, "rollout_s": 3.69, "ppo_s": 3.74, "samples_per_s": 17644, "peak_gb": 11.232},
      {"tier": "L", "params_m": 15.35, "rollout_s": 6.32, "ppo_s": 6.65, "samples_per_s": 10105, "peak_gb": 11.392}
    ]
  },
  "A100-80GB": {
    "raw_env_steps": [
      {"num_envs": 1024, "env_steps_per_s": 3077131, "compile_s": 8.9},
      {"num_envs": 4096, "env_steps_per_s": 6082948, "compile_s": 8.9},
      {"num_envs": 16384, "env_steps_per_s": 7634107, "compile_s": 11.3}
    ],
    "pool_gen": [
      {"window": "distance-17-preset", "pool_size": 200000, "cold_s": 78.9, "warm_s": 1.3, "pool_gb": 0.91},
      {"window": "distance-2-6-early", "pool_size": 200000, "cold_s": 81.8, "warm_s": 1.3, "pool_gb": 0.91}
    ],
    "net_in_loop": [
      {"tier": "S", "params_m": 5.09, "rollout_s": 0.94, "ppo_s": 0.87, "samples_per_s": 72399, "peak_gb": 11.199},
      {"tier": "M", "params_m": 8.56, "rollout_s": 1.32, "ppo_s": 1.2, "samples_per_s": 52021, "peak_gb": 11.441},
      {"tier": "L", "params_m": 15.35, "rollout_s": 2.14, "ppo_s": 2.07, "samples_per_s": 31165, "peak_gb": 11.623}
    ],
    "net_in_loop_2048_envs": [
      {"tier": "S", "params_m": 5.09, "rollout_s": 1.96, "ppo_s": 1.82, "samples_per_s": 69421, "peak_gb": 21.944},
      {"tier": "M", "params_m": 8.56, "rollout_s": 2.77, "ppo_s": 2.54, "samples_per_s": 49354, "peak_gb": 22.126}
    ]
  },
  "H100": {
    "raw_env_steps": [
      {"num_envs": 1024, "env_steps_per_s": 4422237, "compile_s": 4.7},
      {"num_envs": 4096, "env_steps_per_s": 8665093, "compile_s": 4.7},
      {"num_envs": 16384, "env_steps_per_s": 12171158, "compile_s": 5.8}
    ],
    "pool_gen": [
      {"window": "distance-17-preset", "pool_size": 200000, "cold_s": 38.2, "warm_s": 0.8, "pool_gb": 0.91, "peak_gb": 4.154},
      {"window": "distance-2-6-early", "pool_size": 200000, "cold_s": 37.9, "warm_s": 0.8, "pool_gb": 0.91, "peak_gb": 4.161}
    ],
    "net_in_loop": [
      {"tier": "S", "params_m": 5.09, "rollout_s": 0.45, "ppo_s": 0.4, "samples_per_s": 154821, "peak_gb": 11.058},
      {"tier": "M", "params_m": 8.56, "rollout_s": 0.6, "ppo_s": 0.54, "samples_per_s": 114290, "peak_gb": 11.211},
      {"tier": "L", "params_m": 15.35, "rollout_s": 0.93, "ppo_s": 0.88, "samples_per_s": 72114, "peak_gb": 11.443}
    ]
  }
}
```

Bench shape for `net_in_loop`: 1,024 envs × 64 steps × 2 seats;
`rollout_env_steps_per_s` and full memory-stat fields are in the run log
(scratchpad `joe_bench_results.json`); regenerate any of this with
`modal run scripts/joe_modal_bench.py`.
