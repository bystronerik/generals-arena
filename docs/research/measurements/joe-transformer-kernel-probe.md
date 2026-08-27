# Joe transformer — what the forward and backward actually cost (2026-08-27)

Follow-up to [joe-train-phase-profile.md](joe-train-phase-profile.md), on the
silicon production actually uses: **RTX PRO 6000 Blackwell** (sm_120), at the
**X16 production shape**.

Provenance:

- Hardware: vast.ai instance 48926095, `RTX PRO 6000 Blackwell Max-Q
  Workstation Edition`, 97,887 MiB, compute capability 12.0, driver
  610.57.04. Interruptible at $0.40/h; ~52 min total, **≈ $0.39**.
- Scripts: [`scripts/joe_gpu_kernel_probe.py`](../../../scripts/joe_gpu_kernel_probe.py)
  (HLO facts), [`scripts/joe_gpu_trunk_ablation.py`](../../../scripts/joe_gpu_trunk_ablation.py)
  (rollout), [`scripts/joe_gpu_ppo_ablation.py`](../../../scripts/joe_gpu_ppo_ablation.py) (PPO).
- Config: `X16.yaml` unchanged except `pool_size` (irrelevant to transformer
  kernels). depth 16, embed 384, ff x4, 8 heads, head_dim 48,
  **29,551,834 params**, 2048 envs x 128 steps = 524,288 samples/iter,
  minibatch 2048, `adv_top_frac` 0.25 -> 64 minibatches over 131,072 samples.
- Raw JSON: `joe-pro6000-{attention-ab,rollout-ablation,ppo-ablation,hlo-facts}.json`.

## 1. The headline: the transformer is the PPO half, and only the PPO half

Ablation removes the whole depth-16 trunk by making
`SelfAttentionLayer.__call__` the identity. Same weights, same shapes, same
data; three interleaved rounds, best of rounds reported.

| | base | trunk removed | trunk's share |
| --- | --- | --- | --- |
| **Rollout** | 16.078 s | 16.064 s | **0.09 %** |
| **PPO update** | 13.608 s | 0.224 s | **98.3 %** |

So of a ~29.7 s iteration, the transformer is **~46 %, all of it inside the
PPO update**. The rollout does not care whether the network exists.

That is not a typo and not a broken ablation: the identical patch drops the
PPO update from 13.6 s to 0.22 s, so it demonstrably removes the trunk's
compute. In the rollout, 128 sequential steps at batch 4096, the trunk's
work is entirely hidden behind whatever else that loop is doing.

**Resolved in section 5: the rollout is the observation and augmentation
pipeline.**

## 2. Four proposed transformer optimizations, all measured dead

From an optimization review of the network code. Every one was checked on
this card at this shape rather than argued.

| Proposal | Verdict | Evidence |
| --- | --- | --- |
| Fused cuDNN attention (`jax.nn.dot_product_attention`) | **no effect** | 16.087 s vs 16.105 s base |
| Fuse q/k/v into one GEMM | **no effect** | 16.113 s vs 16.105 s base |
| Both together | **no effect** | 16.094 s |
| Remove the `silu` | **no effect** | 16.061 s vs 16.078 s base |
| Kill per-Linear output transposes | **already free** | optimized HLO emits `bitcast`, not `copy` |
| `Precision.HIGH` selecting 3-pass bf16 | **does not happen** | no `algorithm=` on any dot |

Notes that matter for anyone revisiting these:

- cuDNN flash attention **does work** on sm_120 at head_dim 48 — it compiled
  and ran, no fallback. It simply buys nothing here. (S tier at head_dim 44
  would be ineligible: cuDNN needs `head_dim % 8 == 0`.)
- The optimized HLO contains **96 real bf16 transposes** in the attention head
  plumbing, 21.3 GB of transpose output per scan step. Replacing them all with
  flash attention changed the wall clock by 0.1 %, so they are fused into
  neighbours or hidden — a reminder that HLO op counts are not costs.
- Only **64 `__cublas$lt$matmul`** custom calls appear (4 per layer: q, k, v,
  out). The FF GEMMs are inside fusions.

## 3. Reading this against the earlier phase profile

[joe-train-phase-profile.md](joe-train-phase-profile.md) reported, on a T4 at
S tier, that "the network is 99.9 % of the rollout". That measurement stands
for that configuration, but **it does not generalize**, and this file is the
correction: on bf16-native silicon at production shape the network is ~0 % of
the rollout. Two things inflated it there — the 17x Turing bf16 penalty, and
a 32x smaller batch.

Treat "which part dominates" as a property of (card, tier, shape), not of the
code.

## 5. The rollout is the observation + augmentation pipeline

A second box (vast 48931179, same PRO 6000 WS class) bisected the rollout by
switching stages off inside a faithful replica of `collect_rollout`'s scan.
The `states` carry chain is intact in every arm, so nothing is hoisted or
dead-coded. Two of three rounds shown; they agree.

| arm | T=128 | T=64 |
| --- | --- | --- |
| full (everything, stacks all outputs) | 19.58 / 20.06 s | 10.00 / 10.09 s |
| nostack (same compute, scan returns one scalar) | 20.10 / 20.39 s | 10.15 / 10.22 s |
| envonly (`env.step` only, fixed action) | **0.016 s** | — |

Reading it:

- **`env.step` is 0.016 s** for 128 steps x 2048 envs — 0.08 % of the rollout.
- **Stacking the outputs is free.** Removing 16.80 GiB of stacked writes makes
  the rollout *slightly slower*, not faster. The obvious suspect is dead.
- **The network is ~0 %** (section 1).
- Everything else — `_observe_both` (two `get_observation`, two
  `build_cost_grid`, `obs_to_array`, the move and build masks) plus
  `augment_obs` — is therefore **~19.5 s of the ~19.6 s**, about
  **153 ms per scan step**.

A later arm split off the augmentation (vast 48935672, same card class,
two rounds):

| stage | seconds | share of rollout | status |
| --- | --- | --- | --- |
| `augment_obs` | 0.20 / 0.38 | 1.0 % / 1.9 % | established |
| `env.step` | 0.017 | 0.09 % | established |
| output stacking | negative | free | established |
| transformer blocks | ~0 | 0.09 % | established |
| **everything else** | **~18.9** | **~98 %** | **not attributed** |

**The augmentation is not the cost.** `augment_obs` is our largest single
function at 142 lines and is worth 1-2 %, an upper bound at that (the
`full` arm runs first in each round and carries the warm-up).

**What the remaining ~98 % is has NOT been established.** An earlier
revision of this file assigned it to `_observe_both`. That was wrong. The
`envonly` arm sets `do_obs=False` **and `do_net=False`**, so the difference
against it removes the observation pipeline *and* the network together. The
trunk ablation only removed the 16 transformer blocks -- the patch embed,
the policy and value heads, and the categorical sampling over 4410 logits
per sample were never isolated. The bucket therefore holds:

  observations + masks  +  patch embed  +  policy/value heads  +  sampling

Arithmetic says neither half explains it. `get_observation` is roughly
9,700 element-ops per call over a 21x21 board (a 9-slice visibility stack,
four reductions, nine masked multiplies); at 524,288 calls per rollout that
is 5.1 G element-ops, or about **5 ms** at a plausible GPU rate -- against a
measured bucket of **18,900 ms**, a factor of 3,700. The sampling side is
2.3 G element-ops and is no better an explanation. So the cost is
structural, not arithmetic, and it is still unidentified.

The experiment that settles it needs four arms that all keep stacking on and
differ one stage at a time: env only; + observations; + augmentation;
+ network. `env -> obs` gives the observation pipeline and `obsaug -> full`
gives the whole network including the heads.
- It scales **linearly** in steps (T=64 -> T=128 is 1.96x), so it is a
  per-step cost, not an O(T^2) accumulator pathology.

### This refutes the T4 drill-down

[joe-train-phase-profile.md](joe-train-phase-profile.md) reported env +
observations + augmentation at **0.10 s**, 0.1 % of a rollout. That harness
reduced each stage into a scalar accumulator, so XLA elided most of the work;
here `obs_aug` feeds the network, then `env.step`, then the `states` carry,
so it cannot be elided. **The 0.1 % figure is an artifact and should not be
used.** The observation pipeline is not free — it is the rollout.

## 6. Where the optimization budget should go

1. **The unattributed ~98 % of the rollout** — ~54 % of the training
   iteration. It is *not* `augment_obs` (1-2 %), *not* `env.step` (0.09 %),
   *not* the output stacking (free), and *not* the transformer blocks
   (0.09 %). It is the observation pipeline, the network's embed, heads and
   sampling, or something structural across them. Run the four-arm
   experiment above before spending any effort on a candidate.
2. **The PPO update is the transformer**, 98.3 % of 13.6 s. The levers that
   scale it directly are `adv_top_frac` (0.25 today, linear in kept samples)
   and `minibatch_size` / epoch count. Kernel-level attention rewrites are
   measured dead.
3. **Nothing in the attention block.** Four independent rewrites, no effect.
4. **Nothing in `env.step`, and nothing in the rollout's output stacking.**
   16 ms and zero respectively.

Scripts: [`scripts/joe_gpu_rollout_bisect.py`](../../../scripts/joe_gpu_rollout_bisect.py);
raw log `joe-pro6000-rollout-bisect.log`.
