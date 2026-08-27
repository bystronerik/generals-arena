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

## 1. The headline: the transformer blocks are both halves

| | base | trunk removed | trunk's share |
| --- | --- | --- | --- |
| **Rollout** | 48.325 s | 1.273 s | **97.4 %** |
| **PPO update** | 13.608 s | 0.224 s | **98.3 %** |

The 16 transformer blocks are essentially the entire training iteration, on
both sides of it. Everything else is rounding error: the observation
pipeline, the augmentation, `env.step` and the output stacking together
cost **0.289 s of a 48 s rollout (0.6 %)**, and the network's non-block path
-- patch embed, temporal encoder, policy and value heads, the mask add and
the categorical sampling over 4410 logits -- costs **0.983 s (2.0 %)**.

Full four-way split of the rollout (vast 48941959, on-demand RTX PRO 6000,
three rounds within 1 %, all arms in one process on one machine):

| stage | seconds | share |
| --- | --- | --- |
| env + observations + augmentation + stacking | 0.289 | 0.6 % |
| network minus blocks | 0.983 | 2.0 % |
| **16 transformer blocks** | **47.052** | **97.4 %** |

### An earlier revision of this file said the opposite. Here is why.

It reported the rollout trunk share as **0.09 %** -- that removing all 16
blocks left the clock unchanged. That measurement was invalid.
`collect_rollout` is decorated `@jax.jit`. Wrapping it in a fresh `pmap`
per arm still hits the **nested jit cache**: the cache key is the callee's
identity plus static args and input avals, none of which change when a
Python method is monkeypatched. Every arm therefore ran the *first* arm's
compiled binary, and all arms timed identically -- which reads as "this
change does nothing" for any change.

Minimal reproduction (CPU, no GPU needed):

```python
class Layer:
    def __call__(self, x): return x * 2.0
L = Layer()

@jax.jit                       # <- like collect_rollout
def jitted_caller(x): return L(x)

def fresh_wrapper():           # <- like build(arm): new fn, fresh jit
    def f(x): return jitted_caller(x)
    return jax.jit(f)

a = fresh_wrapper()(x)                       # 2.0
Layer.__call__ = lambda self, y: y * 100.0
b = fresh_wrapper()(x)                       # 2.0  <- patch ignored
```

With the callee *not* jitted the patch applies normally. That is the
difference between the invalid runs and the valid ones.

**Invalidated by this bug** (all called the jitted `collect_rollout`):

- the rollout trunk ablation ("blocks are 0.09 %") -- the truth is 97.4 %;
- the attention A/B: cuDNN fused attention, the q/k/v GEMM fusion, and both
  together "showed no effect". Nothing was tested. **These are live
  candidates again**, and they now sit on top of 97 % of the rollout;
- the `no_silu` arm.

**Not affected** (the callee is not jitted, or the arms differ in a static
field that is part of the cache key):

- the PPO trunk ablation -- `ppo_update` is a plain function, and its
  13.6 s -> 0.22 s result always contradicted the rollout ablation;
- the four-arm and three-arm rollout splits, which inline the scan body
  rather than calling `collect_rollout`;
- the `use_bf16` A/B -- `use_bf16` is `eqx.field(static=True)`, so it is
  part of the treedef and gives each arm its own cache entry;
- the HLO analysis, which needs no execution at all, and which pointed at
  the network from the start.

## 2. Proposed transformer optimizations: status

| Proposal | Verdict | Evidence |
| --- | --- | --- |
| **Fused cuDNN attention** | **+5.1 % of the rollout** | 22.078 s vs 23.214 s base |
| **Fuse q/k/v into one GEMM** | **-2.0 %, slightly harmful** | 23.691 s vs 23.214 s base |
| Kill per-Linear output transposes | already free | optimized HLO emits `bitcast`, not `copy` |
| `Precision.HIGH` selecting 3-pass bf16 | does not happen | no `algorithm=` on any dot |
| Remove the `silu` | untested | the invalid A/B; never retried |

Rerun on vast 48944567 (on-demand RTX PRO 6000 Workstation, X16, 2048 envs
x 64 steps, two rounds), with the harness fixed and a guard that compares
**compiled programs** rather than numbers:

| arm | round 0 | round 1 | vs base | cublasLt GEMMs | cuDNN FMHA |
| --- | --- | --- | --- | --- | --- |
| base | 23.214 | 23.576 | 1.000x | 98 | 0 |
| cudnn | 22.078 | 22.323 | **1.051x** | 98 | **16** |
| qkv | 23.691 | 23.893 | **0.980x** | **66** | 0 |

The signatures confirm each arm did what it claimed: the cuDNN arm gained
16 `__cudnn$fmha` custom calls, one per layer, and the q/k/v arm dropped
98 GEMMs to 66 -- exactly 2 fewer per layer, three projections folded into
one. Both rounds agree.

**cuDNN flash attention is worth ~5 % of the rollout.** Modest, but it sits
on top of the 97.4 % that the blocks own, and the same block code runs in
the PPO backward (98.3 %), so the iteration-level win is plausibly similar;
that half is not measured.

**Folding q/k/v is slightly harmful.** Removing 32 GEMM calls per rollout
made it 2 % slower, consistently across both rounds. Fewer, larger GEMMs
lose here -- the per-call weight `concatenate` and the wider N=1152 kernel
cost more than the launches they save. Drop this idea.

Caveats: cuDNN changes the numerics (value sum -2800.2151 against
-2801.2231), so it changes the training trajectory and shifts the joe/joe-rs
parity reference. It also needs `head_dim % 8 == 0`, which X16, M and M7F4
satisfy at 48 but **S tier does not** at 44.

### How the first attempt got this wrong

The 2026-08-27 A/B reported all arms within 0.16 % and concluded "no
effect". It called the `@jax.jit` `collect_rollout`, so the nested jit
cache served the first arm's binary to every arm. The guard here would have
caught it: the three arms now hash to three distinct programs, and an
earlier revision of this run **aborted** rather than report a false null.

One wrinkle worth keeping: the guard originally compared numeric output and
flagged the q/k/v arm as a failure, because folding q/k/v concatenates along
the **output** dimension rather than the contraction dimension, so every
element keeps the same dot product and the same accumulation order and the
result is bit-identical *by construction*. Comparing compiled programs is
the correct test; comparing numbers is not.

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

### Resolved: the network, not the observations

The four-arm run (vast 48940477, on-demand RTX PRO 6000 Workstation, three
rounds, stacking ON in every arm so it cancels):

| arm | r0 | r1 | r2 | delta = stage |
| --- | --- | --- | --- | --- |
| env | 0.062 | 0.061 | 0.062 | env.step + stacking **0.061 s (0.1 %)** |
| + observations | 0.118 | 0.119 | 0.120 | observation pipeline **0.057 s (0.1 %)** |
| + augmentation | 0.285 | 0.290 | 0.293 | `augment_obs` **0.167 s (0.4 %)** |
| + network | 47.207 | 47.749 | 47.978 | network **46.92 s (99.4 %)** |

**The rollout is the network. The observation pipeline is 0.1 %.** An
earlier revision of this file put it at 98.9 %; that was wrong, and this is
the measurement that settles it. The observation and augmentation pipelines
together cost 0.22 s of a 47 s rollout.

Two independent methods agree. The optimized HLO of the *real*
`collect_rollout` attributes **47.5 GB/step of kernel output across 205
kernels** to the network -- including sixteen 654 MB `silu` fusions on the
feed-forward hidden (`bf16[1536,4096,52]`, one per layer) -- against 0.07
GB/step over 43 kernels for `env.step` and 1-3 near-empty kernels for
`get_observation`. The HLO also shows the scan accumulator update is a
`dynamic_update_index_in_dim` writing **in place** into the aliased 18 GB
buffer, which is why removing the stacking never helped.

### Still open: which part of the network

The trunk ablation (section 1) removed all 16 transformer blocks and moved
the rollout by 0.09 %. Taken with the 99.4 % above, that puts the cost in
the network's **non-block** path -- patch embed, temporal encoder, policy
and value heads, the action-mask add, and the categorical sampling over
4410 logits per sample.

That inference chains two runs on two machines and should not be trusted
yet. The trunk ablation measured `full` at 16.078 s where this run measures
47.2 s -- a 2.9x spread between two PRO 6000 variants (Max-Q vs
Workstation), wider than the documented fleet spread, and the two used
different harnesses (the real `collect_rollout` vs the replica here). It
also sits awkwardly against the sixteen 654 MB per-layer `silu` fusions the
HLO shows inside the blocks.

**The next run must contain both arms in one process**: full, blocks-off,
and network-off together. Until then, "the network" is established and
"which part of the network" is not.

## 6. Where the optimization budget should go

1. **The 16 transformer blocks** — 97.4 % of the rollout and 98.3 % of the
   PPO update, so ~97 % of the whole training iteration. There is no second
   target. Attention rewrites (fused cuDNN attention, q/k/v fusion) are
   measured at **+5.1 % of the rollout for cuDNN flash attention** and
   **-2.0 % for the q/k/v fusion**, which should be dropped. The cuDNN win
   is untested on the PPO backward, where the same blocks are 98.3 %.
   `adv_top_frac` and `minibatch_size` scale the PPO half directly.
2. **The PPO update is the transformer**, 98.3 % of 13.6 s. The levers that
   scale it directly are `adv_top_frac` (0.25 today, linear in kept samples)
   and `minibatch_size` / epoch count. Kernel-level attention rewrites are
   measured dead.
3. **Nothing in the attention block.** Four independent rewrites, no effect.
4. **Nothing in `env.step`, and nothing in the rollout's output stacking.**
   16 ms and zero respectively.

Scripts: [`scripts/joe_gpu_rollout_bisect.py`](../../../scripts/joe_gpu_rollout_bisect.py);
raw log `joe-pro6000-rollout-bisect.log`.
