# Joe X16 (29.55M) training pilots — memory, throughput, and the vast market (2026-08-24)

Three pilots that price a ~$100 / 5-day training run of the 30M growth target
**X16: depth 16, embed 384, ff ×4, 29,551,834 params** — the shape
`grow_depth` 7 → 16 reaches from the finished joe-M7F4 checkpoint. Total pilot
spend ≈ $4.6. Nothing here is an arena match; every instance is destroyed and
every pilot R2 prefix purged.

## Verdict

1. **X16 does not fit an 80 GB card at the production rollout shape.** At
   2048 envs × 256 steps the loop OOMs on an H100-80GB (a 23.87 GiB single
   allocation, allocator full). `TF_GPU_ALLOCATOR=cuda_malloc_async` fails
   identically — real demand, not fragmentation. Estimated full-shape demand
   ≈ 100 GiB, which also rules out the 94 GB H100 NVL.
2. **`num_steps: 128` fits with slack and costs ~nothing per sample.** Peak
   52.3 GiB; H100 runs it at 13.95 s/iter, **37.6k samples/s** — 8 % under
   the Phase-1 params-linear model (t/sample ≈ 2.30 µs + 0.754 µs/Mparam),
   so that model holds at deep-thin shapes.
3. **The RTX PRO 6000 (Blackwell, 96 GB) works out of the box** —
   `jax[cuda12]==0.11.0` on sm_120, no changes — at **21.8k samples/s**
   (0.58× H100).
4. **The pmap multi-GPU path works and scales linearly at N=2**, and 2×96 GB
   restores the **unmodified production recipe** (1024 envs/device,
   minibatch 1024/device, T=256, ~52 GiB/device): 44.0k samples/s on one 2×
   rig, 34.2k on another (host spread, not scaling loss).
5. **The interruptible market, not the code, is the open risk: three
   outbids inside ~1 hour** across two 2× PRO 6000 rigs, at min_bid + 5 %
   and at 1.5× min. The outbid → `resume` flow itself worked both times.

## Provenance

- Modal (H100-80GB, `scripts/joe_modal_train.py`, tier `M7F4` +
  `--overrides '{"depth": 16, ...}'`, runs `joe-X16-pilot{,2,3}-20260824`
  on the `morpheus-training` Volume, ≈ $2.0): attempt 1 = production shape
  → OOM; attempt 2 = same + `cuda_malloc_async` → identical OOM; attempt 3
  = `num_steps: 128` → 12/12 iterations. The script now prints
  `Peak device memory` and returns `peak_device_gib`.
- vast single-GPU (`scripts/joe_vast_train.py launch --tier M7F4 --gpu
  RTX_PRO_6000_WS,RTX_PRO_6000_S`, run `joe-X16-pro6000-pilot-20260824`,
  instance 48592063, bid $0.44/h, ≈ $0.28): 15/15 iterations at the T=128
  shape.
- vast 2-GPU (same launcher, `--num-gpus 2`, per-device overrides
  `num_envs 1024, minibatch_size 1024`, run `joe-X16-2x-pilot-20260824`,
  ≈ $1.3): instance 48595523 (2× PRO 6000 S, $1.68/h) reached iter 10 and
  was outbid mid-checkpoint-upload; replacement 48596596 (2× PRO 6000 WS,
  $0.84/h) reached iter 5 and was outbid; a re-bid at $1.20/GPU never
  restarted. `eval_ref_*` was cleared in every pilot config (the `ref/`
  objects exist only in real run prefixes).
- Iteration timings are the loop's own `SPS:` lines at steady state
  (iterations 3+; iteration 1 carries compile). Boards, curriculum, and all
  other keys are `M7F4.yaml` verbatim.

## Measured

| Setup | Shape | s/iter | samples/s | peak GiB |
| --- | --- | ---: | ---: | ---: |
| H100-80GB (Modal) | 2048×256 | — | — | OOM (23.87 GiB alloc) |
| H100-80GB (Modal) | 2048×128 | 13.95 | 37,600 | 52.3 |
| RTX PRO 6000 WS | 2048×128 | 24.05 | 21,800 | fits, not printed |
| 2× PRO 6000 S ($1.68/h) | 2×1024×256 (production) | 23.8 | 44,000 | fits |
| 2× PRO 6000 WS ($0.84/h) | 2×1024×256 (production) | 30.7 | 34,200 | fits |

Scaling reads clean: the 2× S rig's per-device pace (23.8 s per 524k
samples) matches the single WS card (24.05 s), so the PCIe all-reduce tax
(~118 MB × ~128 minibatch steps) is invisible; the slower 2× WS rig is host
spread, the fleet's usual ~1.3×. Cold pool generation: 236 s and 100 s and
78 s on the three vast hosts vs ~40 s on Modal H100 — one-off per boot.

Two per-device semantics to respect at N > 1: `num_envs` and
`minibatch_size` are **per device** (`samples_per_iter = num_devices × 2 ×
num_envs × num_steps`; the pmean makes the effective update batch
`minibatch_size × num_devices`). Launching 2× with the stock 2048s would
silently double both. The |adv| top-25 % filter becomes per-device — a
negligible statistical difference, noted for completeness.

## Not proven

The step-10 checkpoint wrote locally at N=2, but **no R2 upload completed
and no N=2 resume-restore ran** — both outbids landed first. Both paths
compose already-proven pieces (device-agnostic upload of locally written
files; restore then the same `_replicate` every boot performs), so the
residual risk is small, but the production run's first checkpoint is the
real proof. An outbid during the upload window costs everything since the
last durable set — the first pilot lost all 10 iterations exactly this way.
Keep `save_every` small early in a run on any bid instance.

## Price snapshots (2026-08-24, storage 80 GB included, volatile)

Interruptible min_bid: 1× H100 $0.60/$0.80 then $1.60+; 1× PRO 6000
$0.168 then $0.42–0.47; 2× PRO 6000 $0.80 total; 2× H100 $2.67 total;
H200 NVL $1.60. On-demand: 1× PRO 6000 S $0.90; 1× A100-80 $1.02;
1× H100 $2.29–2.56; 2× PRO 6000 $2.24; 2× H100 SXM $3.49; H200 $3.35.

Samples per $100 inside the 5-day window, measured SPS where available:

| Lane | Recipe | Samples / $100 | Churn risk |
| --- | --- | ---: | --- |
| 1× H100 int. $0.63–0.84 | T=128 | ~14.4B | unmeasured |
| 2× PRO 6000 int. $0.84 | production | ~14–17B | **3 outbids/hour today** |
| 1× PRO 6000 S on-demand $0.90 | T=128 | **~8.7B** | none |
| 2× H100 SXM on-demand $3.49 | production | ~7.7B (est.) | none |
| 2× PRO 6000 on-demand $2.24 | production | 5.5–7.1B | none |
| 1× H100 on-demand $2.29 | T=128 | ~5.9B | none |

## Reading

On-demand costs a ~40–50 % samples haircut against the quiet-market
interruptible ideal — and after three outbids in an hour, that haircut buys
something real. The launch decision is market timing, not engineering:
measure churn on the target rung (watch its `min_bid` for a few hours)
before committing, and make observed preemption rate a hard criterion at
the ~$20 go/no-go gate — at ~15 min of boot per preemption, more than
~1 per 4 hours burns the window faster than any throughput lever earns it
back.

A concrete opening that defers the bet: day 1 on the on-demand PRO 6000 S
is $21.60 — almost exactly the $20 gate — and delivers ~1.9B samples
(~3,600 T=128 iterations) undisturbed for the schedule/recruitment
verdict. At the gate, `resume` onto whichever interruptible rung has gone
quiet, or stay on-demand and land ~8.7B total. R2 carries the run across
either branch.

## Caveats

- H200, A100-80, and 2× H100 SPS are estimates from the params-linear
  model and the measured 2× scaling, not measurements.
- The launcher takes the top `dlperf_usd-` offer; landing one specific
  cheap machine means renting it and adopting with `--instance-id`.
- Every price above is a same-day snapshot of a thin market (6–27 offers
  per class); the cheapest single H100 on-demand moved from $1.74 to $2.29
  within the session.
- The e2e overhead on top of pure s/iter (evals, checkpoint writes) is not
  re-measured here; the joe-M7F4 run measured 17 % at `eval_every 50` /
  512 games, and the plan assumes ~5–8 % at `eval_every 100` / 256 games.
