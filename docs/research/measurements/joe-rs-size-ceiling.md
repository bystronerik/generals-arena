# joe-rs model-size ceiling — Rust stack latency ladder (Modal, 2026-08-24)

Rust sibling of [joe-size-ceiling-cpu.md](joe-size-ceiling-cpu.md): the same
AverageJoe-ratio size ladder, measured through joe-rs's own `bench --stages`
harness — the AVX2+FMA intrinsics gemm, PackedB strip-major weights, and the
residual-sweep kernels; the exact per-move path the deployed bot runs.

## Verdict

**joe-rs scales linearly with the compute model (no fixed overhead to
amortize), so its ceiling is lower than the jax stack's: XL24 (68.5M params,
9.1× M) reaches p99 145.8 ms on the slower of the two measured hosts —
inside 150 ms by 3%, with single-move maxima up to 172 ms.** The crossing
sits at ~9× M compute on a slow host and ~11× on a fast one. With margin for
the fleet spread, the joe-rs ceiling is **XL20: depth 20, embed 512, 53.9M
params (7.1× M) — p99 113 ms slow-host / 81 ms fast-host.** The shipped
M7F4 runs at p99 28–36 ms, a 4–5× margin.

## Provenance

- Script: `scripts/joe_rs_modal_size_ceiling.py`. Raw records:
  [joe-rs-size-ceiling.json](joe-rs-size-ceiling.json).
- joe-rs pins its shapes as consts by design, so each tier is its own
  build: `EMBED` / `DEPTH` / `FF_DIM` and the manifest `ff_factor` check
  patched by sed at image build, `cargo build --release`, rustc 1.97.1,
  `target-cpu=x86-64-v3`. The repo's `bots/joe-rs` source is untouched.
- Weights are random F32 (f16-rounding is a no-op for latency): a generated
  `joe-net-v1` artifact per tier; a const/manifest mismatch refuses to load,
  so a drifted sed cannot pass silently.
- Input: the recorded `synthetic-long.in.log` (1,062 turns), the stream the
  parity gate replays. 3 passes per tier, tiers interleaved pass-level
  round robin inside one container.
- Containers: strict (cpu limit 1.0, 2 GB) and burst-4 control (4 GB). The
  burst container was **preempted mid-pass and retried on a slower host**,
  so strict landed on a fast host and burst on a slow one — an accidental
  but useful bracket of the fleet. Cross-container numbers are cross-host;
  within-container ratios are the trustworthy part.

## Results — `total` per move, median of 3 passes

Strict, fast host (M anchor: forward p50 10.4 ms):

| Tier | params | scale | fwd p50 ms | total p99 ms | total max ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| M | 8.56M | 1.00× | 10.4 | 24.5 | 88 |
| M7F4 (shipped) | 13.58M | 1.68× | 17.1 | 35.2 | 55 |
| L | 23.40M | 2.99× | 32.1 | 43.3 | 75 |
| L16 | 38.28M | 5.00× | 53.6 | 65.8 | 93 |
| XL20 | 53.92M | 7.11× | 72.0 | 81.5 | 123 |
| XL24 | 68.46M | 9.07× | 101.2 | 113.4 | 158 |

Burst, slow host (M anchor: forward p50 14.2 ms, ~1.4× slower):

| Tier | fwd p50 ms | total p99 ms | total max ms |
| --- | ---: | ---: | ---: |
| M | 14.2 | 17.6 | 24 |
| M7F4 | 23.6 | 28.4 | 35 |
| L | 42.4 | 48.9 | 58 |
| L16 | 70.7 | 82.4 | 91 |
| XL20 | 98.0 | 113.3 | 130 |
| XL24 | 125.9 | 145.8 | 172 |

The M7F4 anchor (fwd p50 17.1 / 23.6 ms across the two hosts) sits inside
the known joe-rs range (21.2 ms post-residual-sweep, fleet spread ~2×), so
both hosts read as ordinary fleet generations.

## Two structural findings

**1. Rust scales linearly; jax scales sublinearly.** On the slow host,
XL24/M = 8.9× per-move against the 9.07× compute model — joe-rs's fixed
per-move overhead is ~0.3 ms, so every extra FLOP is paid in full. The jax
stack pays ~8–10 ms of fixed dispatch per move and bought 9.07× compute for
only 4.7× time. Consequence: **at M-tier sizes joe-rs is the faster stack,
but by XL sizes jax-CPU catches up or wins** (jax XL24 p50 86 ms on a host
whose M cost 18.4 ms). XLA's large-shape gemm blocking beats the joe-rs
kernels, which were tuned at 384-dim shapes and are memory-bound on the FF
gemm already at M7F4. A big-net joe-rs would need a kernel retune
(cache-blocked FF gemms — the known remaining lever), and any cross-stack
choice at a given size needs a same-host A/B, not these two containers.

**2. The CFS throttle artifact is a Python phenomenon, not a quota
phenomenon.** Under the same hard 1-core quota that froze the jax bench for
~500 ms at a time, the pure single-threaded Rust process shows only mild
tail inflation (p99 ≈ 1.2–2× p50, worst single move 158 ms). The jax
freeze came from the runtime's housekeeping threads tripping the CFS
budget, not from the quota being tight per se. For deployment this removes
the strict-host risk for joe-rs almost entirely.

## RAM

Peak RSS, measured locally with `/usr/bin/time -l` over the same bench path
(allocation sizes are arch-independent; the Modal strict container's hard
2 GB cap held throughout as the coarse check):

- XL24: **834 MB** — the 274 MB artifact read plus the packed f32 weights
  dominate. Fits the 2 GB competition cap with 1.2 GB headroom.
- Shipped M7F4: 171 MB.

Latency, not memory, is the binding constraint at every size on the ladder.

## Caveats

- Latency only; nothing here says a bigger net is stronger or prices its
  training.
- Random weights; real weights change nothing about FLOPs. The artifact is
  F32 like the real export (values f16-rounded there, full-range here).
- One wire log (1,062 turns) per pass; game-phase mix is fixed. The jax
  ladder's 3×1,200 random-obs games agree on shape, so this is not doing
  the work alone.
- Strict = fast host and burst = slow host is an accident of a preemption
  retry; the labels are not the interesting contrast here, the hosts are.

## Reproducing

```bash
.venv/bin/modal run scripts/joe_rs_modal_size_ceiling.py > /tmp/joe_rs_size.log 2>&1 &
```
