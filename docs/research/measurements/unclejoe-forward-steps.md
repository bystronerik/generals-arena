# unclejoe forward pass — the 25-step split on x86

Measured 2026-08-27. `bench --stages` has always answered the move-level
question the same way — the forward pass **is** the move, 99.0–99.6 % of it —
so this splits that one cell into the 25 steps the network actually runs.
Raw record: [unclejoe-forward-steps-modal.json](unclejoe-forward-steps-modal.json).

- Bot: `bots/unclejoe` (X16, depth 16, 29,551,834 params, argmax T = 0)
- Harness: `unclejoe bench --forward-stages`, added for this measurement
- Runner: [`scripts/unclejoe_modal_forward_steps.py`](../../../scripts/unclejoe_modal_forward_steps.py)
- Input: 1,062 turns of the recorded `synthetic-long` wire stream
- Hosts: Modal, **two batches of four single-CPU containers**, one core each,
  `cargo` 1.97.1, `target-cpu=x86-64-v3`, binary compiled once in the image so
  all eight containers ran the same bytes

## Verdict: it is 85 % GEMM and 9 % `exp`, and nothing else is 1 %

| group | AMD 175/1 (avx2, n=5) | AMD 175/17 (avx512, n=1) | Intel 6/85 (avx512, n=2) |
| --- | ---: | ---: | ---: |
| feed-forward GEMMs (`ff1` + `ff2`) | 56.7–56.9 % | 56.5 % | 52.9 % |
| token GEMMs (`q`, `k`, `v`, `attn_out`) | 28.6–29.0 % | 28.4 % | 25.3 % |
| attention GEMMs (`scores`, `context`) | 2.6–2.7 % | 2.7 % | 3.5 % |
| **all GEMM** | **88.8–89.1 %** | **88.5 %** | **82.5–82.6 %** |
| `exp` (`softmax` + `silu`) | 8.5–8.7 % | 9.2 % | 14.0 % |
| layernorms (`norm1`, `norm2`, `norm_out`) | 0.8 % | 0.8 % | 1.0 % |
| attention plumbing (`head_pack`, `scale`, `head_scatter`) | 0.8 % | 0.8 % | 1.2–1.3 % |
| temporal encoder | 0.4–0.5 % | 0.4 % | 0.6 % |
| residual adds + `assemble` | 0.2–0.3 % | 0.2 % | 0.3 % |
| `patchify` + `unpatchify` | 0.1–0.2 % | 0.1 % | 0.2 % |

Two things follow, and neither was visible from the move-level split.

**The AVX2 GEMM is issue-bound; the AVX-512 one is not.** The four 384×384
token projections and the two feed-forward GEMMs sustain 78–107 GFLOP/s. Do
**not** turn that into a percentage of roofline — see the clock warning
below. The disassembly answers the question the ratio cannot. Compiled for
`x86-64-v3` and read at the hot block:

* **AVX2** (inlined into `gemm_bias`; `target-cpu=x86-64-v3` supplies the
  features, so it needs no out-of-line call): 2 k-steps per iteration as
  **16 `vfmadd231ps` + 8 `vbroadcastss` + 4 `vmovups`** and six ops of loop
  and address overhead. All eight ymm accumulators stay in registers —
  **zero spill stores in the whole function**. Eight FMAs per k-step against
  a 2-per-cycle core is 4 cycles, and there is nothing in the block to
  delete. This is the fleet's majority path and it is finished.
* **AVX-512** (out of line — `avx512f` is not in `x86-64-v3`, so it cannot
  inline): per k-step **13 `vfmadd231ps` with `{1to16}` embedded broadcast +
  1 `vmovups`**, but also **nine address-arithmetic ops**, five of them a
  serial `addq %r10, %r13` chain re-deriving the 13 row pointers every
  iteration. 13 FMAs would be 6.5 cycles; the block issues ~27 instructions.
  It is front-end and address bound, not FMA bound, which is the one place
  in either kernel with room in it.

So there is no factor of two hiding in the AVX2 kernel. What is left is the
AVX-512 tile on the three hosts that expose it, and the ~11 % that is not
GEMM.

**`exp` is the whole remainder.** `softmax` and `silu` together cost four to
five times every layernorm, copy, residual and reshape in the pass combined.
`softmax` alone (4.2–7.1 %) costs **more than the two GEMMs it sits between**
— `scores` + `context` are 2.6–3.5 %. Per head-block `softmax` moves 2,704
floats in 11.4–31.1 µs, roughly 12–36 cycles an element at the clocks these
hosts report; `silu` moves 79,872 floats at roughly 3–9 cycles each. Both run the
same `exp_poly`, so the gap is `softmax`'s own shape: a serial max reduction,
a serial `f32` sum chain per 52-wide row, and a true divide per element —
all of them op-order commitments that exist to hold tier-2 parity with the
candle path, not because the arithmetic needs them.

## Every step

Mean ms per turn, and the step's share of the forward pass. A step that runs
once per block or once per head per block runs 16 or 128 times a turn; the
number is the **sum over one turn**, so `mean / calls` is one invocation.

| step | calls/turn | AMD 175/1 avx2 (n=5) | AMD 175/17 avx512 (n=1) | Intel 6/85 avx512 (n=2) |
| --- | ---: | ---: | ---: | ---: |
| patchify | 1 | 0.049–0.067 (0.1–0.2 %) | 0.039 (0.1 %) | 0.113–0.119 (0.2 %) |
| embed | 1 | 0.156–0.198 (0.4–0.5 %) | 0.188 (0.6 %) | 0.296–0.298 (0.5–0.6 %) |
| temporal | 1 | 0.155–0.226 (0.4–0.5 %) | 0.118 (0.4 %) | 0.328–0.344 (0.6 %) |
| assemble | 1 | 0.005–0.007 (0.0 %) | 0.004 (0.0 %) | 0.011 (0.0 %) |
| norm1 | 16 | 0.146–0.167 (0.4 %) | 0.134 (0.4 %) | 0.263–0.272 (0.5 %) |
| q_proj | 16 | 2.646–3.171 (7.2–7.3 %) | 2.312 (7.1 %) | 3.397–3.561 (6.3–6.4 %) |
| k_proj | 16 | 2.612–3.140 (7.1–7.2 %) | 2.290 (7.0 %) | 3.405–3.567 (6.4 %) |
| v_proj | 16 | 2.632–3.155 (7.2–7.3 %) | 2.297 (7.0 %) | 3.402–3.556 (6.3–6.4 %) |
| head_pack | 128 | 0.237–0.275 (0.6–0.7 %) | 0.187 (0.6 %) | 0.471–0.512 (0.9 %) |
| scores | 128 | 0.645–0.756 (1.7–1.8 %) | 0.597 (1.8 %) | 1.377–1.443 (2.6 %) |
| scale | 128 | 0.018–0.021 (0.0 %) | 0.017 (0.1 %) | 0.027–0.028 (0.1 %) |
| softmax | 128 | 1.585–1.856 (4.2–4.3 %) | 1.457 (4.5 %) | 3.807–3.980 (7.1 %) |
| context | 128 | 0.326–0.389 (0.9 %) | 0.298 (0.9 %) | 0.479–0.504 (0.9 %) |
| head_scatter | 128 | 0.041–0.048 (0.1 %) | 0.041 (0.1 %) | 0.170–0.199 (0.3–0.4 %) |
| attn_out | 16 | 2.625–3.147 (7.1–7.3 %) | 2.356 (7.2 %) | 3.353–3.504 (6.3 %) |
| attn_resid | 16 | 0.039–0.048 (0.1 %) | 0.032 (0.1 %) | 0.076–0.080 (0.1 %) |
| norm2 | 16 | 0.145–0.166 (0.4 %) | 0.133 (0.4 %) | 0.259–0.272 (0.5 %) |
| **ff1** | 16 | **10.387–12.308 (28.1–28.4 %)** | **9.175 (28.1 %)** | **14.647–15.355 (27.4 %)** |
| silu | 16 | 1.604–1.868 (4.2–4.4 %) | 1.545 (4.7 %) | 3.663–3.844 (6.8–6.9 %) |
| **ff2** | 16 | **10.505–12.382 (28.4–28.7 %)** | **9.258 (28.4 %)** | **13.658–14.287 (25.5 %)** |
| ff_resid | 16 | 0.050–0.063 (0.1–0.2 %) | 0.033 (0.1 %) | 0.090–0.095 (0.2 %) |
| norm_out | 1 | 0.009–0.010 (0.0 %) | 0.008 (0.0 %) | 0.016 (0.0 %) |
| value_head | 1 | 0.011–0.013 (0.0 %) | 0.008 (0.0 %) | 0.024–0.026 (0.0 %) |
| policy_head | 1 | 0.070–0.083 (0.2 %) | 0.077 (0.2 %) | 0.167–0.173 (0.3 %) |
| unpatchify | 1 | 0.003–0.004 (0.0 %) | 0.004 (0.0 %) | 0.010–0.011 (0.0 %) |
| **forward** | 1 | **36.7–43.4 ms** | **32.6 ms** | **53.5–56.1 ms** |

What each name covers is written next to `FORWARD_STEP_NAMES` in
[`bots/unclejoe/src/nn/net.rs`](../../../bots/unclejoe/src/nn/net.rs). The two
that are not obvious: `patchify` also carries the forward's argument asserts
and the scratch borrow, because the clock starts on entry; `value_head` is
its GEMM, the softmax and the bin-centre dot together.

## The GEMMs, per call

| GEMM step | M × K × N | MFLOP/call | AMD 175/1 avx2 µs (GF/s) | AMD 175/17 avx512 µs (GF/s) | Intel 6/85 avx512 µs (GF/s) |
| --- | --- | ---: | ---: | ---: | ---: |
| embed | 49 × 351 × 384 | 13.21 | 155.9–197.7 (67–85) | 188.4 (70) | 296.0–297.6 (44–45) |
| q_proj | 52 × 384 × 384 | 15.34 | 165.4–198.2 (77–93) | 144.5 (106) | 212.3–222.6 (69–72) |
| k_proj | 52 × 384 × 384 | 15.34 | 163.2–196.2 (78–94) | 143.1 (107) | 212.8–222.9 (69–72) |
| v_proj | 52 × 384 × 384 | 15.34 | 164.5–197.2 (78–93) | 143.6 (107) | 212.6–222.3 (69–72) |
| scores | 52 × 48 × 52 | 0.26 | 5.0–5.9 (44–51) | 4.7 (56) | 10.8–11.3 (23–24) |
| context | 52 × 52 × 48 | 0.26 | 2.5–3.0 (85–102) | 2.3 (112) | 3.7–3.9 (66–69) |
| attn_out | 52 × 384 × 384 | 15.34 | 164.1–196.7 (78–93) | 147.2 (104) | 209.6–219.0 (70–73) |
| ff1 | 52 × 384 × 1536 | 61.34 | 649.2–769.2 (80–94) | 573.4 (107) | 915.4–959.7 (64–67) |
| ff2 | 52 × 1536 × 384 | 61.34 | 656.5–773.9 (79–93) | 578.6 (106) | 853.6–892.9 (69–72) |
| policy_head | 49 × 384 × 90 | 3.39 | 70.0–82.8 (41–48) | 77.2 (44) | 166.5–173.4 (20–20) |
| value_head | 1 × 384 × 128 | 0.10 | 10.7–13.2 (7–9) | 7.8 (13) | 24.2–26.1 (4–4) |

One forward is ~3.03 GFLOP. **The GFLOP/s columns are throughput, not
efficiency** — dividing them by a peak computed from the reported clock
gives a number that is wrong and sometimes impossible (below). The small-N
shapes — `scores` (N = 52),
`policy_head` (N = 90), `value_head` (M = 1) — run at a third to a tenth of
the large shapes' rate, which is what a strip-major kernel with a fixed tile
does on a short row; together they are 2.0 % of the pass, so the inefficiency
is real and worth nothing.

## Hosts

Modal masks `model name` — it reads `unknown`. Hosts are identified by
vendor / family / model, the same way the
[AVX-512 gate](joe-rs-avx512-gate.md) did.

| batch | replica | host | MHz | avx512f | dispatch | forward mean ms | move p50 ms |
| ---: | ---: | --- | ---: | --- | --- | ---: | ---: |
| 0 | 0 | AMD family 175 model 1 | 3087 | no | avx2 | 42.56 | 42.35 |
| 0 | 1 | AMD family 175 model 1 | 3050 | no | avx2 | 39.36 | 38.55 |
| 0 | 2 | AMD family 175 model 1 | 2880 | no | avx2 | 36.72 | 36.83 |
| 0 | 3 | AMD family 175 model 17 | 2550 | yes | avx512 | 32.61 | 32.54 |
| 1 | 0 | Intel family 6 model 85 | 3100 | yes | avx512 | 56.06 | 54.58 |
| 1 | 1 | AMD family 175 model 1 | 2450 | no | avx2 | 41.96 | 41.54 |
| 1 | 2 | AMD family 175 model 1 | 3104 | no | avx2 | 43.44 | 43.02 |
| 1 | 3 | Intel family 6 model 85 | 3100 | yes | avx512 | 53.51 | 53.70 |

The fleet spans **32.6–56.1 ms** for identical bytes, a 1.72× range, which is
the [same fleet variance](../../bots/joe-rs/latency.md) every CPU measurement
here has hit. The two Skylake-SP containers are the slowest despite
dispatching AVX-512 — and they are also where `exp` jumps from 8.6 % to
14.0 %, because the AVX-512 GEMMs speed up around scalar `exp` code that does
not. **Read the shares across hosts and the milliseconds only within one.**

## The clock these hosts report is not the clock they run at

`/proc/cpuinfo`'s `cpu MHz` cannot be used to convert these microseconds
into cycles, and an earlier draft of this page did exactly that. The check
that kills it: the AVX2 kernel issues a known number of FMAs for each of
these shapes — 8 per k-step, over `(N/16)·(M/4)·K` tile-steps — and no core
here retires more than 2 FMAs a cycle. Dividing the measured time by that
floor at the reported MHz gives, for `ff1` / `ff2` / `q_proj`:

| host | ratio to the 2-FMA/cycle floor |
| --- | ---: |
| AMD 175/1 b0r2 @ 2880 MHz | **0.975 / 0.986 / 0.994** |
| AMD 175/1 b1r1 @ 2450 MHz | **0.941 / 0.963 / 0.978** |
| the other three AVX2 hosts | 1.11–1.28 |

Six of those cells are below 1.0, which is not a slow kernel but an
impossible one: the reported MHz understates the real clock by at least
2–6 % on those containers, and by an unknown amount everywhere else.
Modal's sandbox also reports 17 online CPUs to a container that was given
one, so the whole `cpuinfo` block is advisory. Every claim on this page is
therefore made in **microseconds and in shares**, both of which are
clock-free, and the "is the kernel good" question is answered from the
disassembly instead.

## What this does not establish

- **Not an A/B of anything.** No code was changed to be faster; this is one
  binary described. A kernel change has to be measured the way the AVX-512
  gate was: same host, interleaved.
- **The absolute milliseconds are a proxy.** Modal is not the competition
  host, and the fleet spread above is larger than most changes worth making.
- **The split costs a little to take.** 816 `Instant::now()` calls a turn,
  and marks standing between loops the compiler could otherwise fuse. Against
  the uninstrumented `forward` cell in the same container it adds
  0.030–0.052 ms on the AMD hosts (0.08–0.12 %) and 0.089–0.093 ms on the
  Intel ones (0.17 %). The report prints that comparison every run.
- **One board.** 1,062 turns of one 18×19 game. The shapes are fixed by
  `pad_to` 21, so board size cannot move the split, but turn-to-turn variance
  in this stream is all that the p90/p99 columns of the raw record see.

## Reproducing

```bash
modal run scripts/unclejoe_modal_forward_steps.py --batches 2
```

Locally, on any recorded wire log:

```bash
JOE_RS_ARTIFACT=bots/unclejoe/artifact bots/unclejoe/target/release/unclejoe bench --forward-stages < game.in.log
```

The table goes to stderr, one JSON object per split to stdout. The played
path is untouched: `Net::forward` delegates to `forward_staged` with
`NoNetSteps`, whose `mark` is an empty inlined method, so the marks compile
away where they are not wanted. Verified after the change — replies over the
committed smoke stream are byte-identical to the golden, 49 crate tests, the
11 `-m joe` parity gates (tier-2 forward and tier-3 decide against the JAX
oracle), and `tools/mutation_check.py` at 18/18 killed.
