# joe-rs residual sweep — exp, LayerNorm lanes, packed attention

Measured 2026-08-20, immediately after the
[GEMM kernel sweep](joe-rs-gemm-kernel-sweep.md) landed the AVX2 tiles.
That work left a non-GEMM residual (~26 % of the forward on the M3
decomposition); this sweep prices its three pieces. Raw records:
[joe-rs-residual-sweep.json](joe-rs-residual-sweep.json).

The variants, each a small `net.rs` change:

- **fastexp** — a Cephes-style polynomial exp (max relative error ~7.6e-8)
  replaces libm `expf` in `silu_in_place` and `softmax_in_place`. The win is
  not the exp itself: a libm call in a loop blocks vectorization of the
  whole loop, so the SiLU divide over 52×1536×7 elements ran scalar.
- **laneln** — the two serial reductions in `LayerNorm::forward_into` split
  into eight accumulator lanes (the attention dot already did this; the LN
  sums serialized on add latency).
- **packattn** — per-head QK^T and context products packed and run through
  `gemm_bias` (now the AVX2 tiles) instead of the hand-rolled lane loops.
- **combined** — all three.

Unlike the kernel change these reassociate float sums, so they are tier-2
material, not bit-identical. Measured divergence: **0/2400 reply diffs**
on the full `synthetic-long` stream on x86 *and* arm64, max per-turn value
delta 0.0001 (the print resolution), the **11-test parity gate passes**
(306 s), and the seed-0 verification match plays the identical game.

## Modal x86, one core, one container, two interleaved rounds (p50 ms)

| variant | round 1 | round 2 | mean | delta |
| --- | ---: | ---: | ---: | ---: |
| pristine (AVX2 kernel) | 24.01 | 25.15 | 24.58 | — |
| fastexp | 23.50 | 23.20 | 23.35 | −1.23 |
| laneln | 23.45 | 23.62 | 23.54 | −1.04 |
| packattn | 22.37 | 21.76 | 22.07 | −2.51 |
| **combined** | **21.09** | **21.26** | **21.18** | **−3.40 (1.16×)** |

p99 moves the same way: 26.3/26.7 pristine → 23.0/22.2 combined. The
pieces are roughly additive (sum −4.8 vs −3.4 realized). `packattn` is the
largest single piece on x86 — the AVX2 tiles beat the lane loops by more
than the packing costs.

## arm64 dev box (p50 ms, single runs)

pristine 36.51 → fastexp 36.22, laneln 35.92, packattn 35.48, combined
**34.55 (−5.4 %)**. Smaller than x86, as predicted: Apple's libm `expf` is
already ~1.4 ns/element, and NEON has no intrinsics GEMM to lean on.

## Ledger for the day

On the same fast-host generation, the forward went 29.3 ms (pre-kernel)
→ 23.8 (AVX2 GEMM) → 21.2 ms (this sweep): **1.38× in one day**, replies
identical throughout. Against the 150 ms move budget that is 7 forwards a
turn at p50 on this generation; the slow-generation multiple is larger
(the kernel alone was 1.54× there) but was not re-measured for this sweep.
