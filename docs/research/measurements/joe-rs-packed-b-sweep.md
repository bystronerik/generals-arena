# joe-rs packed-B sweep — strip-major weights, f16 refuted, a noalias trap

Measured 2026-08-20, the third kernel round of the day (after the
[AVX2 kernel](joe-rs-gemm-kernel-sweep.md) and the
[residual sweep](joe-rs-residual-sweep.md)). Two candidate levers: pack
`B` strip-major so the kernel reads it strictly forward, and store weights
as f16 expanded by F16C in-kernel. Raw records:
[joe-rs-packed-b-sweep.json](joe-rs-packed-b-sweep.json).

**Why packing matters:** the row-major `(k, n)` walk puts consecutive `k`
steps 6 KB apart on the 1536-wide FF layers — past what the hardware
prefetcher tracks. Packing each 16-column strip contiguous (tail columns
as one `k × n_tail` block) costs nothing per turn: weights pack once at
load, and the attention K/V panels pack at the same cost the old
transpose copies paid. Accumulation order is unchanged, so the result is
**bit-identical** — replies over the full 2,400-turn `synthetic-long`
stream match the pre-change binary byte for byte.

## Microbench, exact graph shapes (Modal x86, one container)

| kernel | GEMM ms/forward |
| --- | ---: |
| shipped AVX2 (row-major B) | 29.89 |
| **packed f32** | **18.00 (1.66×)** |
| packed f16 (F16C converts in-kernel) | 18.65 |

Every shape got faster from packing — qkvo (8.97 → 5.67) as much as FF
(20.34 → 11.91), so the strided walk was never an FF-only problem. **f16
is refuted for speed**: once packing fixes the streaming, the two
`vcvtph2ps` per `k` step cost more than the halved bandwidth saves. Not
adopted (it would still halve resident weight memory, but 2 GB makes that
worthless).

## The noalias trap

The first integration passed the packed data to the safe kernel as
`&PackedB`. A slice parameter carries LLVM `noalias`; a Vec data pointer
loaded through a struct reference does not, and without it LLVM refuses
to autovectorize the kernel — the identical body measured **6.5× slower**
(NEON e2e 175 ms instead of 26.6). The intrinsics path is immune, its
loads being manual — so Modal x86 looked fine while every arm64 run was
broken, and the standalone microbench (slice parameters throughout)
could not see it either. The fix is one line of shape: the dispatcher
unwraps `&b.data` once and both kernels take `bp: &[f32]`. Standing
rule: after touching kernel signatures, e2e-bench the crate build on
both architectures, not just the microbench.

## End-to-end, HEAD vs packed, same container, A/B/A/B

| run | pristine p50 | packed p50 | ratio |
| --- | ---: | ---: | ---: |
| final tree | 29.48 / 29.53 | **16.98 / 17.34** | **1.72×** |
| pre-noalias-fix tree (intrinsics path unaffected) | 31.28 / 31.39 | 19.74 / 19.91 | 1.58× |

Replies identical in both containers. Startup also drops (~159 → 108 ms):
load-time packing replaced the old transpose. On the arm64 dev box the
fixed build measures **26.64 ms p50 against 34.55 (1.30×)** — better than
the hot-cache microbench predicted (1.12×), because the real forward
streams its weights cold, where the layout matters more. The per-turn
attention fills use `fill_row`/`fill_col`; the per-element `set` variant
cost ~0.5 ms/turn on arm64.

## Ledger for 2026-08-20

Same-host ratios, compounded: AVX2 kernel 1.23× → residual sweep 1.16× →
packed B 1.58–1.72× ≈ **2.3–2.5×** against this morning's 46.49 ms M7F4
baseline — ~17–20 ms p50 depending on fleet generation, or 7–8 forwards
per 150 ms move from 3 at the start of the day, with every reply
bit-identical throughout.
