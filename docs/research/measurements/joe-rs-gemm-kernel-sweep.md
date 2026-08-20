# joe-rs GEMM kernel sweep — safe tiles vs AVX2+FMA intrinsics

Measured 2026-08-20, after the M7F4 re-baseline put the forward at
46.49 ms p50 / 52.43 p99 on Modal x86 — past J3's 50 ms p99 target for the
first time. Question: how much of the forward comes back from a better GEMM
kernel, with the crate still dependency-free for intake?

- Raw records: [joe-rs-gemm-kernel-sweep.json](joe-rs-gemm-kernel-sweep.json)
- Adopted result: the AVX2+FMA MR=4 kernel in `bots/joe-rs/src/nn/gemm.rs`,
  runtime-detected, portable safe kernel as the fallback and reference.

## Method

Every contrast is **same-host and in-run**. Three Modal runs on 2026-08-20
put identical pristine code at 29.1, 46.5, and 77.1 ms p50 — fleet
generations differ that much, so a cross-run delta measures the fleet, not
the code. The microbench sweeps all kernel variants in one process; the
end-to-end contrast builds the pristine and the patched crate in one
container and interleaves `joe-rs bench` runs A/B/A/B.

## Microbench: GEMM per forward, exact graph shapes (Modal x86, one core)

| kernel | GEMM ms/forward |
| --- | ---: |
| baseline safe MR=4 (shipped until today) | 33.25 |
| safe MR=5 / MR=6 / MR=8 | 45.87 / 46.23 / 56.42 |
| **AVX2+FMA MR=4** | **23.59** |
| AVX2+FMA MR=6 | 23.83 |
| AVX2+FMA MR=8 | 45.24 |

Two findings. **The safe-Rust tile sweep is a dead end**: every MR other
than the hand-unrolled 4 is 38–70 % slower — LLVM autovectorizes the
explicit four-slice pattern well and nothing else. The same held on the
arm64 dev box (MR=6 was 34.5 vs 26.1 ms). **The intrinsics tiles are
1.41×** over the baseline at the same MR, and MR=6 buys nothing on top, so
MR stays 4. MR=8 spills registers on both paths, as the ymm budget
predicts.

All variants keep one accumulator per output and walk `k` in order, so
every row of the table produces **bit-identical** output — asserted in-run
before timing.

## End-to-end: `joe-rs bench`, pristine vs patched, same container

| host generation | pristine p50 | patched p50 | ratio | pristine p99 | patched p99 |
| --- | ---: | ---: | ---: | ---: | ---: |
| fast (A/B/A/B, 2 rounds) | 29.14 / 29.50 | 23.53 / 24.05 | 1.23× | 31.4 | 25.8 |
| slow (A/B, 1 round) | 77.13 | 50.11 | 1.54× | 101.1 | 65.0 |

The ratio grows on slower hosts — the intrinsics path relieves compute,
and slow generations are compute-poorer. On the J3 ledger: the patched
p99 was 25.8 ms on the fast host and 65.0 on the slow one, against the
50 ms target and the 75 ms tripwire — which generation the sandbox
matches is unknown, so the target is "usually met, tripwire never hit"
rather than a single number.

Both containers also replayed the full 2,400-turn `synthetic-long` stream
through both binaries: **replies byte-identical**. The kernel change is
invisible to parity by construction, not just within tolerance.

## What did not change

- arm64 dev box: 36.51 ms p50 before and after — the `avx` module compiles
  out and the portable kernel is untouched.
- The crate is still a single dependency-free source build; the intrinsics
  live behind `is_x86_feature_detected!` per port-plan §1.
- Verification gate: the patched bot won a `--mode competition` match
  (general captured, turn 261, seed 0 vs aegis).
