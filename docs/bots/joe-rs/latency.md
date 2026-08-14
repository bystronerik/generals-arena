# joe-rs latency

Budget: 150 ms per move, one x86 core; J3's target was full-path p99 ≤ 50 ms
and R1's candle tripwire p99 > 75 ms (port-plan §5, §9).

## Measurement

`joe-rs bench < game.in.log` replays a recorded wire log through the full
per-move path — frame parse, obs pipeline, candle forward, greedy decode,
reply encode to a sink — and reports per-turn percentiles. The input is
`synthetic-long.in.log` (1,572 turns at step 6000, deep late-game states;
it was 1,320 turns at step 5000 — the file is rebuilt from whatever the
longest corpus game is, see [parity.md](parity.md)).

`scripts/joe_rs_modal_bench.py` runs it on a Modal 1-core x86 container
(cargo 1.97.1, `target-cpu=x86-64-v3` from the crate's `.cargo/config.toml`),
mirroring the Phase 2 method. Modal is a proxy, not the target — fleet
generations differ between runs.

## Results (2026-08-15, dependency-free `gemm.rs` path)

| host | p50 | p90 | p99 | max | startup |
| --- | --- | --- | --- | --- | --- |
| Modal x86, 1 core | 21.4 ms | 22.3 ms | 23.7 ms | 34.5 ms | 164 ms |
| dev arm64 (M-series, unpinned) | 22.2 ms | 22.9 ms | 24.0 ms | 29.0 ms | 62 ms |

Raw record: `docs/research/measurements/joe-rs-latency-modal.json`.
The container built the crate from source in 12.4 s — the same build a
sandbox intake would run, against 71–131 s for the old 93-crate graph.

## Verdict

**The bespoke path holds the budget** (2026-08-15): x86 one-core p99 is
23.7 ms — under the old 50 ms J3 target, a third of the 75 ms tripwire, a
sixth of the 150 ms limit, and marginally *faster* than candle was on the
same method. R1's fallback was taken for intake (the vendored build is what
qualification rejected — [packaging.md](packaging.md) §9), not latency, and
it cost nothing on x86.

On arm64 the in-house kernel is ~2.4× slower than candle was (22.2 vs
9.4 ms p50) — candle's aarch64 GEMM microkernels are hand-tuned asm, and
`gemm.rs` is safe Rust that leans on LLVM autovectorization. That gap does
not exist on the x86-64-v3 target the competition runs, and the dev-Mac
number still clears every budget, so it is recorded rather than chased.

## Results (2026-08-14, candle — superseded)

| host | p50 | p90 | p99 | max | startup |
| --- | --- | --- | --- | --- | --- |
| Modal x86, 1 core | 24.2 ms | 24.5 ms | 25.4 ms | 35.6 ms | 1.07 s |
| dev arm64 (M-series, unpinned) | 9.4 ms | 10.0 ms | 11.7 ms | 27.5 ms | 40 ms |

The verdict then was "candle stays (R1 not triggered)": p99 half the 50 ms
target, ~2.4× the Python baseline (p50 9–11 ms / p99 ≈ 19 ms), inside "3×
is acceptable for v1". Kept because the numbers ruled the latency half of
R1 out — what un-ruled candle was the intake build, not this table.
