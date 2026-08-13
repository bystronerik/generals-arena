# joe-rs latency

Budget: 150 ms per move, one x86 core; J3's target was full-path p99 ≤ 50 ms
and R1's candle tripwire p99 > 75 ms (port-plan §5, §9).

## Measurement

`joe-rs bench < game.in.log` replays a recorded wire log through the full
per-move path — frame parse, obs pipeline, candle forward, greedy decode,
reply encode to a sink — and reports per-turn percentiles. The input is
`synthetic-long.in.log` (1,320 turns, deep late-game states).

`scripts/joe_rs_modal_bench.py` runs it on a Modal 1-core x86 container
(cargo 1.97.1, `target-cpu=x86-64-v3` from the crate's `.cargo/config.toml`),
mirroring the Phase 2 method. Modal is a proxy, not the target — fleet
generations differ between runs.

## Results (2026-08-14)

| host | p50 | p90 | p99 | max | startup |
| --- | --- | --- | --- | --- | --- |
| Modal x86, 1 core | 24.2 ms | 24.5 ms | 25.4 ms | 35.6 ms | 1.07 s |
| dev arm64 (M-series, unpinned) | 9.4 ms | 10.0 ms | 11.7 ms | 27.5 ms | 40 ms |

Raw record: `docs/research/measurements/joe-rs-latency-modal.json`.

## Verdict

**Candle stays** (R1 not triggered): x86 one-core p99 is half the 50 ms
target and a sixth of the 150 ms limit. The Python baseline on the same
method was p50 9–11 ms / p99 ≈ 19 ms — joe-rs is ~2.4× slower per move on
x86, comfortably inside "3× the Python baseline is acceptable for v1".
Startup (load + warmup, ~1 s worst measured) sits far inside the ~10 s
first-move grace, versus Python's ~4 s JIT compile. No bespoke kernel is
warranted at these numbers; the fallback path (same safetensors contract)
remains documented in the plan if a future net outgrows this.
