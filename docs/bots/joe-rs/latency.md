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

`joe-rs bench --stages` replays the same log and reports the same total, split
per stage — the table on stderr, one JSON object on stdout. The plain form
prints exactly what it printed before, so every number below stays comparable.

`scripts/joe_rs_modal_bench.py` runs it on a Modal 1-core x86 container
(cargo 1.97.1, `target-cpu=x86-64-v3` from the crate's `.cargo/config.toml`),
mirroring the Phase 2 method. Modal is a proxy, not the target — fleet
generations differ between runs.

## Results (2026-08-15, dependency-free `gemm.rs` path)

| host | p50 | p90 | p99 | max | startup |
| --- | --- | --- | --- | --- | --- |
| Modal x86, 1 core | 21.4 ms | 22.3 ms | 23.7 ms | 34.5 ms | 164 ms |
| dev arm64 (M-series, unpinned) | 22.2 ms | 22.9 ms | 24.0 ms | 29.0 ms | 62 ms |

## Results (2026-08-18, depth 7 / tier M7)

The lineage grew from depth 5 to depth 7 (`joe-M7-vast-20260818-1741`), which
is growth-plan §7's gate 2: the grown net is training-only if the turn budget
breaks. It does not break.

| host | p50 | p90 | p99 | max | startup |
| --- | --- | --- | --- | --- | --- |
| dev arm64 (M-series, unpinned) | 31.15 ms | 31.39 ms | 32.14 ms | 47.89 ms | 73.5 ms |

## Results (2026-08-19, ff x4 / tier M7F4)

The feed-forward width grew from x3 to x4 (`joe-M7F4-vast-20260819-0207`).
The ff4 growth plan §6.2 had already cleared the shape on Modal x86 against a
synthetic artifact; this row is the confirmation on the real exported one.

| host | p50 | p90 | p99 | max | startup |
| --- | --- | --- | --- | --- | --- |
| dev arm64 (M-series, unpinned) | 36.65 ms | 36.76 ms | 37.87 ms | 69.87 ms | 84.4 ms |

2,400 turns of the M7F4 `synthetic-long.in.log`. p99 37.87 ms lands within 2%
of the 37.2 ms the plan predicted for this host, so the real artifact behaves
as the synthetic one did: 25% of the 150 ms move budget, 76% of J3's 50 ms
target, tripwire 75 ms untouched. Against ff x3 on the same host, p99 32.14 ->
37.87 ms (1.18x), matching the predicted ~+20% forward.

The `max` of 69.87 ms is the one number worth watching: it is the largest
single-turn figure this bot has recorded, and it sits at 93% of the 75 ms
tripwire. p50 and p99 are 2 ms apart, so this is one outlier turn on an
unpinned dev box rather than a shifted distribution — the Modal x86 run in the
plan saw max 46.1 ms. Re-read it there before treating it as real.

1,838 turns of the M7 `synthetic-long.in.log`. Against the depth-5 arm64 row
above: p50 1.40x, p99 1.34x — the growth plan predicted ~+40% forward cost and
that is what arrived. p99 is 21% of the 150 ms move budget and 64% of J3's
50 ms target; R1's 75 ms tripwire is not near. Startup grew 62 -> 73.5 ms on a
35% larger artifact.

Not yet measured on Modal x86 one core, which is the authoritative host and ran
slightly *faster* than this arm64 dev box at depth 5 (p50 21.4 vs 22.2 ms).
Run `scripts/joe_rs_modal_bench.py` to confirm before a rated round.

Raw record: `docs/research/measurements/joe-rs-latency-modal.json`.
The container built the crate from source in 12.4 s — the same build a
sandbox intake would run, against 71–131 s for the old 93-crate graph.

## Where the move goes (2026-08-16)

`joe-rs bench --stages` replays the same log and splits the move. The forward
is not most of it; it is essentially all of it. Modal x86, one core, 2,280
turns of `synthetic-long.in.log` at step 23500 — the raw record is
`docs/research/measurements/joe-net-n0-latency-modal.json`.

| stage | mean ms | p50 | p99 | share of mean |
| --- | ---: | ---: | ---: | ---: |
| parse | 0.012 | 0.010 | 0.027 | 0.06% |
| raw + cost | 0.010 | 0.009 | 0.026 | 0.05% |
| mask | 0.008 | 0.007 | 0.019 | 0.04% |
| `augment_obs` | 0.045 | 0.042 | 0.087 | 0.21% |
| normalize | 0.004 | 0.003 | 0.007 | 0.02% |
| **forward** | **20.879** | **20.811** | **23.010** | **98.2%** |
| decode + reply | 0.003 | 0.003 | 0.004 | 0.01% |
| value on stderr | 0.291 | 0.274 | 0.514 | 1.37% |
| total | 21.251 | 21.197 | 23.444 | — |

Each stage is sorted on its own, so the p99 column does not sum: the turn with
the slow forward is rarely the turn with the slow parse. Only the mean column
adds up.

Two readings worth keeping.

**The whole observation pipeline is 0.067 ms.** Parse, raw, cost, `augment_obs`
and normalize together are a third of a percent of the move. Anything that
proposed to save time by not rebuilding the observation would be optimizing
0.3% of a move.

**The free-eval telemetry costs four times what `augment_obs` costs.** One
unbuffered `eprintln!` per turn is 0.274 ms p50 — larger than every real
computation outside the forward, put together. It is 1.4% of the move against a
150 ms limit, so it is recorded rather than removed: taking it out would fork
the content hash for a number that does not matter yet.

The same split on dev arm64 puts the forward at 22.66 ms p50 of a 22.73 ms
move (99.7%) and `augment_obs` at 0.015 ms — the shape is the same and the obs
pipeline is, if anything, cheaper relative to the forward.

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
