# joe-rs AVX-512 GEMM path — gate and same-host A/B

Measured 2026-08-27. Adds a third runtime-dispatched path to
`bots/joe-rs/src/nn/gemm.rs`: an AVX-512F tile, preferred over AVX2+FMA
when `avx512f` is detected. Raw records:
[joe-rs-avx512-gate.json](joe-rs-avx512-gate.json) (two batches of four
single-core Modal containers, minutes apart).

**Why the tile is MR=13 × NR=16.** One zmm register covers a whole
16-column strip, so a row costs one accumulator instead of the AVX2
tile's two. The packed-B sweep's "MR above 4 loses" was a 16-register
ymm budget and does not carry over: 13 accumulators + one B vector + one
broadcast use 15 of 32 zmm registers, 13 chains cover FMA latency ×
throughput on the two-pipe hosts, and 13 divides 52, so the token-stream
GEMMs run with no row tail. Accumulation stays one fused chain per
output element with `k` in order — the same order as the AVX2 and
portable kernels — so all three paths are **bit-identical**.

The compile target stays `x86-64-v3`: the
[M0 CPU probe](morpheus-rs-cpu-probe.md) saw v3 hosts in the fleet next
to the v4 majority, so AVX-512 exists only behind runtime detection, and
`selfcheck` now reports the choice as `runtime_gemm`.

## Gates (all containers, both batches)

- `cargo test --release` in the changed tree passes on real AVX-512
  silicon — the bitwise kernel tests run the avx512 tiles against the
  portable reference there, not just on the arm64 dev box.
- Replies over the full recorded `synthetic-long` wire stream are
  **byte-identical** to the HEAD binary on every host class: avx512
  dispatch, avx2 fallback, both.
- Local: 49 crate tests, the 15 s pytest suite, and a finished
  `--mode competition` match (joe-rs beat aegis, capture at turn 232).
- arm64 dev box (portable path, untouched): 25.2 ms p50 against a 25.0
  baseline — noise.

## End-to-end, HEAD vs avx512, same container, A/B/A/B

Modal masks `model name`; hosts are identified by family/model.

| host (family model) | avx512f | dispatch | head p50 ms | wip p50 ms | ratio |
| --- | --- | --- | ---: | ---: | ---: |
| Intel 6 143 (Sapphire Rapids, v4) | yes | avx512 | 21.5 | **17.8** | **1.21×** |
| Intel 6 85 (Skylake-SP, v4), 3 containers | yes | avx512 | 26.7–27.0 | 25.1–26.4 | 1.01–1.07× |
| AMD 175 1 (v3), 4 containers | no | avx2 | 18.5–24.5 | 18.4–23.9 | ~1.00× |

**Reading the split:** the lane math (32 f32 lanes/cycle vs 16, load-port
bound at ~93%) predicts well above 1.2×. Sapphire Rapids, which has no
512-bit license downclock to speak of, delivers 1.21×. Skylake-SP
sustains 512-bit FMA only at a reduced license frequency that slows the
whole core, and nets 1.05× on the clean containers — still a win, never
a loss, so dispatch does not special-case it. The v3 fallback is free.

**AMD 175 17 (Zen, v4), the probe's fleet majority:** not drawn in the
joe-rs batches, but the unclejoe batch below landed one. It measures
**1.12–1.16×** on the X16 workload — no downclock, between the SKX and
SPR numbers.

## The unclejoe port (same kernel, X16 network)

`bots/unclejoe` is a code fork of joe-rs whose `gemm.rs` had not
diverged, so the port is a verbatim file copy plus the `runtime_gemm`
selfcheck line. X16 changes only the depth (7 → 16); every GEMM shape —
EMBED 384, 8 heads, FF 1536, 52 tokens — is identical, so the MR=13
tile applies unchanged. Same gates, all green: bitwise kernel tests on
AMD Zen and Intel SKX silicon, byte-identical replies over the full
`synthetic-long` stream on every host class, a finished
`--mode competition` match (capture at turn 272), arm64 unchanged at
57.1 ms p50 (2.3× joe-rs, as the FLOP ratio predicts).

| host (family model) | dispatch | head p50 ms | wip p50 ms | ratio |
| --- | --- | ---: | ---: | ---: |
| AMD 175 17 (Zen, v4) | avx512 | 38.3–38.8 | **33.1–34.8** | **1.12–1.16×** |
| Intel 6 85 (Skylake-SP, v4) | avx512 | 59.3–60.3 | 55.7–61.0 | ~1.0–1.06× |
| AMD 175 1 (v3), 2 containers | avx2 | 38–42 | 38–42 | ~1.00× |

The X16 absolutes are the reason the port matters more than joe-rs's:
on the slow SKX generation the head binary's p99 reached 70 ms against
the 150 ms move budget, and the fleet-majority v4 hosts now run
~34 ms p50 instead of ~39. **The working tree now differs from the
submitted unclejoe v2 content hash** — shipping this requires a v3
resubmission; the on-disk record here is the code change, not a
submission.

## Context: why attention itself was not the lever

This round started as a "FlashAttention on CPU" question. It does not
transfer: N_TOKENS is 52, so a head's whole score matrix is 10.8 KB —
L1-resident — and the attention products are ~2% of the forward's ~1.33
GFLOP, while the packed-B kernel already runs compute-bound at roughly
60–70% of single-core FMA peak. The memory traffic that fused attention
eliminates does not exist at this scale; the vector width did.
