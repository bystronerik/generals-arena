# unclejoe weight export — X16 from R2

unclejoe's artifact is its **own export** from the R2 run
`joe-X16-gcp-20260825` — not a copy of joe's or joe-rs's artifact, so no
fan-out tool tracks it and no sync step exists. The whole chain lives in
`bots/unclejoe/`:

```bash
.venv/bin/python scripts/joe_export_bot.py \
    --run-name joe-X16-gcp-20260825 --out bots/unclejoe/artifact
.venv/bin/python bots/unclejoe/tools/quantize_artifact.py   # f16 first, always
.venv/bin/python bots/unclejoe/tools/convert_artifact.py
.venv/bin/python bots/unclejoe/tools/pack_artifact.py bots/unclejoe/artifact
export PATH="$HOME/.cargo/bin:$PATH"
cargo build --release --manifest-path bots/unclejoe/Cargo.toml
# the parity corpus is keyed to the network — rebuild it, do not reuse.
# archive first: capture_fixtures.py writes in place and skips existing .npz.
mv data/joe/unclejoe-parity/games data/joe/unclejoe-parity/games.step<OLD>
.venv/bin/python bots/unclejoe/tools/capture_fixtures.py --play
.venv/bin/python bots/unclejoe/tools/make_synthetic_long.py
.venv/bin/python bots/unclejoe/tools/capture_fixtures.py --capture
.venv/bin/python bots/unclejoe/tools/capture_fixtures.py --selection-golden
.venv/bin/python bots/unclejoe/tools/make_smoke_fixture.py
.venv/bin/pytest bots/unclejoe/tests/ -m joe
.venv/bin/python bots/unclejoe/tools/mutation_check.py
.venv/bin/pytest tests/test_joe_artifact_manifest.py
```

`joe_export_bot.py` resolves the **latest complete checkpoint set** via
`training/joe/store.py` (`resolve_latest`) — R2 credentials from the
gitignored `.env` only — verifies the download, sanity-loads it into the
deployment template (arch fields, including depth 16, flow from the run's
config into the manifest), and writes `ema.eqx` + `manifest.json`. The
script is shared with joe and carries no tier pins; the pinned shape sites
are unclejoe's own copies (see below).

## Provenance of the shipped artifact (export of 2026-08-25)

The run is young and still training, so "latest" is a moving target — this
table is what makes the pick reproducible. The manifest's `r2_key` + sha
recover the exact bytes.

| field | value |
| --- | --- |
| run | `joe-X16-gcp-20260825` |
| resolved step | **2500** (eval wr 95.7% vs the frozen M7F4 reference) |
| r2_key | `joe/joe-X16-gcp-20260825/checkpoints/joe-X16-gcp-20260825_ema_2500.eqx` |
| f32 `.eqx` sha256 (`pre_quantization_weights_sha256`) | `6a9265433474b8a46f375967913acca4e7b5ffe8a3f34b4a40e5c647b1079faa` |
| f16-rounded `.eqx` sha256 (`weights_sha256`) | `b354c17cbb62e2d4e79f267c2686c737b39d183b434c92414fc384191a81cca0` |
| `model.safetensors` sha256 | `e12c235e2139dd2f84de076e59ffcc897e51f136847dac6cce431f378cf94d7b` |
| `model.packed` sha256 (joe-net-v2, 117 rans0 + 435 raw parts) | `faa45fd269fc0f3db0bd088c3188a6ea9dbadb5a32908028bfd1b243f07632a6` |

## The pinned shape sites (tier X16)

The refusal pins are deliberate edits, never derived from the manifest — a
mis-shaped artifact must refuse loudly rather than mis-load. unclejoe pins
depth 16, **276 leaves, 29,551,834 params** (16 leaves and 1,774,464 params
per ff x4 block, 20 non-block leaves) in:

1. `bots/unclejoe/src/nn/net.rs` — `DEPTH = 16` (EMBED 384 and FF_DIM 1536
   unchanged; X16 is M7F4 grown in depth only). The loader's manifest
   cross-check then guards `depth`/`ff_factor` at load.
2. `bots/unclejoe/tools/quantize_artifact.py` — the two counts.
3. `bots/unclejoe/tools/convert_artifact.py` — `DEPTH` plus the two counts.

joe-rs's copies stay at 7 / 132 / 13,581,658; the two crates' pins never
track each other.

## Zip size and the D2 cap ambiguity

The submission zip (packed container STORED, safetensors and `.eqx`
excluded, sources minified) built at **50,189,332 B = 47.86 MiB**:

- `arena.bundle` cap 50·2²⁰ (52,428,800 B): **fits, 2.14 MiB spare**.
- strict 5·10⁷ reading (50,000,000 B): **misses by 0.18 MiB** — the packed
  container alone is 50,152,349 B.

That gap was decision **D2** in
[weights-pack-plan.md](../joe-rs/weights-pack-plan.md). **Resolved
empirically 2026-08-25**: Erik submitted this exact zip and the judge
accepted it. An accepted 50,189,332 B zip is over the strict 5·10⁷ reading,
so the enforced cap is not the strict one — the `arena.bundle` 50·2²⁰
reading (or at least this zip's size) is the real limit. The int8-FF
fallback is dead; the lossless path is the path. `build.sh` reconstructs the safetensors
at intake through `unclejoe unpack-artifact`, refusing on any digest
mismatch.

## Latency (measured over real competition games)

`unclejoe bench` replays of four recorded corpus games through the full
per-move path, release build, dev arm64 (M-series, unpinned), **re-measured
2026-08-27** after the forward-pass work in
[unclejoe-forward-ab](../../research/measurements/unclejoe-forward-ab.md):

| game | turns | p50 | p90 | p99 | max | startup | was (2026-08-25) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| synthetic-long | 1496 | 36.66 | 39.42 | 46.39 | 78.50 | 77.8 ms | 60.20 |
| joe-seed10 | 748 | 36.60 | 38.14 | 40.79 | 64.92 | 120.9 ms | 60.68 |
| castle_rush-seed3 | 551 | 37.07 | 38.36 | 42.81 | 82.92 | 103.0 ms | 60.32 |
| aegis-seed0 | 272 | 36.33 | 37.92 | 40.72 | 42.82 | 91.3 ms | 60.22 |

Worst p99 46.4 ms against the 140 ms working deadline: **~94 ms margin** on
this host.

**The 2026-08-25 column was measuring the compiler, not the network.** Under
`lto = "fat"` + `codegen-units = 1`, LLVM inlined the portable GEMM kernel
into `forward_staged` at eleven call sites and spilled the tile
accumulators; `#[inline(never)]` on that kernel is worth 1.64x here and
**nothing at all** on the competition host, where the portable kernel is not
even present in the `x86-64-v3` build. Treat every arm64 latency figure
recorded before that date as inflated by roughly that factor, this bot's and
joe-rs's alike — the two crates share the kernel. The deadline knobs joe-rs parses are dead code here too — the
turn budget is the full 140 ms. Do not quote joe's Python-side numbers for
this bot, and do not build a musl static binary (measured 4x allocator
regression on candle-era joe-rs; portable builds target old glibc). The
competition fleet spans 29–77 ms p50 for *M7F4* code
([modal fleet variance](../joe-rs/latency.md)); a slow-generation x86 host
roughly doubles these numbers, which still fits the budget but with a
thinner margin — re-measure on Modal x86 before relying on headroom there.

The 2026-08-25 submission of this exact bot passed the judge's latency
limits, so the X16 tier is confirmed viable on real judge hardware — a
pass/fail signal only; the judge reports no percentiles.
