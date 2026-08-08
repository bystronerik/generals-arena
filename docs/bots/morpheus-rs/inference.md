# Morpheus-rs inference

How the Rust bot runs the network, and why it runs it the way the rewrite plan
said not to.

Milestone M3 of the [rewrite plan](rewrite-plan.md). The artifact contract is
§3; the parity surfaces are in [parity-harness.md](parity-harness.md).

## The decision, and the plan it overrode

Plan §3 chose **candle first**, ranked tract second, and put a hand-written
fixed-shape kernel last — "entered only if profiling shows the framework
leaving ≥30% on the table". Its reasoning was that the network is tiny
(~0.2 GFLOPs) and the shipped 11.9 ms root inference therefore implied
TorchScript was overhead-bound, so a good library would recover most of it.

Both halves of that reasoning turned out to be wrong, and the same paragraph
also told us how to find out: the decision rule is "keep the fastest option
whose MAE ≤ 1e-5". So all three were measured.

| engine | batch 1 p99 | vs TorchScript | batch 4 p99 | vs TorchScript |
| --- | ---: | ---: | ---: | ---: |
| TorchScript (the oracle) | 6.09 ms | 1.00× | 20.61 ms | 1.00× |
| candle 0.9 | 27.84 ms | **4.57×** | 127.77 ms | **6.20×** |
| tract-onnx 0.21 | 8.53 ms | **1.40×** | 48.26 ms | **2.34×** |
| morpheus-rs (shipped) | 3.63 ms | **0.60×** | 14.34 ms | **0.70×** |

M3 Pro, one thread, identical inputs. Full figures and method:
[`morpheus-rs-inference-bench.md`](../../research/measurements/morpheus-rs-inference-bench.md).

R1's tripwire is "best pure-Rust option ≥1.3× TorchScript p99 at batch 1 **and**
4". candle trips it at 4.6×/6.2× and tract at 1.40×/2.34×, so the fallback
ladder — candle → tract → bespoke → vendored `ort` — was walked to its third
rung, exactly as §9 prescribes. It stops there: the bespoke kernel wins
outright, and `ort`'s vendored `.so` never has to be argued about.

### Why candle loses, specifically

Not "frameworks are slow". One op:

| candle op, batch 1 | ms/call | in morpheus-rs |
| --- | ---: | ---: |
| depthwise 3×3, groups=128 | 1.14 | 0.02 |
| pointwise 64→128 | 0.12 | 0.10 |
| stem conv 3×3, 49→64 | 0.61 | 0.30 |

candle implements grouped convolution by splitting the input and looping, so a
depthwise layer with 128 groups becomes 128 separate convolutions. Twelve
blocks of that is 13.7 ms — half of candle's entire forward — for 6% of the
network's arithmetic. Its pointwise GEMMs, where the FLOPs actually are, are
perfectly respectable.

That is a property of MobileNet-shaped networks meeting a library optimised for
transformers, and it is not something a different call site fixes. The spike
that found it is kept at
[`tools/spikes/candle-ops`](../../../bots/morpheus-rs/tools/spikes) so the
claim can be re-derived rather than believed.

### The correction worth carrying forward

§3's estimate that TorchScript was "running at ~17 effective GFLOPS —
overhead-dominated" came from the shipped `offline_p99_ms` table, which M0
already found understates live cost. Measured directly, single-threaded
TorchScript does 212 MFLOP in 6.09 ms — **35 GFLOP/s**, which is not
overhead-dominated, it is a competent GEMM library on a fast core. The bespoke
engine reaches 60 GFLOP/s and wins by 1.68× at batch 1, at the top of the range
§3 guessed at but for a different reason than it gave.

So the plan's *conclusion* about which engine to ship was wrong and its
*decision rule* was right. The throughput thesis in §7 never depended on an
inference win anyway; this is a bonus on top of the transition kernel.

## What the engine is

`crates/core/network.rs` plus `gemm.rs`, about 1,080 lines including their
tests. No dependencies, no `unsafe`, no intrinsics.

- **Layout.** Channel-major planes of 448 floats — 441 board cells rounded up
  to a whole number of 16-wide GEMM column tiles. The seven pad columns start
  at zero and stay there; the two reductions that could see them, GroupNorm's
  statistics and the masked global pool, both loop over the 441 real cells.
- **Stem.** im2col into a 441×448 buffer, then one GEMM. The buffer is 790 KB
  and preallocated.
- **Pointwise layers.** `gemm_bias`, a 4×16 register-tiled kernel with `n`
  outermost so the 16-column strip of activations stays in L1 across every row
  block of weights.
- **Depthwise.** Per channel, per row, nine taps into a row-local accumulator.
- **Heads.** 1×1 convolutions are the same GEMM with a row tail; the scalar
  heads are a matvec over the 128-wide masked global feature vector.
- **Batch is a loop.** Four sequential forwards, not a widened tensor. It costs
  a little arithmetic intensity and buys constant memory, one set of shapes,
  and no allocation on the per-turn path (§6) — and four of ours still beat one
  batched TorchScript call.

### `mul_add` is the whole performance story

Rust compiles floating-point with contraction **off**: `acc += a * b` emits a
separate multiply and add, and LLVM is not allowed to fuse them because fusing
changes the result. Writing `a.mul_add(b, acc)` explicitly took the pointwise
layers from 11 to 38 GFLOP/s and the forward from 11.3 ms to 5.5 ms — from
losing to TorchScript by 1.8× to beating it.

Everything else measured smaller. GroupNorm's statistics went from 1.37 ms to
0.44 by splitting one serial `sum +=` chain into eight lanes.

| stage | ms (batch 1) | share |
| --- | ---: | ---: |
| pointwise | 2.30 | 66% |
| group_norm | 0.44 | 13% |
| stem GEMM | 0.30 | 9% |
| depthwise | 0.28 | 8% |
| elementwise | 0.15 | 4% |
| im2col | 0.02 | 1% |

### Do not trust a per-stage number from a different build

The depthwise row above says 0.28 ms. Earlier in M3 the same source measured
**1.10 ms**, and the change came from edits elsewhere in the crate. Three
candidate causes were tested and all three refuted: the softmax width change,
the profiler's stage ordering, and three formulations of the loop itself
(read-modify-write passes, clamped ranges, padded rows) which measured within
1% of each other at *both* speeds.

What is left is the build. `lto = "fat"` with `codegen-units = 1` lets any new
code path shift inlining and vectorisation across the whole crate, and M3 added
several. The practical rule: re-measure a stage, never quote one from memory,
and hold onto the whole-forward figure — which was stable to 0.01 ms across
every run.

## The artifact

`tools/convert_artifact.py` reads the *committed* TorchScript artifact of the
frozen oracle, verifies every digest its manifest claims, and writes
`artifact/model.safetensors` plus a manifest recording the source digest.
Training and its exports are untouched: a new checkpoint means re-running the
converter, not editing an exporter.

Two things about it are load-bearing:

- **It writes the safetensors bytes itself.** `safetensors.torch.save_file`
  serializes its `__metadata__` from a Rust `HashMap`, whose iteration order
  varies between processes, so two conversions of identical weights produce
  different files. `artifact/` is inside the bot's content hash, so that would
  mint a new bot identity for no change — and `--check` could never tell
  "the artifact drifted" from "the library reshuffled a dict". The writer is
  canonical; the result is round-tripped through the reference reader so the
  layout is not graded by its own mirror image.
- **The bot refuses to start on a mismatch.** `inference.rs` re-checks the
  manifest version, all three schema tags, the four architecture dimensions,
  the quantization format, the 17 army-bin edges, and the SHA-256 of the
  weights — the same guardrails as `inference.validate_manifest`, because a bot
  that happily loads last month's checkpoint fails silently in every other way.

One graph serves all three of the Python's TorchScript entry points, switched
by [`Heads`]: `Policy` for belief proposal and enemy priors, `PolicyWdl` for
root and leaf evaluation, `All` for recovery and diagnostics.

## Parity

Two new surfaces, both in the full-corpus run:

- **`net`** — one recorded tensor through all three entry points on each side,
  compared head by head against the three TorchScript modules. 1,399 cases,
  worst MAE 4.05e-6 against §5's 1e-5 budget, worst single element 1.05e-5.
- **`prior`** — `legal_normalized_policy` and `backup_value` on identical f32
  logits. Worst 6.56e-7 on a prior mass, 1.19e-7 on the backup value.

Unlike every earlier surface, `net` **cannot** be bit-exact: the two engines
sum the same products in different orders and the Rust kernels fuse their
multiply-adds. So §5's tolerance is doing real work here for the first time —
2.5× headroom, against the thousandfold slack M2 found under the tensor's
1e-6. The harness reports the worst |Δ| every run so that headroom stays a
number rather than folklore.

`prior` is the exception that proves M2's rule about widths. The first version
computed the softmax in f64 — more accurate — and disagreed with the oracle in
the eighth decimal, because `torch.softmax` runs at the tensor's dtype and
every decision morpheus has ever made was made on that f32 rounding. The port
mirrors the width and keeps the f64 *result*, exactly as the tensor builder
mirrors `float32`/`float64` site by site.

`net` also checks something no single entry point can: the three exits share a
trunk, so their common heads must be **identical** on each side. A drift there
is not float noise, it is an entry point computing something else.

## What M3 did not settle

- **The x86 measurement itself.** M3's exit gate is written against x86 and
  the arm64 win above does not settle it: x86 TorchScript dispatches to oneDNN
  kernels tuned harder than anything the NEON path gets. The run is
  `scripts/morpheus_rs_modal_inference_bench.py`. Even when it lands it is one
  Modal container, and M0 found separate runs on different fleet generations —
  a shape, not a qualification. M7 accepts deployment numbers.
- Why an unrelated edit moved the depthwise stage 4×. Refuting three
  hypotheses narrowed it to whole-crate LTO, but narrowing is not identifying,
  and the next person to optimise a stage here should know that.
- Nothing here is wired into a decision. `NetworkEvaluator`'s shaping blend
  needs `tactics.py` (M5) and its tensor needs `particle_summary` (M4); M3
  ships the engine and the two arithmetic helpers around it, and the bot still
  plays M1's first-legal-move.
