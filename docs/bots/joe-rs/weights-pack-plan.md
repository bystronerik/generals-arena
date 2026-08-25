# Weights pack plan — X16 EMA checkpoints into the submission zip

Goal: ship EMA checkpoints of run `joe-X16-gcp-20260825` (29,551,834
params, 276 leaves) inside the submission zip cap. Under the current
f16-in-f32 storage the X16 artifact deflates to ~65.9 MiB — over every
reading of the cap. Codec numbers:
[joe-weights-zip-budget.md](../../research/measurements/joe-weights-zip-budget.md).

**Scope**: export and packaging only. R2 checkpoints stay f32 `.eqx`,
untouched (see non-goals). Nothing here changes training or the run.

## 1. The budget and decision D2 (pick the cap reading first)

| | bytes | X16 artifact budget¹ |
| --- | ---: | ---: |
| `arena.bundle.MAX_ZIP_BYTES` | 50·2²⁰ | ~49.4 MiB |
| strict "50 MB" reading | 5·10⁷ (47.68 MiB) | ~47.1 MiB |

¹ cap minus deflated source (~0.5 MiB, dependency-free crate; re-measure
at packaging).

Projected X16 artifact sizes (M7F4-measured B/param × 29.55M — projections
until G1):

- **f16 + 2-byte plane split (lossless): 1.73 B/param → 48.76 MiB.**
  Passes the 50·2²⁰ reading with ~0.6 MiB to spare; **fails strict**.
- int8/row FF matrices (18.9M params) + f16-split rest: ~34.0 MiB.
  Passes both readings; lossy — inherits the f16-precedent gates.
- rANS instead of deflate: ~5–10% over the shuffle number (~47.5 MiB
  total) — knife-edge on strict, real loader complexity. Not planned.

**D2**: decide which cap the bundle must satisfy before Phase 1. If
50·2²⁰ (what our own packager enforces): the lossless path is the plan.
If strict: lossless alone does not fit and the int8-FF fallback becomes
the main path, gates and all. A probe submission of a >47.68 MiB,
<50 MiB zip would settle the ambiguity empirically.

## 2. Format: `joe-net-v2` packed artifact

The zip carries `model.packed` + the manifest; **`model.safetensors`
leaves the zip** and is reconstructed at intake. Manifest additions:

- `pack_format: "joe-net-v2"`, plus per-part entries: tensor name, shape,
  codec (`f16p` = f16 values, 2-byte plane split; `f32` = raw), byte
  offsets into `model.packed`.
- `packed_sha256`; the existing `safetensors_sha256` stays and becomes
  the reconstruction target the unpacker must reproduce.

`model.packed` holds the transformed raw bytes (~59.1 MB for X16); the
zip's own deflate is the entropy coder — no compression code ships.
Every tensor is `f16p` (all values are f16-representable after
`quantize_artifact.py`; widening f16→f32 is bit-exact, so the format is
lossless relative to today's artifact — including norms, biases, and
heads). `f32` exists as an escape hatch only.

## 3. Reconstruction at intake

`build.sh` gains one step after `cargo build`:

    target/release/joe-rs unpack-artifact

A new subcommand (pure Rust, no deps, ~150 lines in `io/` or a new
`nn/pack.rs`): read manifest parts → inverse plane split → widen
f16→f32 → write `model.safetensors` → **verify `safetensors_sha256`,
refuse to proceed on mismatch**. `net.rs` and the loader are untouched:
they read a byte-identical safetensors, so goldens, the parity corpus,
and `mutation_check` pins are unaffected by construction. Unpacked-size
cap is fine: packed 59 MB + reconstructed 118 MB against 512 MiB.

## 4. Pack tool (dev side)

`bots/joe-rs/tools/pack_artifact.py` (dev-time, outside the content
hash), run after `convert_artifact.py`:

1. Refuse unless joe's manifest carries `quantized: "f16"` AND every
   value f16-round-trips bit-exactly (belt and suspenders — an f16
   container must never be the thing that rounds).
2. Write `model.packed` + manifest entries.
3. Decode its own output in Python and compare bytes against
   `model.safetensors`; refuse to keep the output on any mismatch.

Packager change: `package_submission.py` includes `model.packed`,
excludes `model.safetensors`, and asserts the built zip against the D2
cap, reporting the packaging.md §2 table.

## 5. Phases and gates

- **Phase 0 / G1 — measure before building.** Pull one X16 EMA from R2
  (`joe/joe-X16-gcp-20260825/checkpoints/*_ema_*.eqx`, any step past the
  early dip), f16-round it, and run the codec measurement from the
  zip-budget doc on the real tensors. The 1.73 B/param is an M7F4
  extrapolation; X16's nine freshly-grown blocks may compress
  differently in either direction. G1 + D2 select the path:
  lossless-only, or int8-FF fallback.
- **Phase 1 / G2 — format + tools.** Pack tool, Rust unpacker, and a
  small synthetic fixture under `bots/joe-rs/tests/` (outside the
  content hash): pack in Python → unpack in Rust → bytes equal.
- **Phase 2 / G3+G4 — real chain.** On the chosen X16 EMA: quantize →
  convert → pack → clean-dir intake-style build → unpack → sha match →
  parity replay bit-identical to a direct-safetensors run → built zip
  under the D2 cap. Then the standard competition matchup gate.
- **Phase 3 — only if the lossy fallback is selected**: the f16
  precedent applies in full — frame agreement on recorded games plus a
  rated-round exclusion before anything ships. Note the X16 tier itself
  also needs the four pinned shape sites edited (DEPTH 16, 276 leaves,
  29,551,834 params, docs) — a prerequisite of any X16 export,
  independent of this plan.

## 6. Risks

- **R1**: X16 B/param differs from the M7F4 extrapolation. G1 resolves
  this before any code is written.
- **R2**: the cap ambiguity (D2). The lossless path lives entirely
  inside the ambiguous 47.68–50 MiB window; deciding late invalidates
  Phase 1 work.
- **R3**: a checkpoint that skipped `quantize_artifact.py` would be
  silently rounded by an f16 container. The pack tool's round-trip
  check (§4.1) turns this into a refusal.
- **R4**: tools that read `model.safetensors` dev-side (parity tests,
  `mutation_check`) — unaffected: packing does not delete the file;
  only the zip omits it. Audit anyway in Phase 2.

## 7. Non-goals

- **Compressing the R2 checkpoints themselves.** They are f32 originals;
  f32 mantissas are pure entropy (deflate ratio 0.928 measured), so
  lossless storage compression saves ~7% for real pipeline complexity,
  and rounding them would move the single quantization point. Not worth
  it.
- Integer quantization of the shipped X16 unless G1 + D2 force the
  fallback path.
- rANS or any entropy coder beyond the zip's deflate (~5–10% for real
  loader complexity; revisit only if G1 lands within 1 MiB of the cap).
