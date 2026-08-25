# rANS plan — a pure-Rust entropy decoder for the packed artifact

Add an `rans0` codec to the [weights pack plan](weights-pack-plan.md)'s
`joe-net-v2` format: streams entropy-coded with the range variant of
Asymmetric Numeral Systems (Duda, arXiv:1311.2540), encoded by the Python
pack tool, decoded by a hand-written, dependency-free Rust decoder inside
`joe-rs unpack-artifact`.

## 0. The measured payoff — read this before implementing

Measured 2026-08-25 on the real X16 step-1750 EMA (entropy bounds; a
static-table rANS lands within ~0.3% of these):

| scheme | MiB | vs deflate `f16p` (48.55) |
| --- | ---: | ---: |
| rANS order-0, global plane tables | 49.87 | **worse** |
| **rANS order-0, per-tensor plane tables** | **47.85** | **−0.70 MiB** |
| rANS order-1, global tables | 48.93 | worse |
| floor: 16-bit symbols, per-tensor tables, payload only | 47.33 | −1.22 MiB |

Two conclusions the plan must respect. First, **the earlier "5–10% over
deflate" estimate is refuted**: deflate's block-adaptive Huffman is
already near the static floor on this data, and only the per-tensor-table
configuration beats it at all — table granularity is the entire win.
Second, **rANS does not rescue the strict 47.68 MiB cap reading**: 47.85
MiB is still ~0.2 MiB over that reading's ~47.6 MiB artifact budget, and
even the perfect-coder floor leaves no real margin. What `rans0` buys is
+0.7 MiB of headroom under the `arena.bundle` 50·2²⁰ cap (1.40 →
2.10 MiB) — and, under a strict-cap D2, a ~3% shave on the int8-FF
fallback's code streams. Decision D2 in the pack plan is unchanged by
this work.

## 1. Variant and references

The `rans_byte` variant from Fabian Giesen's public-domain reference
(github.com/rygorous/ryg_rans, `rans_byte.h`), the implementation zstd's
generation of coders descends from:

- **State**: `u32`, normalization interval lower bound `L = 1 << 23`.
- **Encode** (per symbol, frequency `f`, cumulative start `c`, precision
  `k` = scale bits): `x = ((x / f) << k) + (x % f) + c`, after emitting
  low bytes while `x >= ((L >> k) << 8) * f`.
- **Decode**: slot `s = x & ((1 << k) - 1)`; find the symbol whose
  cumulative range contains `s`; `x = f * (x >> k) + s - c`; read bytes
  (`x = (x << 8) | byte`) while `x < L`.
- **LIFO**: the encoder processes symbols in reverse order and writes its
  buffer backwards; the decoder reads forward. This is the classic
  footgun — the Python encoder must `reversed()` its symbol stream.
- The flushed 4-byte final state heads the stream; the decoder seeds
  from it.

Secondary reference: the range-ANS tutorial arXiv:2001.09186. The
reciprocal-multiplication and SIMD paths in ryg_rans are explicitly out
of scope — decode runs once at intake next to a full cargo build.

## 2. Format: the `rans0` codec in `joe-net-v2`

Per part (one part = one tensor plane, after the f16 2-byte plane split):

    scale_bits: u8         (fixed 14 in v1; validated 8..=15)
    freqs: [u16; 256]      (raw frequencies, must sum to 1 << scale_bits)
    payload_len: u32
    payload: [u8]          (4-byte LE initial state, then the byte stream)

Tables travel in the container, so **frequency normalization exists only
in the Python encoder** — there is no cross-language normalization to
keep in sync. The decoder validates the sum and builds the cumulative
table plus a `1 << scale_bits` slot→symbol LUT (16 KiB, transient per
part). Every symbol present in the plane gets frequency ≥ 1; the encoder
asserts this before encoding. Total table overhead at per-tensor
granularity: 276 KiB (552 tables), already counted in the 47.85 MiB.

`model.packed` becomes high-entropy bytes, so the packager stores that
zip member uncompressed (STORED) — deflating rANS output buys nothing.

## 3. Implementation

- **Python encoder** in the pack tool (`pack_artifact.py`): normalize
  frequencies to `1 << 14` (largest-remainder method, floor 1 for
  present symbols), encode reversed with numpy-free integer arithmetic,
  then **self-decode and compare bytes** before writing — the tool
  refuses to emit a stream it cannot itself decode.
- **Rust decoder**: `src/nn/pack.rs`, roughly 120 lines plus validation.
  No new dependencies, no unsafe. `u64` intermediates where overflow
  analysis is not obvious at a glance — this code runs once per intake,
  clarity beats cycles.
- **Loudness over leniency** (the loader's house rule): truncated
  payload, frequency sum ≠ `1 << scale_bits`, a decoded slot mapping to
  a zero-frequency symbol, or trailing unconsumed bytes are all hard
  refusals naming the part. The final gate is unchanged from the pack
  plan: the reconstructed `model.safetensors` must match the manifest's
  `safetensors_sha256` or intake fails.

## 4. Phases and gates

- **P1 — Python encoder/decoder pair.** Encode the real X16 EMA parts,
  self-decode, and report the exact zip size. Gate: the `rans0` artifact
  must beat the deflated `f16p` artifact (expected ~47.9 vs 48.55 MiB);
  if it does not, stop and keep `f16p` — the format loses nothing, since
  codecs are per-part.
- **P2 — Rust decoder + fixtures.** Golden fixtures under
  `bots/joe-rs/tests/` (outside the content hash): a tiny
  hand-verifiable stream, several Python-encoded random streams
  (including single-symbol and all-256-symbol tables), and the
  adversarial refusal cases from §3. Byte-exact equality in a `cargo
  test`.
- **P3 — integration.** The `rans0` arm in `unpack-artifact`, the
  STORED zip member in the packager, then the pack plan's G3 unchanged:
  clean-dir intake rehearsal, sha match, parity replay identical, zip
  under the D2 cap, competition matchup gate.

## 5. Risks

- **R1 — encoder/decoder drift.** The formulas must agree forever.
  Mitigated by tables-in-container (no shared normalization), the
  encoder's self-decode, the fixture suite, and the terminal sha gate —
  a drifted decoder cannot produce a loadable artifact silently.
- **R2 — the payoff is 0.7 MiB.** Known and accepted up front (§0);
  this is a margin-and-craft feature, not a cap rescue. If P1 measures
  worse than deflate, the stop is explicit.
- **R3 — LIFO confusion producing reversed tensors.** Caught by the
  first fixture and by the sha gate; called out in §1 so review looks
  for it.
