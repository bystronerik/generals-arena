# X16 G1 — pack entropy of the real EMA checkpoint (2026-08-25)

Gate G1 of the [weights pack plan](../../bots/joe-rs/weights-pack-plan.md):
measure the real bytes-per-parameter of run `joe-X16-gcp-20260825` before
building the pack tooling. Checkpoint: `..._ema_1750.eqx` (step 1750,
sha `f3606f63…`, fetched from R2), deserialized through the deployment
template (depth 16, ff ×4, 276 leaves, 29,551,834 params), f16-rounded
first — the single-quantization-point effect. zlib level 9 throughout.

## Result: the projection holds — 1.72 B/param

| codec | MiB | B/param |
| --- | ---: | ---: |
| f16-in-f32 deflate (today) | 65.34 | 2.32 |
| f32 + 4-byte plane split | 51.98 | 1.84 |
| true f16 deflate | 52.01 | 1.85 |
| **f16 + 2-byte plane split (`f16p`)** | **48.55** | **1.72** |
| int8/row, all big 2-D | 26.17 | 0.93 |
| int8 FF matrices + `f16p` rest | 34.09 | 1.21 |

Deflated joe-rs source is 0.05 MiB (19 files, dependency-free crate), so
the cap verdicts are:

- **`arena.bundle` 50·2²⁰: the lossless `f16p` zip FITS, 1.40 MiB spare.**
- **Strict 5·10⁷ (47.68 MiB): MISSES by 0.92 MiB.** The int8-FF fallback
  (34.1 MiB) covers this reading with 13 MiB spare — lossy, f16-precedent
  gates required.

## Risk R1 resolved: no young-block optimism

Per-block `f16p` rates are uniform — 1.71 to 1.75 B/param across all 16
blocks, the nine freshly-spliced blocks indistinguishable from the seven
mature M7F4 blocks at step 1750. The mantissa entropy is already
saturated this early, so the number is stable against further training
and an end-of-run re-measure is a formality, not a gate.

## What remains open

Only D2, the cap reading. The lossless path lives in the 0.92-to-1.40 MiB
window between the two readings; no codec refinement inside this plan
changes that (rANS ~5–10% ≈ 2.4–4.8 MiB would land `f16p` at ~44–46 MiB —
the one lossless lever left if strict must hold without going lossy).

## Reproducing

Fetch the EMA via `training/joe/store.py` (`resolve_latest`,
`download_file`), then the measurement block in this doc's session; codec
definitions match [joe-weights-zip-budget.md](joe-weights-zip-budget.md).
