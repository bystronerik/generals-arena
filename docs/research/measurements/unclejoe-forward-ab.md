# unclejoe forward pass — four optimisations, measured against the split

Measured 2026-08-27, straight after
[the 25-step split](unclejoe-forward-steps.md). Five optimisation passes were
run over the split, one per group; four changes came out of them and one was
rejected on its own numbers. Raw record:
[unclejoe-forward-ab-modal.json](unclejoe-forward-ab-modal.json).

- Bot: `bots/unclejoe` (X16, depth 16, 29,551,834 params, argmax T = 0)
- Harness: [`scripts/unclejoe_modal_forward_ab.py`](../../../scripts/unclejoe_modal_forward_ab.py)
  — every arm compiled into one image, then run **interleaved in one
  container**, base first and last
- Input: 1,062 turns of the recorded `synthetic-long` wire stream
- Hosts: six single-CPU Modal containers over two runs, `avx2` and `avx512`
  dispatch both represented

## Result: +6.8 % to +9.5 % of the forward, bit-exact

| host | dispatch | base forward | all four | gain | base noise floor |
| --- | --- | ---: | ---: | ---: | ---: |
| AMD 175/17 | **avx512** | 33.50 ms | 31.03 ms | **+7.38 %** | 1.90 % |
| AMD 175/1 | avx2 | 37.84 ms | 35.26 ms | **+6.82 %** | 4.40 % |
| AMD 175/1 | avx2 | 44.33 ms | 40.10 ms | +9.54 % | 18.07 % |

The third row's gain is inside its own noise floor and is reported for
completeness, not as evidence. The first two clear theirs by 4× and 1.5×.

**Nothing the network computes changed.** The full-corpus parity gate against
the JAX oracle prints, for the patched tree and for `base`, the same line to
the character:

```
[tier2] 728 frames: max |dlogit| 2.575e-04 (per-frame rel 1.428e-05),
        max |dvalue| 5.335e-06, max |dbin| 1.945e-04 (per-frame rel 2.160e-05)
[tier3] 728/728 greedy actions equal (100.0000%)
```

Those are also the numbers already recorded beside the pins in
`tests/test_unclejoe_parity.py`. Replies over the whole wire stream are
byte-identical to base's on all six containers — including the AVX-512 one,
where the dispatch runs tiles the arm64 development box never executes.

## What each change did, per step

Per-step means over the interleaved runs. Each step's own base spread is the
floor it has to clear; the `forward` cell cannot resolve any of these except
the exp one, which is why they are read here and not there.

| step | share of forward | speedup | on |
| --- | ---: | ---: | --- |
| `softmax` | 4.2–7.1 % | **2.76–2.87×** | 6/6 hosts |
| `silu` | 4.2–6.9 % | **1.84–2.24×** | 6/6 hosts |
| `norm1` / `norm2` / `norm_out` | 0.8–1.0 % | **1.94–2.14×** | 6/6 hosts |
| `policy_head` | 0.2–0.3 % | **1.47–1.60×** | 3/3 hosts |
| `scores` | 1.7–2.6 % | **1.06–1.09×** | 3/3 hosts |

### 1. `exp_poly` was scalar on x86 and nobody knew (the whole win)

Rust's `f32 as i32` **saturates**; no x86 instruction does, so LLVM lowers it
as `fptosi.sat` and **scalarizes it**, refusing to vectorize any loop that
contains one. `exp_poly`'s final `2^kf` scaling had one. aarch64 hides this
completely — `fcvtzs` saturates in hardware — so the development box showed
nothing wrong for as long as this code has existed.

Replacing the cast with a magic-number add (`kf + 1.5·2²³`, whose bit pattern
is `0x4B400000 + kf`, and `0x4B400000 << 23 ≡ 0 mod 2³²`, so the bias falls
out of the shift for free) removes it. A `vcmpunordps`/`vblendvps` select
restores NaN propagation, which the saturating form gave for nothing. The
whole-crate x86 instruction counts move `vcvttss2si` **58 → 0** and packed
`vfmadd213ps` **72 → 150**. An exhaustive sweep of all 2³² inputs finds **0
bitwise mismatches**, NaN payloads included.

`softmax` also had a second, independent problem: its `sum` accumulated
inside the same loop as the `exp`, and a non-reassociable float reduction
stops LLVM vectorizing **the whole loop**. Splitting exp from sum — same
values, same order, bit-exact — is why `softmax` gains more than `silu`.

The x86 arm is `#[cfg(target_arch = "x86_64")]`-scoped: on aarch64 the cast
form measured 9 % *faster*, so each target keeps the spelling that suits it.

### 2. The layernorm's eight lanes did the opposite of what the comment said

`LayerNorm::forward_into` used eight accumulator lanes "so the reductions
vectorize instead of serializing on add latency". On x86, LLVM packed those
eight lanes into **one ymm register**, turning eight independent scalar
chains into a *single* serial `vaddps` chain, and spilled the row between the
two passes — 35 ymm spill stores and a 1,192-byte stack frame per call.
Software-pipelining the three passes across rows removes the spill entirely
and lets the two chains and the divide issue together. Same lane partition,
same true divide, same unfused affine: bit-exact.

### 3. The 4-column tail cost a third of the `scores` GEMM

`NR = 16`, so the 52-wide score panel is three full strips **plus a 4-column
tail**, and the 48-wide context panel has none. That is the entire `scores`
vs `context` asymmetry the split turned up. The tail's inner loop had a
runtime trip count, so it never unrolled: one accumulator, a 48-deep
dependent FMA chain, nothing else in flight. The control that proves it —
`(52,52,52)` against `(52,52,48)` — is **+8 % FLOPs for +52 % time**.

Giving the tail a compile-time width and blocking `MR` rows fixes it.
`policy_head` (90 wide → a 10-column tail) gains most, 1.47–1.60×.

### 4. `#[inline(never)]` on the portable kernel — arm64 only

Under `lto = "fat"` + `codegen-units = 1`, LLVM inlined `gemm_bias_portable`
into `forward_staged` at eleven call sites and spilled the tile
accumulators. Keeping it out of line is **1.64× on the arm64 development
box** (interleaved A/B: 60.6–64.2 → 37.4–38.7 ms p50).

It is worth **exactly nothing** on the competition host, and that is not an
estimate: `gemm_bias_portable` does not appear anywhere in the
`x86-64-v3` build — the only occurrences of "portable" in the emitted
assembly are unrelated rustc paths. It is carried for the development loop
and for the honesty of every future local measurement, not for the judge.

## Rejected: next-strip software prefetch in the GEMM

Written to close a 9–13 % intra-host gap between the streaming GEMMs and the
cache-resident `context`. Measured, on three AVX2 containers:

| step | speedup |
| --- | ---: |
| `ff1` | 1.066 / 1.069 / 1.167× |
| `ff2` | **0.919 / 0.923 / 0.938×** |
| forward | −0.04 % / −0.04 % / +5.54 % (last inside an 18 % floor) |

`ff1` gains and `ff2` loses about as much: `ff2`'s strips are 96 KB, so
prefetching the next one evicts the current. Rejected against the criterion
its author set **before** seeing the data — adopt only if no arm regresses.

## What this does not establish

- **The two GEMM groups are 85 % of the forward and contributed nothing.**
  The AVX2 kernel's hot block is 16 `vfmadd231ps` + 8 `vbroadcastss` +
  4 `vmovups` per 2 k-steps with zero spills; `MR = 4` is forced by 52's
  factorization, not tuned. The remaining lead is the AVX-512 tile's nine
  address-arithmetic ops per k-step, which nobody has tried yet.
- **`head_pack`'s panel-wide K transpose did not show.** It measured 1.59×
  in isolation and lands in the noise in-app on all three hosts; it is
  carried because it is simpler than the 52 `fill_col` calls it replaces,
  not because it was measured to pay.
- **`context` regressed slightly** on one host (0.958×) against `scores`'s
  larger gain. Net positive, but it is not free.
- **The absolute milliseconds remain a proxy.** Modal is not the competition
  host and the fleet spread exceeds every gain here; only the within-container
  ratios are evidence.

## Reproducing

```bash
python scripts/unclejoe_modal_forward_ab.py --stage <tree> --as <arm-name>
modal run scripts/unclejoe_modal_forward_ab.py --containers 3 --rounds 3
```

Correctness is a gate before timing: every arm replays the wire log in play
mode and its replies must equal base's byte for byte, which is the only place
an AVX2 or AVX-512 tile error can surface at all.
