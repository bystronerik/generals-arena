# Joe weights vs the 50 MiB zip cap — codec ladder (2026-08-24)

Can XL24 (68.46M params) weights fit a submission zip? The cap is 50 MiB
(`arena.bundle.MAX_ZIP_BYTES`; the judge's stricter possible reading is
47.68 MiB — [packaging](../../bots/joe-rs/packaging.md) §2). Source is
small since the dependency-free rebuild, so weights get ~47–49 MiB of it.

All numbers measured on the **real trained M7F4 artifact** (13.58M params,
f16-rounded, `bots/joe-rs/artifact/model.safetensors`), zlib level 9, then
scaled per-param to XL24. Quality proxy: argmax agreement vs the shipped
weights over 400 random-obs moves through the full jax per-move path —
**a weak proxy** (random obs have low-margin logits; the f16 gate's real
number was 99.93% on recorded frames), useful for ordering only.

## Codec ladder

| codec | B/param | M7F4 MiB | XL24 est MiB | argmax agree | fits 50? |
| --- | ---: | ---: | ---: | ---: | --- |
| f16-in-f32 + deflate (today) | 2.34 | 30.3 | **152.9** | 100% | no — 3× over |
| + byte shuffle (lossless) | 1.85 | 23.9 | 120.6 | 100% | no |
| true f16 + 2-byte shuffle | 1.73 | 22.5 | 113.2 | 100% | no |
| int8 per-row absmax | 0.91 | 11.8 | 59.4 | 92.5% | no (XL20: 46.8 ✓) |
| int6 per-row absmax | 0.71 | 9.2 | 46.5 | 73.2% | barely |
| int5 per-row absmax | 0.60 | 7.8 | 39.3 | 51.0% | yes |
| int4 block-32 | 0.53 | 6.9 | 34.5 | 36.8% | yes |

Deflate already sits at the order-0 symbol entropy (int8 codes: 7.05 bits
measured, deflate spends 7.3), so a custom rANS coder buys only ~5–10%
— packing cannot close the 3× gap, only lower precision can. Small
tensors (norms, biases, temporal encoders, heads, <3% of params) stay f32
in every quantized row; quantization applies to the 2-D matrices only.

## Readings

1. **The zip cap, not latency, is the binding size constraint under
   today's codec.** Today's 2.34 B/param supports only ~21M params — even
   the plain L tier (23.4M → 52.2 MiB) misses the cap, while the latency
   ladders cleared XL20/XL24. The lossless byte-shuffle alone (zero risk,
   loader reorders bytes at read) unlocks L at 41.3 MiB.
2. **XL20 + int8 is the coherent sweet spot: 46.8 MiB.** XL20 was already
   the latency recommendation ([joe-rs ladder](joe-rs-size-ceiling.md));
   int8 is the mildest quantization (92.5% agreement on the harsh random
   proxy) and lands inside both cap readings.
3. **XL24 requires int6-class storage (46.5 MiB) and naive absmax int6 is
   not good enough** (73% agreement). The known recovery routes, in
   increasing cost: calibrated quantization (GPTQ-style error
   compensation, using recorded parity-corpus frames as calibration
   data), mixed precision (int8 attention + int6 FF ≈ 0.79 B/param ≈
   51.7 MiB — still over, so mixing alone does not save XL24), or a short
   QAT finetune on vast with fake-quant forward.
4. Quantization here is a **storage codec only**: the loader dequantizes
   to f32 at intake, runtime stays f32 — the latency ladders and the RAM
   numbers are unaffected. The cost is a `joe-net-v2` artifact format:
   pure-Rust dequant in the loader, new manifest tag, regenerated
   self-goldens/parity corpus, and `mutation_check` updates.

## Validation bar

The f16 precedent is the template: frame agreement on recorded games plus
a rated-round exclusion (3,544 games for f16). Anything below f16 must
pass the same two gates before it ships; the random-obs agreement above
is a screening number, not evidence of strength.

## Discrepancy noted

The remembered "M7F4 zip 19.3 MiB" figure is inconsistent with this
direct measurement (30.3 MiB deflated at level 9 on the shipped, verified
f16-rounded artifact). The 2026-08-24 measurement supersedes it.
