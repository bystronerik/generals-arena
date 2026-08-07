# Morpheus float32 deployment decision

> Verdict: **float32 TorchScript replaces static int8 as the deployment
> format.** The int8 artifact was both lossy and slower; float32 is bit-exact
> and the fastest measured runtime at 0.25M and at 2M parameters.

Date: 2026-08-07. Checkpoint: `ckpt-00100000-e5b415184274`
(run `scraped-classes15-2026-08-06-1`).

## Defect that triggered this

The shipped int8 artifact was calibrated by feeding `torch.randn` batches to
the PTQ observers (`export.py`), while real observation tensors are sparse and
mostly in `[0, 1]`. On the trained checkpoint the resulting float→int8 errors
were far past the soft limits (2.0 / 1.0 / 1.0):

| Output | int8 artifact MAE |
| --- | ---: |
| policy | 21.4 – 22.2 |
| wdl | 23.3 – 23.7 |
| pass_logit | 102 – 114 |

The artifact was exported with `--allow-parity-fail`
(`parity_assert_skipped: true` in the old manifest), so the gate never fired.
The original preflight limits were derived from a random-weights probe, where
randn calibration happens to match the activation distribution — the defect
only appears with trained weights, whose logits reach magnitude ~100.

A second mismatch: the artifact was quantized with **qnnpack on the arm64 Mac**
while the judge runs x86 Linux.

## Why float32 instead of fixing calibration

`torch.set_num_threads(1)` on the Modal judge-like x86 CPU host (the judge
grants one dedicated core), p50 of 3 reps, random weights:

**0.25M parameters (deployed size)** —
[`morpheus-export-preflight-baseline-1thread.json`](morpheus-export-preflight-baseline-1thread.json)

| Candidate | root (b=1) | leaf (b=4) | proposal (b=64) |
| --- | ---: | ---: | ---: |
| float32 TorchScript | **6.1 ms** | **18.5 ms** | **613 ms** |
| int8 qnnpack | 11.1 ms | 42.7 ms | 998 ms |
| int8 fbgemm / x86 | 50 ms | 200 ms | 3320 ms |

**1.93M parameters (2M scale-up candidate, trunk 192 / expansion 384)** —
[`morpheus-export-preflight-2m-1thread.json`](morpheus-export-preflight-2m-1thread.json)

| Candidate | root (b=1) | leaf (b=4) | proposal (b=64) |
| --- | ---: | ---: | ---: |
| float32 TorchScript | **23.2 ms** | **86.1 ms** | **2485 ms** |
| int8 qnnpack | 45.6 ms | 180.6 ms | 3880 ms |
| int8 fbgemm / x86 | ~114 ms | ~453 ms | ~8200 ms |

Int8 never wins at either size: all 37 GroupNorms stay float, so the quantized
graph pays dequant → GroupNorm → quant per block, and the convolutions are too
small for int8 arithmetic to amortize that. The 2 GB memory cap is nowhere near
binding (float32 artifacts are ~1.1 MB each).

Consequences for a 2M resize: float32 root at 23 ms fits the 125 ms deadline,
but an 86 ms leaf batch caps search at roughly one leaf batch per move. A 2M
net needs either a reduced simulation target, a smaller width, or an
architecture whose norm layer quantizes cleanly — int8-on-GroupNorm is not the
lever.

## What shipped

- `export.py` gained a float32 TorchScript export (`torch.jit.script`, no
  calibration step); it is now the default for `export_from_checkpoint` and
  `scripts/morpheus_export.py`. Int8 stays available via `--format int8`.
- The deployed artifact trio was re-exported float32 from the same checkpoint:
  **every float→export MAE is exactly 0.0**, no `--allow-parity-fail`.
- `inference.py` skips the quantized-engine requirement for float32 manifests.
- `deployment.json` / `DeploymentConfig` identity updated to
  `torch.jit.script+float32`.

Float32 also removes the deprecated quantized-tensor ops
(pytorch/pytorch#184982) from the deployed path — the `Quantizer.cpp`
deprecation warning no longer fires at export or match time.

## Caveats

- Probe latencies use random weights (latency is weight-independent) and 3
  reps; treat ±10% as noise.
- `deployment.json` p99 numbers still date from the int8 qualification run on
  the M3 host and should be re-measured (they are, if anything, pessimistic
  now).
- Earlier cadence measurements (`data/games/morpheus-cadence-*`) were played
  with randn-calibrated int8 checkpoint bots; conclusions drawn from them
  measured quantization noise as much as checkpoint strength.
