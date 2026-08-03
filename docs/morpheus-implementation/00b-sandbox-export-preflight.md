# Part 00b: Sandbox export preflight

## Deliverable

Prove that one pinned competition-sandbox runtime can load and execute a static
8-bit version of the specified convolution, group-normalization, policy, and
WDL path before the full network is implemented.

**Touches**

- `scripts/`: add the isolated export probe.
- `training/morpheus/`: add random-weight fixtures and parity tests.
- `bots/`, `arena/`, and `data/bot_versions/`: none.
- `competition-module/`: read its canonical dependency pins only.

## Prerequisites

- [Part -1: Process substrate](-1-process-substrate.md)

## Source specifications

- [Network](../bots/morpheus/network.md)
- [Sandbox dependency pins](../../requirements-sandbox.txt)
- [Canonical judge pins](../../competition-module/competition/requirements.txt)

## Defaults and replacement measurement

Start from the specified 64-channel architecture and static 8-bit deployment
requirement. Use random weights. Do not spend training compute.

The preflight replaces assumptions about runtime support with measured export,
load, parity, memory, and latency results under the exact pinned packages.

## Implementation boundary

Test only sandbox-available runtimes. Do not add ONNX Runtime or another
unavailable package.

Exercise the batch shapes required for root, leaf, and enemy-proposal
inference. Record quantization format, unsupported operators, fallback
operators, serialized size, peak RSS, and float-to-export error.

This part validates a path. Part 04 still owns the complete model and manifest.

## Isolated test

```bash
python -m pytest training/morpheus/tests/test_sandbox_export.py \
  -m morpheus -q
```

```bash
python scripts/morpheus_export_preflight.py \
  --requirements requirements-sandbox.txt \
  --output docs/research/measurements/morpheus-export-preflight.json
```

## Specification gaps

The specs do not select the runtime, quantization scheme, calibration method,
or acceptable parity error. This preflight must report candidates rather than
hide an unsupported operator behind floating-point fallback.

## Exit criterion

Answer `yes` if at least one pinned runtime exports, reloads, and executes the
required random-weight graph with bounded memory and recorded parity error.

Answer `no` if every path needs an unavailable dependency or cannot represent
the specified graph. A `no` blocks Part 04 until the network or deployment
design changes.
