# Part 04: Network and export

## Deliverable

Implement the shared policy, WDL value, hidden-state, margin, and termination
heads. Add a versioned model manifest and export adapters for candidate static
8-bit CPU runtimes that exist in the competition sandbox.

The model definition must be usable by training without shipping training-loop
code in the bot bundle.

**Touches**

- `bots/`: add the runtime model definition, inference adapter, and tests.
- `arena/`: use the existing bundle import audit.
- `scripts/`: add export and benchmark entry points.
- `data/bot_versions/`: none.
- `training/morpheus/`: import the shared model definition for learning.

## Prerequisites

- [Part 03: Observation and actions](03-observation-actions.md)
- [Part 00b: Sandbox export preflight](00b-sandbox-export-preflight.md) with a
  `yes` result.

## Source specifications

- [Network](../bots/morpheus/network.md)
- [Action space](../bots/morpheus/action-space.md)
- [Observation tensor](../bots/morpheus/observation-tensor.md)
- [Frozen artifact](../bots/morpheus/evaluation.md#frozen-artifact)
- [Sandbox dependency pins](../../requirements-sandbox.txt)
- [Canonical judge pins](../../competition-module/competition/requirements.txt)

## Defaults and replacement measurement

This part starts with the specified 64-channel network and its approximate
0.35-million-parameter budget. It treats both values as initial guesses.

Part 09 replaces the deployment width, inference runtime, quantization, and
batch shapes through the coupled one-core benchmark. This part must not select
one of those items from parameter count alone.

## Implementation boundary

Keep the architecture and all playing-time model code inside the Morpheus bot
closure. Keep optimizers, replay buffers, and Modal code outside the closure.

The export must contain exact tensor and action schema versions, architecture,
quantization description, training run identity, and full weight SHA-256.

Test all heads on padded 18–21 rectangles. Test perspective sign, legal policy
normalization, and symmetry equivariance. Measure float-to-export error on
fixed tensors before Part 09 measures game-time behavior.

Define and version the 16 logarithmic enemy-army bin edges in the model schema.
Part 12 must import these edges for auxiliary labels instead of defining a
second copy.

## Isolated test

```bash
python -m pytest bots/morpheus/tests/test_network.py \
  bots/morpheus/tests/test_export.py \
  -m morpheus -q
```

```bash
python scripts/morpheus_export.py \
  --checkpoint training/morpheus/tests/fixtures/tiny-checkpoint \
  --output /tmp/morpheus-export
```

```bash
python -m arena.bundle morpheus --force
```

```bash
python competition-module/competition/matchup.py \
  bots/morpheus/run.sh bots/smoke/run.sh \
  --mode competition --seed 0
```

## Specification gaps

The specs do not select a training framework, CPU inference runtime, 8-bit
scheme, scale granularity, calibration set, normalization group count, bias
rules, auxiliary-head layout, or scalar-target normalization. This part owns
the missing army-bin edges because its output schema defines that head.

The competition sandbox lists Torch, JAX, Numba, NumPy, and safetensors, but it
does not list ONNX Runtime. Do not plan an unavailable runtime.

Static 8-bit convolution support and group normalization compatibility require
an export spike. A weight file that is 8-bit but expands to an unsafe runtime
does not satisfy the deployment requirement.

## Exit criterion

Answer `yes` if all heads have the specified shapes, perspective and symmetry
tests pass, the versioned army-bin edges are recorded, a sandbox-available
static 8-bit candidate loads from a manifest, and the bundle and competition
gates finish.

Answer `no` if no candidate export preserves the model contract. Part 09 makes
the final runtime choice.
