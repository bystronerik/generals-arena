# Part 00: Modal JAX preflight

## Deliverable

Prove whether the competition transition can run correctly on one Modal
Nvidia A100 80 GB before Morpheus selects a self-play architecture.

The preflight produces a JSON and Markdown report under
`docs/research/measurements/`. It records the device, JAX backend, cold compile
time, warm transition throughput, device memory, transfer time, and CPU/GPU
state parity.

**Touches**

- `bots/`: none.
- `arena/`: none.
- `scripts/`: add the thin Modal preflight entry point.
- `data/bot_versions/`: none.
- `competition-module/`: read and import only.

## Prerequisites

None.

## Source specifications

- [Training compute](../bots/morpheus/training.md#training-compute)
- [Training compute open question](../bots/morpheus/open-questions.md#training-compute)
- [Competition rules](../../RULES.md)

## Defaults and replacement measurement

This part assumes the decided training device and budget: Modal,
`gpu="A100-80GB"`, and approximately 48 A100 hours.

It assumes no GPU throughput. The replacement measurement is a small Modal run
over the competition preset. Local Apple Silicon is not a substitute for this
CUDA measurement.

## Implementation boundary

Use the current `GeneralsEnv.step(state, actions, pool)` API with fixed `21 ×
21` padded states. Compile a `jax.vmap` transition inside `jax.lax.scan`.

Run the same seeds and joint actions in a Modal CPU function and GPU function.
Compare armies, ownership, castles, time, winner, and public totals. Include
fixtures that exercise castle building and deathtouch.

Use `competition-module/examples/vectorized_example.py` as an API example.
Do not copy the generic README throughput claim. Do not use experimental
examples that omit the state pool or competition modifiers.

## Isolated test

```bash
python -m pytest training/morpheus/tests/test_jax_preflight.py \
  -m morpheus -q
```

```bash
modal run scripts/morpheus_modal_preflight.py
```

The Modal command writes
`docs/research/measurements/morpheus-jax-preflight.{json,md}`.

## Specification gaps

The submodule vectorizes the game transition. It does not implement Morpheus
particles, information hashes, matrix search, or backup. This preflight cannot
prove that the complete self-play stack benefits from the GPU.

GPU JAX installation is separate from the judge's CPU-only dependency set. The
Modal image must record its exact JAX, jaxlib, CUDA, and driver versions.

## Exit criterion

Answer `yes` only if JAX reports an A100 80 GB GPU, the competition transition
and modifiers compile, and every CPU/GPU state comparison matches.

Answer `no` otherwise. A `no` selects CPU self-play and reserves the A100 for
learning. Record all A100 time against the 48-hour budget.
