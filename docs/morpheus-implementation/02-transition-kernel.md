# Part 02: Transition kernel

## Deliverable

Provide one transition contract with two implementations:

- a bundle-safe deployment implementation under `bots/morpheus/`;
- a training adapter over `GeneralsEnv(mode="competition")`.

Both implementations consume a complete state and two five-integer actions.
Both return the next state and terminal result.

**Touches**

- `bots/`: add the deployment state and transition modules.
- `arena/`: none.
- `scripts/`: none.
- `data/bot_versions/`: none.
- `training/morpheus/`: add the JAX engine adapter when Part 00 passes.

## Prerequisites

- [Part -1: Process substrate](-1-process-substrate.md)

Part 00 is not a hard dependency for the deployment transition. Its result
selects the optional GPU training adapter. A delayed or failed Modal preflight
must not block the bundle-safe transition or CPU adapter.

## Source specifications

- [Search transition](../bots/morpheus/search.md#decision)
- [Particle contents](../bots/morpheus/belief-state.md#particle-contents)
- [Competition rules](../../RULES.md)

The differential oracle is `competition-module/competition/matchup.py`
`make_transition(GeneralsEnv(mode="competition"))`.

## Defaults and replacement measurement

This part has no open-question default. Exact competition behavior is a hard
requirement.

## Implementation boundary

The deployment bot must not import `competition-module`. Implement the required
state and transition inside the Morpheus closure. Keep the implementation
independent of network, belief, and tree code.

The transition must cover:

- build-first resolution and exact live build cost;
- chasing, reinforcing, smaller-source priority, and the remaining tie order;
- full and half moves, combat, capture, and ownership transfer;
- structure and 50-turn growth;
- simultaneous general capture;
- deathtouch and chase defense from turn 800;
- turn-1200 truncation at the driver boundary.

The training adapter must use fixed-shape batched states. It must retain
`jax.jit`, `jax.vmap`, and `jax.lax.scan` when Part 00 permits the GPU path.

## Isolated test

```bash
python -m pytest bots/morpheus/tests/test_transition_kernel.py \
  -m morpheus -q
```

The test compares both implementations with the competition engine on tactical
fixtures and generated legal and invalid joint actions.

```bash
python competition-module/competition/matchup.py \
  bots/morpheus/run.sh bots/smoke/run.sh \
  --mode competition --seed 0
```

## Specification gaps

The unified bot convention says strategy code does not import
`competition-module`, while Morpheus requires the exact competition
transition. The bundle-safe duplicate is therefore necessary. Differential
tests are the guard against rules drift.

The design does not specify a canonical in-memory state type. This part must
define one without changing any game rule.

## Exit criterion

Answer `yes` if every fixture and generated transition matches the competition
engine and the runnable shell still finishes the competition gate.

Answer `no` on the first state, result, modifier, or turn-index mismatch.
