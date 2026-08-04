# Morpheus complete-turn cost (Part 09a, Phases 0–2)

First implementation increment after
[`09a-complete-turn-cost.md`](../../morpheus-implementation/09a-complete-turn-cost.md).

## Changes

- Phase 0: split propose vs particle-transition timing; record hashing/reply as
  elapsed; charge enemy-prior outside `selection`; per-consumer forward
  counters; strict JSON without `NaN`; sweep digest and commit hash; warm-up
  belief remeasure so one proposal spike cannot lock admission forever.
- Phase 1: skip castle-cost grids when neither action is BUILD; call
  `step_base` before turn 800; reuse legal masks / cost grids in expand and
  mandatory helpers.
- Phase 2: build enemy obs/memory once; dedupe on
  `(obs, memory, prev_action)` before tensors; array-backed internal
  observations on belief and search hot paths.

## Remeasure (single Torch thread)

Config: `scripts/configs/morpheus/online-sweep-32-belief-root.json`
(32 particles, 8 sims, 140 ms, proposal batch 32).

Raw report:
[`morpheus-complete-turn-cost.json`](morpheus-complete-turn-cost.json).

Isolated belief-plus-root microbench (no search, 18×18):

| Component | Approx p99 |
| --- | ---: |
| `belief_proposal` (early unique-heavy) | ~70–140 ms |
| `belief_proposal` (warm) | ~15–30 ms |
| `particle_transitions` | ~4–5 ms |
| `root_inference` | ~6 ms |
| Belief + root (warm) | ~25–50 ms |

Full coupled scenario (with search) still reports Part 09 verdict `no`:
belief-plus-root ok rate improved above zero, but min simulations and normal
p99 gates still fail. `enemy_prior_batch` is finite (no longer `NaN`).

## Phase 2 gate

Warm 32-particle belief plus root fits a 140 ms internal deadline with margin
for search and a positive guard. Early turns with many unique enemy infos still
consume most of the deadline on proposal under one Torch thread.

## Next

Start Phase 3 (selection without network; staged enemy priors). Do not lower
the 8-simulation minimum or promote 8 particles.
