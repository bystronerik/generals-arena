# Morpheus complete-turn cost (Part 09a, Phases 0–4)

Continuation of
[`09a-complete-turn-cost.md`](../../morpheus-implementation/09a-complete-turn-cost.md).

## Phase 4 changes

- Reuse observation, memory, and digests within one search depth step.
- Add `info_state_key_prehashed` / `enemy_info_hash_prehashed`.
- Cache particle enemy-info hashes by reservoir `version` in `backup_node`.
- Skip full 3970-entry prior arrays when widening is frozen.
- Sort only legal candidate indices in `policy_ordered_candidates`.

## Gate

`bots/morpheus/tests/test_hash_backup_parity.py` checks fixed-seed root
statistics and actions against the Phase-3 baseline after the same completed
simulations.

## Phase 3 (committed)

Selection performs zero network calls; enemy priors stage outside selection.
See commit message for Part 09a Phase 3.

## Earlier phases

Warm 32-particle belief plus root fits a 140 ms deadline. Part 09 remains `no`
pending Phases 5–6.

## Next

Start Phase 5 (network export entry points). Do not lower the 8-simulation
minimum or promote 8 particles.
