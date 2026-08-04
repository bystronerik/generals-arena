# Morpheus complete-turn cost (Part 09a, Phases 0–3)

Continuation of
[`09a-complete-turn-cost.md`](../../morpheus-implementation/09a-complete-turn-cost.md).

## Phase 3 changes

- `select_path` performs zero network calls.
- An unexpanded node stops for leaf evaluation.
- Missing enemy-table priors return `EnemyPriorRequest`.
- Runtime batches enemy-prior tensors, resumes selection, then batches leaves.
- Enemy-prior and leaf timing stay on their own components.
- Frozen-batch and partial-simulation rules are preserved.

## Gate

`bots/morpheus/tests/test_selection_no_network.py` proves `select_path` does not
call the evaluator. Materialisation and leaf evaluation remain outside
selection.

## Earlier phases

Phases 0–2 remain as recorded in the previous revision of this note: warm
32-particle belief plus root fits a 140 ms deadline; Part 09 is still `no`
pending search cost work (Phases 4–6).

## Next

Start Phase 4 (hashing and backup). Do not lower the 8-simulation minimum or
promote 8 particles.
