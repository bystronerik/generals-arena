# Morpheus complete-turn cost (Part 09a, Phases 0–5)

Continuation of
[`09a-complete-turn-cost.md`](../../morpheus-implementation/09a-complete-turn-cost.md).

## Phase 5 changes

- Export dedicated online TorchScript entry points:
  - `model_policy.pt` — policy + pass for belief and enemy priors;
  - `model_policy_wdl.pt` — policy + pass + WDL for root and leaves.
- Keep the full-head `model.pt` for recovery and diagnostics.
- Online search does not evaluate auxiliary heads.
- Soft parity limits: `ONLINE_PARITY_LIMITS` in `bots/morpheus/export.py`.

## Gate

`bots/morpheus/tests/test_online_entry_points.py` checks:

- float online methods match full-head policy / WDL;
- qnnpack export MAE within existing soft limits;
- fbgemm / x86 when the host provides those engines (skip on Darwin
  qnnpack-only).

Modal Linux CPU seat
(`scripts/morpheus_modal_online_entry_points.py`) recorded **yes** for both
`fbgemm` and `x86`. Report:
[`morpheus-online-entry-points.md`](morpheus-online-entry-points.md).

## Phase 4 (committed)

Reuse observation digests; cache enemy-info weights; skip full prior arrays
when frozen; sort only legal candidates.

## Earlier phases

Warm 32-particle belief plus root fits a 140 ms deadline. Part 09 remains `no`
pending Phase 6 re-qualification.

## Next

Start Phase 6 (Part 09 re-qualification). Do not lower the 8-simulation
minimum or promote 8 particles.
