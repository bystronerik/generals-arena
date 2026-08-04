# Morpheus complete-turn cost (Part 09a, Phases 0–6)

Continuation of
[`09a-complete-turn-cost.md`](../../morpheus-implementation/09a-complete-turn-cost.md).

## Phase 6: Part 09 re-qualification

Verdict: **no**.

Gate order on Apple M3 Pro:

1. 32 / 8 targeted — fail (belief+root incomplete, min sims missed)
2. Board sizes and turn bands — fail
3. Positive admission guard — fail (no survivor)
4. Full 32 / 64 / 128 sweep — 0 / 36 survivors
5. Competition vs smoke — truncated draw at turn 1200
6. Submission harness — `REJECT normal_reply_timeout` (8 faults)

Canonical report:
[`morpheus-online-runtime.md`](morpheus-online-runtime.md) /
[`morpheus-online-runtime.json`](morpheus-online-runtime.json).

Gate-1 targeted report:
[`morpheus-complete-turn-cost-gate1.json`](morpheus-complete-turn-cost-gate1.json).

Do not lower the 8-simulation minimum, select a zero guard, or promote 8
particles.

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

Warm 32-particle belief-plus-root *forecast* can sit under a 140 ms deadline.
Timed warm moves still miss belief-plus-root completeness and the 8-simulation
floor, so Part 09 stays `no`.

## Next

Reduce complete-turn cost further under Part 09a. Re-run Phase 6 only after a
32-particle timed configuration finishes belief plus root and at least 8
simulations on every warm normal move.
