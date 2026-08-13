# Joe r2 — step 3500 vs step 3000, matched round (2026-08-13)

Second arena round for `bots/joe`, measuring the newly exported EMA
checkpoint (**step 3500** of `joe-M-vast-20260813-0213`) against the
checkpoint the first round rated (**step 3000**). Both checkpoints come
from the same still-running Phase 4 training job
([plan](../strategies/averagejoe-competition-plan.md)).

## Verdict: **unproven** — improvement by every reading except the one that counts

The contrast is **+104.5 ± 49.4 Elo, CI₉₅ [+7.8, +201.3],
P(step 3500 stronger) = 0.9829**. Per
[decision-rule.md](../../arena/decision-rule.md) that is `unproven`, and
only just: `P ≥ 0.95` passes, but the interval's low end (+7.8) misses the
`> +10` floor by 2.2 Elo. Every structural gate passes (below), so the
failure is precision, not validity — and no replication round has run,
which the rule requires independently of any threshold.

Direct evidence points the same way: on 20 matched maps played both
ways, step 3500 beat step 3000 **26W–13L–1D** (66.7% of decisive games).

## Why the baseline was re-measured instead of reused

r1 rated step 3000 as `joe@04974d0e206b`. Comparing the new checkpoint
against *that* number would have been a cross-round comparison, which the
decision rule forbids. So `bots/joe_base/` — a frozen copy of the
step-3000 program, weights verified byte-identical by sha256
(`dcc99e80…`) — played this round alongside the new one.

That precaution earned its keep. The same program in two rounds:

| Entity | Round | Rating |
| --- | --- | ---: |
| `joe@04974d0e206b` | joe-entry-r1 | 2481.9 |
| `joe_base@384b7e012261` | joe-r2 | 2540.9 |

**+59.0 ± 71.1 Elo of drift between byte-identical programs** — the same
phenomenon the rule documents at +46 Elo for macaria. Had the new
checkpoint been compared against r1's number, the claim would have been
**+163 Elo** instead of the honest **+104.5**: roughly 36% of the
apparent gain was round drift.

## Design

- Round `joe-r2`, 440 games, all `mode=competition`, engine
  `9e3b9d13cca5`, `--seat-policy alternate --strict-versions`.
- **Matched maps.** `expand_pair_seeds` derives its seed draw from a
  per-pair RNG keyed on the two bot ids, so a shared `--round-seed` gives
  different arms different maps. An explicit `--seeds 2001-2020` overrides
  that: both arms played the same 20 maps against each panel bot, in both
  orientations (40 games per arm-opponent).
- Panel: the r1 panel unchanged — morpheus-rs, macaria, sosipolis, metro,
  cm_expander (anchor). 200 panel games per arm, plus 40 head-to-head.
- Host: M3 Pro, 11 jobs, no other load. The two arms are deterministic;
  morpheus-rs is deadline-driven, so host state binds on the panel side.

## Results

| Opponent | step 3500 | step 3000 |
| --- | :---: | :---: |
| `morpheus-rs` | 38W 2L 0D | 36W 4L 0D |
| `macaria` | 38W 2L 0D | 40W 0L 0D |
| `sosipolis` | 39W 1L 0D | 39W 1L 0D |
| `metro` | 39W 1L 0D | 37W 3L 0D |
| `cm_expander` | 40W 0L 0D | 39W 0L 1D |
| **panel total** | **194W 6L 0D** | **191W 8L 1D** |
| head-to-head | 26W 13L 1D | — |

Fit after the full-pool refit:

| Entity | Rating | Games | W/L/D |
| --- | ---: | ---: | --- |
| `joe@b7cd053f905b` (3500) | 2645.4 ± 54.1 | 240 | 220/19/1 |
| `joe_base@384b7e012261` (3000) | 2540.9 ± 49.8 | 240 | 204/34/2 |

Gate: both registered ✓; one engine, one panel, one seed set ✓; same
connectivity group ✓; 240 games per arm (≥ 200) ✓; 40 per arm-opponent
(≥ 30) ✓; 239 and 238 decisive (≥ 60) ✓. Only the verdict threshold
fails.

## The panel is saturated

Both arms win ~97% of panel games. A panel that both arms beat almost
always carries almost no information about which is better — nearly all
the discriminating evidence came from the 40 head-to-head games, which is
why the interval is ±49 despite 240 games per arm.

Consequences for the next round:

- Spend games on **head-to-head volume**, not panel breadth. Modelling the
  head-to-head as a logistic pairing (p ≈ 0.667, 173.7 Elo per logit)
  puts `SE = 12.75` — the "proven flat" precision — at roughly **840
  head-to-head games**, about 40 minutes at the observed rate.
- Longer term the panel needs opponents joe does not dominate. Archived
  joe checkpoints are the natural ladder; the paper's own reference-Elo
  eval works this way.

## Standing

This is a measurement of an interim checkpoint, not a published claim.
The final Phase 4 checkpoint is the one worth a replication round. Two
things carry forward: the +104.5 point estimate as the current best guess
at 500 iterations of progress, and the confirmation that the frozen-copy
baseline protocol is worth its cost.

Joe still built **zero castles** in all 440 games — unchanged from r1,
and now measured across two checkpoints 500 iterations apart.
