# S4 trail penalty rated round (s4-trail-r1)

**Claims tested:** (1) the S4 trail penalty
([spec](../strategies/joe-rs-noundo.md)) at δ = 6 and δ = 4 plays no worse
than the shipped δ = 0 default; (2) the full joe-rs selection stack
(S1 Gumbel T = 1 + S4 δ = 6) plays no worse than the pure argmax of the
same network (Python joe). All four entities hold the f16-rounded
`joe-M7F4-vast-20260819-0207` **step-11000** checkpoint.

## Setup

- Entities: baseline `joe-rs@90ad9423d5f0` (S1, penalty off);
  `joe_rs_d6@a8f88ad3c7c0` and `joe_rs_d4@d1348740eada` (frozen copies,
  `run.sh` pins `JOE_RS_NOUNDO` 6/4, window 8); `joe@c0bf59ae3986`
  (Python, plain argmax). The copies differ from joe-rs only in `run.sh`,
  so each contrast measures its knob.
- Panel: aegis, macaria, boom, cm_hunter, cm_expander (anchor). Panel
  games satisfy the per-opponent gate and connectivity; every arm beats
  the panel ~24:1, so the contrast information is the head-to-head mass.
- `--seat-policy alternate --strict-versions --mode competition --timeout
  600` throughout; engine `9e3b9d1`; Rust-only invocations at 8 jobs,
  invocations with joe at 6 (one JAX process per game).
- Round `s4-trail-r1` (2026-08-21): panel invocations 202, 212 (all Rust
  arms + panel, 16 games/pair) and 222, 232 (joe + panel); H2H
  invocations 101/121/171 (base–d6: 1,032 games), 111/131 (base–d4:
  782), 151/161 (joe–d6: 750), 181 (joe–base: 400). 4,276 games total;
  974–2,374 per arm.
- Host: the dev arm64 Mac (11 cores, 18 GB), idle apart from the runner.
  No arm is deadline-driven — every bot computes one fixed forward per
  turn.

## Gates

All pass for every contrast: one engine, all `mode=competition`, all
arms registered and in one connectivity group with the global anchor,
≥ 974 games per arm, ≥ 32 games per (arm, opponent), ≥ 845 decisive per
arm.

## Results

| Contrast | Δ ± SE | CI₉₅ | P(B > A) | H2H (B–A–draw) | Verdict |
| --- | --- | --- | --- | --- | --- |
| d6 − base | **−0.67 ± 9.83** | (−19.9, +18.6) | 0.473 | 481–477–74 | **no change (proven flat)** |
| d4 − base | **−0.33 ± 12.47** | (−24.8, +24.1) | 0.490 | 356–356–70 | **no change (proven flat)** |
| d6 − joe | −15.88 ± 11.10 | (−37.6, +5.9) | 0.076 | 325–360–65 | unproven |
| base − joe | −15.21 ± 12.31 | (−39.3, +8.9) | 0.108 | 170–182–48 | unproven |

## Reading

**The trail penalty costs nothing.** Both δ arms are proven flat against
the shipped default — d4's head-to-head is literally tied — while the
diagnostic grid's behavioral gains (zero strict cycles, revisit halved,
mean conversion −68 turns) ride along free. The round supports flipping
the default to δ = 6 once it replicates.

**The joe lean sits entirely on the S1 side, and it is a lean, not a
verdict.** The decomposition is clean: d6 − joe (−15.9) ≈ (base − joe)
(−15.2) + (d6 − base) (−0.7). Neither joe contrast reaches any verdict
threshold (regression needs CI-high < −10; both are positive), and the
prior round measured the same S1-vs-argmax quantity on the step-10000
checkpoint at **+3.78 ± 10.69** ([joe-selection-s1](joe-selection-s1.md)).
Two same-design rounds, opposite mild leans, both CIs spanning zero: the
joint picture is a selection stack strength-neutral against the argmax
within this measurement's ~±15 Elo resolution. Proving a −15 effect real
would need SE ≈ 2.5 — tens of thousands of games — and is not the
decision this round was bought for.

## Replication caveat (decision-rule)

This is r1. A separately scheduled r2 — same arms, fresh round-seeds —
is owed before these verdicts are final, and it doubles as the second
look at the joe lean. No arm is deadline-driven, so the M6 fragility
binds least here, but the rule stands. By decision the default flipped to
δ = 6 on 2026-08-21 ahead of r2, and the frozen arm directories were
deleted; their registry entries and this round's stored games remain the
evidence. r2 can still reverse the flip.
