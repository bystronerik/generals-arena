# 013 — boom: migrated fast-expand economy

Bot: [`bots/boom/`](../../../bots/boom/). Baselines: `expand_plus`,
`castle_builder`, `smoke`, `late_rush`. Migration of the generals-bot Boom
strategy (grid-native rewrite; see
[`strategies/boom.md`](../strategies/boom.md)).

## Hypothesis

Break-even-scheduled expansion (`army ≥ distance` runs, frontier-first
captures) out-produces the pool, the measured home guard survives the
mid-game, and the endgame latch converts the lead before the 1200-turn
draw. City purchases were rewritten as castle builds funded by field
surplus (`cost × 2 ≤ spare`).

**One change vs the source:** rules adaptation only — city module →
castle-build module, deathtouch finish/defence, `FORCE_COMMIT_TURN = 950`
draw guard. Other thresholds carried over unchanged as hypotheses.

## Seed grid

- Opponents: `smoke`, `expand_plus`, `castle_builder`, `late_rush`.
- Seeds: 0, 1, 2.
- Mode: `--mode competition` only.

## Metrics (verification matches, seed 0)

| Matchup | Games | boom W-L-D | End turn | Castles built |
| --- | --- | --- | --- | --- |
| boom vs smoke | 1 | 1-0-0 | 379 (capture) | 0 |
| boom vs expand_plus | 1 | 1-0-0 | 337 (capture) | 0 |

No faults. Castle builds fired 0 times: free captures and the attack latch
dominated the ladder in both games. The build channel is unit-tested
(`test_boom_builds_castle_from_idle_surplus`); whether
`BUILD_SPARE_MULTIPLE = 2.0` should loosen so builds fire in real games is
the follow-up question for the measurement round.

## Decision

**Keep**, provisionally. Revert if the full grid shows boom losing its
general mid-boom (guard model failure), drawing at 1200 with a large army
lead (latch failure), or faulting.
