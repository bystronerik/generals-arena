# Baseline validation — optimize-existing Parameter revision 1

Generated: 2026-07-31T23:45:00Z
Grid: each revised bot vs `smoke`, seed 0, `--mode competition`
Source: [`docs/research/strategies/optimize-existing.md`](../strategies/optimize-existing.md) § Parameter revision 1

## Summary

| Bot | W | L | D | Turns | Castles | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `expand_plus` | 0 | 0 | 1 | 1200 | 0 | Draw (truncated); land 182 vs 106 |
| `castle_builder` | 0 | 0 | 1 | 1200 | 3 | Draw (truncated); castles at turns 426, 476, 588 |
| `general_hunter` | 0 | 0 | 1 | 1200 | 0 | Draw (truncated); probe path enabled (no faults) |
| `smoke` | — | — | — | — | — | Frozen; no code change |

**Totals:** 3 games | 0 W | 0 L | 3 D | Draw rate 100% | Mean turns 1200

All matches finished cleanly (no faults, no crashes).

## Parameter changes applied

| Bot | Key tweaks |
| --- | --- |
| `smoke` | None (frozen anchor per R1.2) |
| `expand_plus` | `dest_value(enemy)` 12→6; H1 fogged-general army `1 + turn//2`; reserve cap removed |
| `castle_builder` | `MAX_TURN_TO_BUILD` 1050→600; H1 fogged-general army fix; probe candidate sample cap |
| `general_hunter` | H2 dispatch-tracked probes; `APPROACH_START=450`; `SENTRY_FROM` 700→450; `DEFEND_FROM` 780→500; H1 fogged-general army fix; probe candidate sample cap |

Shared defect fixes (H1, probe sampling) applied in each bot's `strategy_common.py` copy to prevent drift.

## Diversity checks

| Bot | Axis | Identity | Distance | Source |
| --- | --- | --- | --- | --- |
| `smoke` | Baseline | Unchanged frozen policy | Still first-valid expander | R1.2 freeze |
| `expand_plus` | Land rate | No castles, no hunt; scoring tier fix only | Enemy/neutral tie breaks on frontier_gain | R1.3 + optimize-existing §4.2 |
| `castle_builder` | Economy conservative | Still cap 3, floor price 35; earlier stop at 600 | Builds stop before `castle_rush` end gate | R1.4 + diversity §4 timing axis |
| `general_hunter` | Deathtouch kill | Probes not main-stack rush; no castles | Earlier approach vs `late_rush` commit window | R1.5 + optimize-existing §3.7 |

## Observations

- **H1 fix:** No repeated suicide W1 attacks observed; all three bots ran full 1200 turns without faults.
- **castle_builder timing:** Three castles built, all before turn 600 (`MAX_TURN_TO_BUILD` gate). Last castle at turn 588.
- **general_hunter probes:** Dispatch tracking restored; match completed under 150 ms/move budget (no fault storm).
- **expand_plus scoring:** Held more land than `smoke` at truncation (182 vs 106) but did not convert to a win on seed 0.

## Next steps

- Run paired seed grid (0–2, both seat orders) per [`experiment-protocol.md`](../experiment-protocol.md).
- Score R1.3 land prediction once §5.2 telemetry is in the rating store.
- Run `general_hunter` against `fog_scout`, `army_convey`, and `late_rush` before round 2.
