# Post-revision validation — round1 Parameter revision 1

Generated: 2026-07-31T23:40:00Z
Grid: each revised bot vs `smoke`, seed 0, `--mode competition`
Revisions applied from `docs/research/strategies/*/md` § Parameter revision 1.
`optimize-existing.md` has no Parameter revision 1 — `smoke`, `expand_plus`, `castle_builder`, and `general_hunter` unchanged.

## Summary

| Bot | W | L | D | Turns | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| `fog_scout` | 1 | 0 | 0 | 403 | Win (player 0 capture) |
| `army_convey` | 1 | 0 | 0 | 447 | Win (player 0 capture) |
| `garrison` | 0 | 0 | 1 | 1200 | Draw (truncated) |
| `late_rush` | 1 | 0 | 0 | 374 | Win (player 0 capture); faster commit schedule |
| `splitter` | 0 | 0 | 1 | 1200 | Draw (truncated) |
| `choke_control` | 0 | 0 | 1 | 1200 | Draw (truncated) |
| `phase_switch` | 0 | 0 | 1 | 1200 | Draw (truncated); 3 castles built |
| `castle_rush` | 0 | 0 | 1 | 1200 | Draw (truncated); 4 castles built |

**Totals:** 8 games | 3 W | 0 L | 5 D | Draw rate 62.5% | Mean turns 908.1

All matches finished cleanly (no faults, no crashes).

## Parameter changes applied

| Bot | Key tweaks |
| --- | --- |
| `fog_scout` | `OPPONENT_BONUS` 2.0→3.5; fog march min army 3; vision saturation scaling |
| `army_convey` | Convey min army 3; frontier neighbor weight 1.5; opponent mult 3.0 |
| `garrison` | Phase floors 8/14/20/10; pressure divisor 3; stale enemy 30 turns; expansion scoring |
| `late_rush` | Earlier rally/accum/commit turns; reserve 10/12/12; main stack min 10; stall limit 8 |
| `splitter` | `SPLIT_MIN_ARMY` 16, `SPLIT_MARGIN` 4; threat-gated split; probe runner at turn 1000 |
| `choke_control` | Threat-scaled hold floor; `K` 8; abandon-hold; gate-only `W_CHOKE` 0.35; push-through at 800 |
| `phase_switch` | `EARLY_END` 60; `MAX_OWN_CASTLES` 3 |
| `castle_rush` | `RUSH_SURPLUS_MARGIN` 8; `RUSH_MIN_LAND` 6 |

## Observations vs round1 baseline (same seed, vs smoke)

- **Wins preserved or improved:** `fog_scout`, `army_convey`, and `late_rush` still beat `smoke` on seed 0. `late_rush` finished in 374 turns (was 685 pre-revision in round1 grid).
- **Draws unchanged:** `garrison`, `splitter`, `choke_control`, `phase_switch`, and `castle_rush` still draw at 1200 vs passive `smoke`.
- **Economy bots build as intended:** `phase_switch` built 3 castles (was capped at 2); `castle_rush` built 4 castles.
- **No regressions:** zero losses and zero faults in this mini-grid.

## Next steps

- Run full seed grid (0–2) and both seat orders per [`experiment-protocol.md`](../experiment-protocol.md).
- Compare draw rate and loss count against [`round1.md`](round1.md) before rating updates.
