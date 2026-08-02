# 022 — sosipolis vs macaria param tune (90% goal)

Bot: [`bots/sosipolis/`](../../../bots/sosipolis/).
Prior: [`021-sosipolis-phase-defense.md`](021-sosipolis-phase-defense.md).

## Goal

Win ≥90% of games vs `macaria` under competition mode. After each param
change, verify with ≥10 games (seeds alternate, both seats).

## Grid

Opponent: `macaria`. Seeds 0–4 alternate (10 games) unless noted. `--no-ratings`.

## Tune table

| Round | Change | W-L-D | Mean turns | Keep? |
| --- | --- | --- | ---: | --- |
| t0 | r3a baseline | 0-10-0 | ~291 | baseline |
| t1 | `CASTLE_MAX=0` | 0-10-0 | ~308 | yes (survival) |
| t2 | lower home banks / defense | 0-10-0 | ~305 | revert |
| t3 | budgets 90 + land bonuses + stage 55 | 0-10-0 | **369** | best survival |
| t4 | harder convert (stage 70, land slots 0) | 0-10-0 | 355 | revert |
| t5 | stage 35 + `CASTLE_MAX=1` (seeds 5–9) | 0-10-0 | 286 | revert |
| t6 | stage 20 + candidate bonus 90 | 0-10-0 | 343 | partial keep wiring |
| t7 | candidate bonus 200, low defense | 0-10-0 | 308 | revert |
| t8 | candidate rally → nearest home | 0-10-0 | 301 | bad rally |
| t9 | rally → candidate near contact sector | 0-10-0 | 350 | no win gain |

## Probe blocker (seed 0 losses)

Recorded traces (`sosipolis-diag-*`):

- Phase stays `search` then `contact`.
- `enemy_general_sighted` stays false for the whole game.
- Candidate count falls only slowly (~170 → ~120).
- Strike MCTS never runs.

Without sight, strike params cannot produce wins.

## Also fixed

- Added [`bots/sosipolis/agent.py`](../../../bots/sosipolis/agent.py) so
  `--record` / instrumented runner loads the bot (needs `agent.py`).

## Decision

Param-only moves on this panel stay at **0%** vs `macaria`. Best survival is
still **t3**. Next work must change contact hunt structure (directed push into
fog candidates behind the enemy front), not more strike/defense knobs.

Reports: [`../measurements/sosipolis-t0.md`](../measurements/sosipolis-t0.md)
through [`../measurements/sosipolis-t9.md`](../measurements/sosipolis-t9.md).
