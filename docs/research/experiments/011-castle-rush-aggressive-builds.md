# 011 — castle_rush: aggressive castle parameters

Bot: [`bots/castle_rush/`](../../../bots/castle_rush/). Baselines: `castle_builder`, `expand_plus`, `smoke`.

## Hypothesis

Building castles earlier (turn 10 vs 20), more often (cooldown 25 vs 40), and more of them (cap 4 vs 2), with a thinner surplus margin (5 vs 15), yields higher late-game income than `castle_builder` — at the cost of slower early expansion from resting the general more.

**One change vs `castle_builder`:** the build parameters (`RUSH_START`, `RUSH_CAP`, `RUSH_COOLDOWN`, `RUSH_MIN_LAND`, `RUSH_SURPLUS_MARGIN`). Build, relocate, and expand sub-procedures are reused from `castle_builder` / `expand_plus`.

## Seed grid

- Opponents: `smoke`, `expand_plus`, `castle_builder`, `general_hunter`.
- Seeds: 0, 1, 2.
- Mode: `--mode competition` only.

## Metrics (verification match, seed 0 vs smoke)

| Matchup | Games | castle_rush W-L-D | Mean turns | Castles built |
| --- | --- | --- | --- | --- |
| castle_rush vs smoke | 1 | 0-0-1 | 1200.0 | 4 (turns 108, 164, 276, 928) |

Full seed grid pending tournament run. Primary signal: castles built per game (target >= 3 on average vs `castle_builder`'s 2).

## Decision

**Keep**, provisionally. Revert if `castle_rush` builds fewer castles per game than `castle_builder`, faults more often, or loses to `general_hunter` by a margin worse than `castle_builder`.

**Telemetry gap:** same as `002` — income claim needs final land/army at truncation (schema follow-up).
