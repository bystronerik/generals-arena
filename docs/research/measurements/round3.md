# Heuristic measurement — round3

Generated: 2026-07-31T09:56:33Z
Games: 140 | Draw rate: 55.0% | Mean turns: 913.3

## Winrate by bot

| Bot | Games | W | L | D | Winrate | Draw rate | Mean turns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `army_convey` | 28 | 26 | 1 | 1 | 92.9% | 3.6% | 622.1 |
| `late_rush` | 20 | 14 | 4 | 2 | 70.0% | 10.0% | 608.8 |
| `cm_harvester` | 6 | 4 | 2 | 0 | 66.7% | 0.0% | 452.7 |
| `cm_hunter` | 6 | 4 | 2 | 0 | 66.7% | 0.0% | 452.7 |
| `fog_scout` | 20 | 11 | 0 | 9 | 55.0% | 45.0% | 842.5 |
| `cm_expander` | 6 | 1 | 2 | 3 | 16.7% | 50.0% | 965.7 |
| `phase_switch` | 28 | 2 | 4 | 22 | 7.1% | 78.6% | 1090.3 |
| `expand_plus` | 16 | 1 | 4 | 11 | 6.2% | 68.8% | 1047.3 |
| `castle_builder` | 8 | 0 | 0 | 8 | 0.0% | 100.0% | 1200.0 |
| `castle_rush` | 28 | 0 | 4 | 24 | 0.0% | 85.7% | 1106.8 |
| `choke_control` | 20 | 0 | 5 | 15 | 0.0% | 75.0% | 1037.0 |
| `cm_random` | 6 | 0 | 2 | 4 | 0.0% | 66.7% | 971.8 |
| `garrison` | 20 | 0 | 7 | 13 | 0.0% | 65.0% | 1036.6 |
| `smoke` | 48 | 0 | 20 | 28 | 0.0% | 58.3% | 893.6 |
| `splitter` | 20 | 0 | 6 | 14 | 0.0% | 70.0% | 998.9 |

## Round Elo (this grid only)

| Rank | Bot | Elo | Games | W | L | D |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `army_convey` | 1742.8 | 28 | 26 | 1 | 1 |
| 2 | `fog_scout` | 1615.6 | 20 | 11 | 0 | 9 |
| 3 | `late_rush` | 1591.7 | 20 | 14 | 4 | 2 |
| 4 | `cm_hunter` | 1530.0 | 6 | 4 | 2 | 0 |
| 5 | `cm_harvester` | 1524.7 | 6 | 4 | 2 | 0 |
| 6 | `castle_builder` | 1489.5 | 8 | 0 | 0 | 8 |
| 7 | `cm_expander` | 1488.4 | 6 | 1 | 2 | 3 |
| 8 | `cm_random` | 1474.7 | 6 | 0 | 2 | 4 |
| 9 | `phase_switch` | 1473.6 | 28 | 2 | 4 | 22 |
| 10 | `expand_plus` | 1472.1 | 16 | 1 | 4 | 11 |
| 11 | `castle_rush` | 1464.1 | 28 | 0 | 4 | 24 |
| 12 | `choke_control` | 1438.9 | 20 | 0 | 5 | 15 |
| 13 | `splitter` | 1426.7 | 20 | 0 | 6 | 14 |
| 14 | `garrison` | 1409.7 | 20 | 0 | 7 | 13 |
| 15 | `smoke` | 1357.3 | 48 | 0 | 20 | 28 |

## Notable matchups

- **fast_win**: `late_rush` beat opponent in 374 turns (late_rush vs smoke, seed 0)
- **fast_win**: `army_convey` beat opponent in 303 turns (army_convey vs choke_control, seed 0)
- **fast_win**: `late_rush` beat opponent in 383 turns (late_rush vs splitter, seed 0)
- **high_castles**: phase_switch vs castle_rush — 3 vs 4 castles, draw (seed 0)
- **high_castles**: castle_rush vs phase_switch — 4 vs 3 castles, draw (seed 0)
- **high_castles**: castle_builder vs castle_rush — 3 vs 4 castles, draw (seed 0)
- **high_castles**: castle_rush vs castle_builder — 4 vs 3 castles, draw (seed 0)
- **high_castles**: castle_builder vs castle_rush — 3 vs 4 castles, draw (seed 1)
- **high_castles**: castle_rush vs castle_builder — 4 vs 3 castles, draw (seed 1)
- **high_castles**: castle_builder vs phase_switch — 3 vs 3 castles, draw (seed 0)
- **high_castles**: phase_switch vs castle_builder — 3 vs 3 castles, draw (seed 0)
- **high_castles**: castle_builder vs phase_switch — 3 vs 3 castles, draw (seed 1)
- **high_castles**: phase_switch vs castle_builder — 3 vs 3 castles, draw (seed 1)
- **high_castles**: castle_rush vs phase_switch — 4 vs 3 castles, draw (seed 0)
- **high_castles**: phase_switch vs castle_rush — 3 vs 4 castles, draw (seed 0)
- **high_castles**: castle_rush vs phase_switch — 4 vs 3 castles, draw (seed 1)
- **high_castles**: phase_switch vs castle_rush — 3 vs 4 castles, draw (seed 1)
- **fast_win**: `cm_hunter` beat opponent in 245 turns (cm_hunter vs smoke, seed 0)
- **fast_win**: `cm_hunter` beat opponent in 375 turns (smoke vs cm_hunter, seed 0)
- **fast_win**: `cm_hunter` beat opponent in 187 turns (cm_hunter vs smoke, seed 1)

## Open questions

- Do economy-cluster bots (castle_builder, castle_rush, phase_switch) separate on Elo?
- Which new bots beat smoke on both seeds 0 and 1?
- Are draw-heavy matchups truncating before strategic differences show?
- Should the next round add expander_python or general_hunter as anchors?

Machine-readable: [`round3.json`](round3.json)
