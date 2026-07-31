# Heuristic measurement — round1

> **Not comparable to any post-refactor rating.** The Elo table below comes
> from the sequential Elo model that was removed on 2026-07-31. It is
> order-dependent: replaying the same games in a different order moved ratings
> by up to 250 Elo, and round-report Elo was built in list order, so it also
> disagreed with the leaderboard of its own day. Winrates, draw rates and turn
> counts on this page are unaffected — they are counts, not estimates. See
> [`docs/arena/ratings.md`](../../arena/ratings.md).

Generated: 2026-07-30T23:37:23Z
Games: 58 | Draw rate: 56.9% | Mean turns: 931.3

## Winrate by bot

| Bot | Games | W | L | D | Winrate | Draw rate | Mean turns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `army_convey` | 10 | 9 | 0 | 1 | 90.0% | 10.0% | 555.8 |
| `late_rush` | 10 | 8 | 2 | 0 | 80.0% | 0.0% | 602.5 |
| `fog_scout` | 10 | 6 | 0 | 4 | 60.0% | 40.0% | 761.3 |
| `expand_plus` | 8 | 1 | 3 | 4 | 12.5% | 50.0% | 921.9 |
| `phase_switch` | 14 | 1 | 3 | 10 | 7.1% | 71.4% | 1063.5 |
| `castle_builder` | 4 | 0 | 0 | 4 | 0.0% | 100.0% | 1200.0 |
| `castle_rush` | 14 | 0 | 2 | 12 | 0.0% | 85.7% | 1132.4 |
| `choke_control` | 10 | 0 | 3 | 7 | 0.0% | 70.0% | 978.7 |
| `garrison` | 10 | 0 | 4 | 6 | 0.0% | 60.0% | 1034.8 |
| `smoke` | 16 | 0 | 5 | 11 | 0.0% | 68.8% | 997.6 |
| `splitter` | 10 | 0 | 3 | 7 | 0.0% | 70.0% | 982.5 |

## Round Elo (this grid only)

| Rank | Bot | Elo | Games | W | L | D |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `army_convey` | 1616.0 | 10 | 9 | 0 | 1 |
| 2 | `fog_scout` | 1579.9 | 10 | 6 | 0 | 4 |
| 3 | `late_rush` | 1575.9 | 10 | 8 | 2 | 0 |
| 4 | `castle_builder` | 1495.2 | 4 | 0 | 0 | 4 |
| 5 | `expand_plus` | 1477.3 | 8 | 1 | 3 | 4 |
| 6 | `castle_rush` | 1473.8 | 14 | 0 | 2 | 12 |
| 7 | `phase_switch` | 1473.6 | 14 | 1 | 3 | 10 |
| 8 | `choke_control` | 1457.1 | 10 | 0 | 3 | 7 |
| 9 | `splitter` | 1455.5 | 10 | 0 | 3 | 7 |
| 10 | `smoke` | 1453.5 | 16 | 0 | 5 | 11 |
| 11 | `garrison` | 1442.2 | 10 | 0 | 4 | 6 |

## Notable matchups

- **fast_win**: `army_convey` beat `splitter` in 341 turns (seed 0)
- **high_castles**: `phase_switch` vs `castle_rush` — 2 vs 4 castles, draw (seed 0)
- **high_castles**: `castle_builder` vs `castle_rush` — 3 vs 4 castles, draw (seeds 0, 1)
- **high_castles**: `castle_rush` vs `phase_switch` — 4 vs 2 castles, draw (seeds 0, 1)

## Global arena leaderboard (all stored games)

Updated: 2026-07-30T23:37:32Z | Rated games: 150

| Rank | Bot | Elo | Games | W | L | D |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `army_convey` | 1689.6 | 20 | 18 | 0 | 2 |
| 2 | `fog_scout` | 1627.4 | 21 | 12 | 0 | 9 |
| 3 | `late_rush` | 1623.3 | 20 | 16 | 4 | 0 |
| 4 | `expander_python` | 1500.0 | 13 | 0 | 0 | 13 |
| 5 | `general_hunter` | 1500.0 | 12 | 0 | 0 | 12 |
| 6 | `castle_builder` | 1487.9 | 20 | 0 | 0 | 20 |
| 7 | `phase_switch` | 1457.7 | 28 | 2 | 6 | 20 |
| 8 | `castle_rush` | 1457.4 | 28 | 0 | 4 | 24 |
| 9 | `expand_plus` | 1453.3 | 28 | 1 | 6 | 21 |
| 10 | `smoke` | 1431.9 | 48 | 0 | 10 | 38 |
| 11 | `choke_control` | 1429.6 | 21 | 0 | 6 | 15 |
| 12 | `splitter` | 1428.0 | 21 | 0 | 6 | 15 |
| 13 | `garrison` | 1413.8 | 20 | 0 | 7 | 13 |

## Open questions

- Do economy-cluster bots (castle_builder, castle_rush, phase_switch) separate on Elo?
- Which new bots beat smoke on both seeds 0 and 1?
- Are draw-heavy matchups truncating before strategic differences show?
- Should the next round add expander_python or general_hunter as anchors?

Machine-readable: [`round1.json`](round1.json)
