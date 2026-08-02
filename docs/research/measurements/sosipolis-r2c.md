# Heuristic measurement — sosipolis-r2c

Generated: 2026-08-01T19:48:17Z
Games: 20 | Draw rate: 0.0% | Mean turns: 289.6

## Winrate by bot

| Bot | Games | W | L | D | Winrate | Draw rate | Mean turns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `macaria` | 20 | 20 | 0 | 0 | 100.0% | 0.0% | 289.6 |
| `sosipolis` | 20 | 0 | 20 | 0 | 0.0% | 0.0% | 289.6 |

## Round Elo (this grid only)

**Round-local ratings.** Fitted over this round's games only and anchored on the round's most-played bot, keyed on `bot_id` rather than on the content hash. They are **not comparable** to `data/ratings/leaderboard.md` or to any other round. Use the global fit for decisions.

Anchor: `macaria` pinned at 1500.0.

| Rank | Entity | Rating | 95% CI | Games | W | L | D | Prov. |
| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | `macaria` | 1500.0 | [1500, 1500] | 20 | 20 | 0 | 0 |  |
| 2 | `sosipolis` | 1076.3 | [848, 1305] | 20 | 0 | 20 | 0 |  |

## Notable matchups

- **fast_win**: `macaria` beat opponent in 186 turns (sosipolis vs macaria, seed 0)
- **fast_win**: `macaria` beat opponent in 336 turns (sosipolis vs macaria, seed 1)
- **fast_win**: `macaria` beat opponent in 176 turns (macaria vs sosipolis, seed 2)
- **fast_win**: `macaria` beat opponent in 146 turns (macaria vs sosipolis, seed 3)
- **fast_win**: `macaria` beat opponent in 219 turns (sosipolis vs macaria, seed 3)
- **fast_win**: `macaria` beat opponent in 151 turns (macaria vs sosipolis, seed 4)
- **fast_win**: `macaria` beat opponent in 320 turns (sosipolis vs macaria, seed 4)
- **fast_win**: `macaria` beat opponent in 179 turns (macaria vs sosipolis, seed 5)
- **fast_win**: `macaria` beat opponent in 127 turns (sosipolis vs macaria, seed 5)
- **fast_win**: `macaria` beat opponent in 272 turns (macaria vs sosipolis, seed 6)
- **fast_win**: `macaria` beat opponent in 148 turns (sosipolis vs macaria, seed 6)
- **fast_win**: `macaria` beat opponent in 119 turns (sosipolis vs macaria, seed 7)
- **fast_win**: `macaria` beat opponent in 366 turns (macaria vs sosipolis, seed 8)
- **fast_win**: `macaria` beat opponent in 223 turns (sosipolis vs macaria, seed 8)
- **fast_win**: `macaria` beat opponent in 139 turns (macaria vs sosipolis, seed 9)
- **fast_win**: `macaria` beat opponent in 342 turns (sosipolis vs macaria, seed 9)

## Open questions

- Which bots separate on Elo with games-per-pair sampling?
- Are draw-heavy matchups truncating before strategic differences show?
- Should the next round add expander_python or cm_* anchors?

Machine-readable: [`sosipolis-r2c.json`](sosipolis-r2c.json)
