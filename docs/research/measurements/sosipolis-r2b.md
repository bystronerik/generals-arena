# Heuristic measurement — sosipolis-r2b

Generated: 2026-08-01T19:46:27Z
Games: 20 | Draw rate: 0.0% | Mean turns: 416.6

## Winrate by bot

| Bot | Games | W | L | D | Winrate | Draw rate | Mean turns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `macaria` | 20 | 20 | 0 | 0 | 100.0% | 0.0% | 416.6 |
| `sosipolis` | 20 | 0 | 20 | 0 | 0.0% | 0.0% | 416.6 |

## Round Elo (this grid only)

**Round-local ratings.** Fitted over this round's games only and anchored on the round's most-played bot, keyed on `bot_id` rather than on the content hash. They are **not comparable** to `data/ratings/leaderboard.md` or to any other round. Use the global fit for decisions.

Anchor: `macaria` pinned at 1500.0.

| Rank | Entity | Rating | 95% CI | Games | W | L | D | Prov. |
| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | `macaria` | 1500.0 | [1500, 1500] | 20 | 20 | 0 | 0 |  |
| 2 | `sosipolis` | 1076.3 | [848, 1305] | 20 | 0 | 20 | 0 |  |

## Notable matchups

- **fast_win**: `macaria` beat opponent in 326 turns (macaria vs sosipolis, seed 0)
- **fast_win**: `macaria` beat opponent in 278 turns (sosipolis vs macaria, seed 0)
- **fast_win**: `macaria` beat opponent in 299 turns (sosipolis vs macaria, seed 1)
- **fast_win**: `macaria` beat opponent in 276 turns (sosipolis vs macaria, seed 2)
- **fast_win**: `macaria` beat opponent in 178 turns (sosipolis vs macaria, seed 3)
- **fast_win**: `macaria` beat opponent in 281 turns (macaria vs sosipolis, seed 4)
- **fast_win**: `macaria` beat opponent in 231 turns (sosipolis vs macaria, seed 4)
- **fast_win**: `macaria` beat opponent in 179 turns (macaria vs sosipolis, seed 5)
- **fast_win**: `macaria` beat opponent in 127 turns (sosipolis vs macaria, seed 5)
- **fast_win**: `macaria` beat opponent in 320 turns (macaria vs sosipolis, seed 7)
- **fast_win**: `macaria` beat opponent in 135 turns (macaria vs sosipolis, seed 8)

## Open questions

- Which bots separate on Elo with games-per-pair sampling?
- Are draw-heavy matchups truncating before strategic differences show?
- Should the next round add expander_python or cm_* anchors?

Machine-readable: [`sosipolis-r2b.json`](sosipolis-r2b.json)
