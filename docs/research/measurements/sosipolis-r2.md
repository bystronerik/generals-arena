# Heuristic measurement — sosipolis-r2

Generated: 2026-08-01T19:40:35Z
Games: 40 | Draw rate: 5.0% | Mean turns: 307.9

## Winrate by bot

| Bot | Games | W | L | D | Winrate | Draw rate | Mean turns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `macaria` | 40 | 38 | 0 | 2 | 95.0% | 5.0% | 307.9 |
| `sosipolis` | 40 | 0 | 38 | 2 | 0.0% | 5.0% | 307.9 |

## Round Elo (this grid only)

**Round-local ratings.** Fitted over this round's games only and anchored on the round's most-played bot, keyed on `bot_id` rather than on the content hash. They are **not comparable** to `data/ratings/leaderboard.md` or to any other round. Use the global fit for decisions.

Anchor: `macaria` pinned at 1500.0.

| Rank | Entity | Rating | 95% CI | Games | W | L | D | Prov. |
| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | `macaria` | 1500.0 | [1500, 1500] | 40 | 38 | 0 | 2 |  |
| 2 | `sosipolis` | 994.8 | [785, 1205] | 40 | 0 | 38 | 2 |  |

## Notable matchups

- **fast_win**: `macaria` beat opponent in 326 turns (sosipolis vs macaria, seed 0)
- **fast_win**: `macaria` beat opponent in 302 turns (sosipolis vs macaria, seed 1)
- **fast_win**: `macaria` beat opponent in 196 turns (macaria vs sosipolis, seed 2)
- **fast_win**: `macaria` beat opponent in 229 turns (sosipolis vs macaria, seed 3)
- **fast_win**: `macaria` beat opponent in 151 turns (macaria vs sosipolis, seed 4)
- **fast_win**: `macaria` beat opponent in 188 turns (sosipolis vs macaria, seed 4)
- **fast_win**: `macaria` beat opponent in 179 turns (macaria vs sosipolis, seed 5)
- **fast_win**: `macaria` beat opponent in 165 turns (sosipolis vs macaria, seed 5)
- **fast_win**: `macaria` beat opponent in 180 turns (macaria vs sosipolis, seed 6)
- **fast_win**: `macaria` beat opponent in 156 turns (sosipolis vs macaria, seed 6)
- **fast_win**: `macaria` beat opponent in 182 turns (sosipolis vs macaria, seed 7)
- **fast_win**: `macaria` beat opponent in 219 turns (macaria vs sosipolis, seed 8)
- **fast_win**: `macaria` beat opponent in 271 turns (sosipolis vs macaria, seed 8)
- **fast_win**: `macaria` beat opponent in 325 turns (sosipolis vs macaria, seed 9)
- **fast_win**: `macaria` beat opponent in 199 turns (macaria vs sosipolis, seed 10)
- **fast_win**: `macaria` beat opponent in 121 turns (sosipolis vs macaria, seed 10)
- **fast_win**: `macaria` beat opponent in 324 turns (macaria vs sosipolis, seed 11)
- **fast_win**: `macaria` beat opponent in 186 turns (sosipolis vs macaria, seed 11)
- **fast_win**: `macaria` beat opponent in 185 turns (macaria vs sosipolis, seed 12)
- **fast_win**: `macaria` beat opponent in 196 turns (sosipolis vs macaria, seed 12)

## Open questions

- Which bots separate on Elo with games-per-pair sampling?
- Are draw-heavy matchups truncating before strategic differences show?
- Should the next round add expander_python or cm_* anchors?

Machine-readable: [`sosipolis-r2.json`](sosipolis-r2.json)
