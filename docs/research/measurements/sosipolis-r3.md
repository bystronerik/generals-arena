# Heuristic measurement — sosipolis-r3

Generated: 2026-08-02T03:18:32Z
Games: 40 | Draw rate: 5.0% | Mean turns: 308.7

## Winrate by bot

| Bot | Games | W | L | D | Winrate | Draw rate | Mean turns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `macaria` | 40 | 38 | 0 | 2 | 95.0% | 5.0% | 308.7 |
| `sosipolis` | 40 | 0 | 38 | 2 | 0.0% | 5.0% | 308.7 |

## Round Elo (this grid only)

**Round-local ratings.** Fitted over this round's games only and anchored on the round's most-played bot, keyed on `bot_id` rather than on the content hash. They are **not comparable** to `data/ratings/leaderboard.md` or to any other round. Use the global fit for decisions.

Anchor: `macaria` pinned at 1500.0.

| Rank | Entity | Rating | 95% CI | Games | W | L | D | Prov. |
| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | `macaria` | 1500.0 | [1500, 1500] | 40 | 38 | 0 | 2 |  |
| 2 | `sosipolis` | 1000.3 | [793, 1207] | 40 | 0 | 38 | 2 |  |

## Notable matchups

- **fast_win**: `macaria` beat opponent in 373 turns (sosipolis vs macaria, seed 0)
- **fast_win**: `macaria` beat opponent in 291 turns (macaria vs sosipolis, seed 1)
- **fast_win**: `macaria` beat opponent in 293 turns (sosipolis vs macaria, seed 1)
- **fast_win**: `macaria` beat opponent in 233 turns (macaria vs sosipolis, seed 2)
- **fast_win**: `macaria` beat opponent in 371 turns (sosipolis vs macaria, seed 2)
- **fast_win**: `macaria` beat opponent in 320 turns (macaria vs sosipolis, seed 3)
- **fast_win**: `macaria` beat opponent in 222 turns (sosipolis vs macaria, seed 3)
- **fast_win**: `macaria` beat opponent in 151 turns (macaria vs sosipolis, seed 4)
- **fast_win**: `macaria` beat opponent in 227 turns (sosipolis vs macaria, seed 4)
- **fast_win**: `macaria` beat opponent in 278 turns (macaria vs sosipolis, seed 5)
- **fast_win**: `macaria` beat opponent in 277 turns (sosipolis vs macaria, seed 5)
- **fast_win**: `macaria` beat opponent in 372 turns (macaria vs sosipolis, seed 6)
- **fast_win**: `macaria` beat opponent in 146 turns (sosipolis vs macaria, seed 6)
- **fast_win**: `macaria` beat opponent in 322 turns (macaria vs sosipolis, seed 7)
- **fast_win**: `macaria` beat opponent in 175 turns (sosipolis vs macaria, seed 7)
- **fast_win**: `macaria` beat opponent in 273 turns (macaria vs sosipolis, seed 8)
- **fast_win**: `macaria` beat opponent in 139 turns (macaria vs sosipolis, seed 9)
- **fast_win**: `macaria` beat opponent in 325 turns (sosipolis vs macaria, seed 9)
- **fast_win**: `macaria` beat opponent in 104 turns (macaria vs sosipolis, seed 10)
- **fast_win**: `macaria` beat opponent in 178 turns (sosipolis vs macaria, seed 10)

## Open questions

- Which bots separate on Elo with games-per-pair sampling?
- Are draw-heavy matchups truncating before strategic differences show?
- Should the next round add expander_python or cm_* anchors?

Machine-readable: [`sosipolis-r3.json`](sosipolis-r3.json)
