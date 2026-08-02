# Heuristic measurement — sosipolis-tip2

Generated: 2026-08-02T12:39:16Z
Games: 10 | Draw rate: 0.0% | Mean turns: 332.7

## Winrate by bot

| Bot | Games | W | L | D | Winrate | Draw rate | Mean turns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `macaria` | 10 | 9 | 1 | 0 | 90.0% | 0.0% | 332.7 |
| `sosipolis` | 10 | 1 | 9 | 0 | 10.0% | 0.0% | 332.7 |

## Round Elo (this grid only)

**Round-local ratings.** Fitted over this round's games only and anchored on the round's most-played bot, keyed on `bot_id` rather than on the content hash. They are **not comparable** to `data/ratings/leaderboard.md` or to any other round. Use the global fit for decisions.

Anchor: `macaria` pinned at 1500.0.

| Rank | Entity | Rating | 95% CI | Games | W | L | D | Prov. |
| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | `macaria` | 1500.0 | [1500, 1500] | 10 | 9 | 1 | 0 |  |
| 2 | `sosipolis` | 1244.6 | [1013, 1476] | 10 | 1 | 9 | 0 |  |

## Notable matchups

- **fast_win**: `macaria` beat opponent in 337 turns (macaria vs sosipolis, seed 1)
- **fast_win**: `macaria` beat opponent in 319 turns (sosipolis vs macaria, seed 2)
- **fast_win**: `sosipolis` beat opponent in 323 turns (macaria vs sosipolis, seed 3)
- **fast_win**: `macaria` beat opponent in 286 turns (sosipolis vs macaria, seed 3)
- **fast_win**: `macaria` beat opponent in 151 turns (macaria vs sosipolis, seed 4)
- **fast_win**: `macaria` beat opponent in 132 turns (sosipolis vs macaria, seed 4)

## Open questions

- Which bots separate on Elo with games-per-pair sampling?
- Are draw-heavy matchups truncating before strategic differences show?
- Should the next round add expander_python or cm_* anchors?

Machine-readable: [`sosipolis-tip2.json`](sosipolis-tip2.json)
