# Heuristic measurement — sosipolis-tip1

Generated: 2026-08-02T12:19:30Z
Games: 10 | Draw rate: 0.0% | Mean turns: 323.0

## Winrate by bot

| Bot | Games | W | L | D | Winrate | Draw rate | Mean turns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `macaria` | 10 | 9 | 1 | 0 | 90.0% | 0.0% | 323.0 |
| `sosipolis` | 10 | 1 | 9 | 0 | 10.0% | 0.0% | 323.0 |

## Round Elo (this grid only)

**Round-local ratings.** Fitted over this round's games only and anchored on the round's most-played bot, keyed on `bot_id` rather than on the content hash. They are **not comparable** to `data/ratings/leaderboard.md` or to any other round. Use the global fit for decisions.

Anchor: `macaria` pinned at 1500.0.

| Rank | Entity | Rating | 95% CI | Games | W | L | D | Prov. |
| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | `macaria` | 1500.0 | [1500, 1500] | 10 | 9 | 1 | 0 |  |
| 2 | `sosipolis` | 1244.6 | [1013, 1476] | 10 | 1 | 9 | 0 |  |

## Notable matchups

- **fast_win**: `macaria` beat opponent in 228 turns (sosipolis vs macaria, seed 0)
- **fast_win**: `macaria` beat opponent in 387 turns (macaria vs sosipolis, seed 1)
- **fast_win**: `macaria` beat opponent in 287 turns (sosipolis vs macaria, seed 1)
- **fast_win**: `macaria` beat opponent in 381 turns (macaria vs sosipolis, seed 2)
- **fast_win**: `macaria` beat opponent in 374 turns (sosipolis vs macaria, seed 2)
- **fast_win**: `macaria` beat opponent in 224 turns (macaria vs sosipolis, seed 3)
- **fast_win**: `macaria` beat opponent in 151 turns (macaria vs sosipolis, seed 4)
- **fast_win**: `sosipolis` beat opponent in 228 turns (sosipolis vs macaria, seed 4)

## Open questions

- Which bots separate on Elo with games-per-pair sampling?
- Are draw-heavy matchups truncating before strategic differences show?
- Should the next round add expander_python or cm_* anchors?

Machine-readable: [`sosipolis-tip1.json`](sosipolis-tip1.json)
