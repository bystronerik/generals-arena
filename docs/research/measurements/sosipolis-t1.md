# Heuristic measurement — sosipolis-t1

Generated: 2026-08-02T03:27:17Z
Games: 10 | Draw rate: 0.0% | Mean turns: 308.1

## Winrate by bot

| Bot | Games | W | L | D | Winrate | Draw rate | Mean turns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `macaria` | 10 | 10 | 0 | 0 | 100.0% | 0.0% | 308.1 |
| `sosipolis` | 10 | 0 | 10 | 0 | 0.0% | 0.0% | 308.1 |

## Round Elo (this grid only)

**Round-local ratings.** Fitted over this round's games only and anchored on the round's most-played bot, keyed on `bot_id` rather than on the content hash. They are **not comparable** to `data/ratings/leaderboard.md` or to any other round. Use the global fit for decisions.

Anchor: `macaria` pinned at 1500.0.

| Rank | Entity | Rating | 95% CI | Games | W | L | D | Prov. |
| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | `macaria` | 1500.0 | [1500, 1500] | 10 | 10 | 0 | 0 |  |
| 2 | `sosipolis` | 1165.8 | [915, 1417] | 10 | 0 | 10 | 0 |  |

## Notable matchups

- **fast_win**: `macaria` beat opponent in 186 turns (macaria vs sosipolis, seed 0)
- **fast_win**: `macaria` beat opponent in 335 turns (macaria vs sosipolis, seed 1)
- **fast_win**: `macaria` beat opponent in 299 turns (sosipolis vs macaria, seed 1)
- **fast_win**: `macaria` beat opponent in 166 turns (macaria vs sosipolis, seed 2)
- **fast_win**: `macaria` beat opponent in 276 turns (sosipolis vs macaria, seed 2)
- **fast_win**: `macaria` beat opponent in 326 turns (sosipolis vs macaria, seed 3)
- **fast_win**: `macaria` beat opponent in 148 turns (macaria vs sosipolis, seed 4)

## Open questions

- Which bots separate on Elo with games-per-pair sampling?
- Are draw-heavy matchups truncating before strategic differences show?
- Should the next round add expander_python or cm_* anchors?

Machine-readable: [`sosipolis-t1.json`](sosipolis-t1.json)
