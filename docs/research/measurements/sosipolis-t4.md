# Heuristic measurement — sosipolis-t4

Generated: 2026-08-02T03:33:39Z
Games: 10 | Draw rate: 0.0% | Mean turns: 354.7

## Winrate by bot

| Bot | Games | W | L | D | Winrate | Draw rate | Mean turns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `macaria` | 10 | 10 | 0 | 0 | 100.0% | 0.0% | 354.7 |
| `sosipolis` | 10 | 0 | 10 | 0 | 0.0% | 0.0% | 354.7 |

## Round Elo (this grid only)

**Round-local ratings.** Fitted over this round's games only and anchored on the round's most-played bot, keyed on `bot_id` rather than on the content hash. They are **not comparable** to `data/ratings/leaderboard.md` or to any other round. Use the global fit for decisions.

Anchor: `macaria` pinned at 1500.0.

| Rank | Entity | Rating | 95% CI | Games | W | L | D | Prov. |
| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | `macaria` | 1500.0 | [1500, 1500] | 10 | 10 | 0 | 0 |  |
| 2 | `sosipolis` | 1165.8 | [915, 1417] | 10 | 0 | 10 | 0 |  |

## Notable matchups

- **fast_win**: `macaria` beat opponent in 284 turns (macaria vs sosipolis, seed 0)
- **fast_win**: `macaria` beat opponent in 334 turns (sosipolis vs macaria, seed 0)
- **fast_win**: `macaria` beat opponent in 355 turns (macaria vs sosipolis, seed 1)
- **fast_win**: `macaria` beat opponent in 336 turns (sosipolis vs macaria, seed 1)
- **fast_win**: `macaria` beat opponent in 166 turns (macaria vs sosipolis, seed 2)
- **fast_win**: `macaria` beat opponent in 332 turns (sosipolis vs macaria, seed 2)
- **fast_win**: `macaria` beat opponent in 333 turns (macaria vs sosipolis, seed 4)
- **fast_win**: `macaria` beat opponent in 231 turns (sosipolis vs macaria, seed 4)

## Open questions

- Which bots separate on Elo with games-per-pair sampling?
- Are draw-heavy matchups truncating before strategic differences show?
- Should the next round add expander_python or cm_* anchors?

Machine-readable: [`sosipolis-t4.json`](sosipolis-t4.json)
