# Heuristic measurement — sosipolis-r3a

Generated: 2026-08-02T03:21:41Z
Games: 20 | Draw rate: 0.0% | Mean turns: 382.8

## Winrate by bot

| Bot | Games | W | L | D | Winrate | Draw rate | Mean turns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `macaria` | 20 | 19 | 1 | 0 | 95.0% | 0.0% | 382.8 |
| `sosipolis` | 20 | 1 | 19 | 0 | 5.0% | 0.0% | 382.8 |

## Round Elo (this grid only)

**Round-local ratings.** Fitted over this round's games only and anchored on the round's most-played bot, keyed on `bot_id` rather than on the content hash. They are **not comparable** to `data/ratings/leaderboard.md` or to any other round. Use the global fit for decisions.

Anchor: `macaria` pinned at 1500.0.

| Rank | Entity | Rating | 95% CI | Games | W | L | D | Prov. |
| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | `macaria` | 1500.0 | [1500, 1500] | 20 | 19 | 1 | 0 |  |
| 2 | `sosipolis` | 1140.4 | [932, 1349] | 20 | 1 | 19 | 0 |  |

## Notable matchups

- **fast_win**: `macaria` beat opponent in 276 turns (macaria vs sosipolis, seed 0)
- **fast_win**: `macaria` beat opponent in 378 turns (sosipolis vs macaria, seed 0)
- **fast_win**: `macaria` beat opponent in 284 turns (macaria vs sosipolis, seed 2)
- **fast_win**: `macaria` beat opponent in 174 turns (sosipolis vs macaria, seed 3)
- **fast_win**: `macaria` beat opponent in 151 turns (macaria vs sosipolis, seed 4)
- **fast_win**: `macaria` beat opponent in 381 turns (sosipolis vs macaria, seed 4)
- **fast_win**: `macaria` beat opponent in 278 turns (macaria vs sosipolis, seed 5)
- **fast_win**: `macaria` beat opponent in 129 turns (sosipolis vs macaria, seed 5)
- **fast_win**: `macaria` beat opponent in 174 turns (macaria vs sosipolis, seed 6)
- **fast_win**: `macaria` beat opponent in 173 turns (sosipolis vs macaria, seed 7)
- **fast_win**: `macaria` beat opponent in 173 turns (macaria vs sosipolis, seed 8)
- **fast_win**: `macaria` beat opponent in 173 turns (sosipolis vs macaria, seed 8)

## Open questions

- Which bots separate on Elo with games-per-pair sampling?
- Are draw-heavy matchups truncating before strategic differences show?
- Should the next round add expander_python or cm_* anchors?

Machine-readable: [`sosipolis-r3a.json`](sosipolis-r3a.json)
