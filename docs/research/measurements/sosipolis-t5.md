# Heuristic measurement — sosipolis-t5

Generated: 2026-08-02T03:36:35Z
Games: 10 | Draw rate: 0.0% | Mean turns: 286.3

## Winrate by bot

| Bot | Games | W | L | D | Winrate | Draw rate | Mean turns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `macaria` | 10 | 10 | 0 | 0 | 100.0% | 0.0% | 286.3 |
| `sosipolis` | 10 | 0 | 10 | 0 | 0.0% | 0.0% | 286.3 |

## Round Elo (this grid only)

**Round-local ratings.** Fitted over this round's games only and anchored on the round's most-played bot, keyed on `bot_id` rather than on the content hash. They are **not comparable** to `data/ratings/leaderboard.md` or to any other round. Use the global fit for decisions.

Anchor: `macaria` pinned at 1500.0.

| Rank | Entity | Rating | 95% CI | Games | W | L | D | Prov. |
| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | `macaria` | 1500.0 | [1500, 1500] | 10 | 10 | 0 | 0 |  |
| 2 | `sosipolis` | 1165.8 | [915, 1417] | 10 | 0 | 10 | 0 |  |

## Notable matchups

- **fast_win**: `macaria` beat opponent in 229 turns (macaria vs sosipolis, seed 5)
- **fast_win**: `macaria` beat opponent in 127 turns (sosipolis vs macaria, seed 5)
- **fast_win**: `macaria` beat opponent in 177 turns (macaria vs sosipolis, seed 6)
- **fast_win**: `macaria` beat opponent in 170 turns (sosipolis vs macaria, seed 6)
- **fast_win**: `macaria` beat opponent in 221 turns (sosipolis vs macaria, seed 7)
- **fast_win**: `macaria` beat opponent in 173 turns (macaria vs sosipolis, seed 8)
- **fast_win**: `macaria` beat opponent in 221 turns (sosipolis vs macaria, seed 8)
- **fast_win**: `macaria` beat opponent in 139 turns (macaria vs sosipolis, seed 9)
- **fast_win**: `macaria` beat opponent in 371 turns (sosipolis vs macaria, seed 9)

## Open questions

- Which bots separate on Elo with games-per-pair sampling?
- Are draw-heavy matchups truncating before strategic differences show?
- Should the next round add expander_python or cm_* anchors?

Machine-readable: [`sosipolis-t5.json`](sosipolis-t5.json)
