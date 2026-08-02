# Heuristic measurement — round7

Generated: 2026-08-02T17:43:16Z
Games: 750 | Draw rate: 0.7% | Mean turns: 383.3

## Winrate by bot

| Bot | Games | W | L | D | Winrate | Draw rate | Mean turns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `proteus` | 250 | 180 | 70 | 0 | 72.0% | 0.0% | 367.1 |
| `macaria` | 250 | 178 | 71 | 1 | 71.2% | 0.4% | 352.7 |
| `blitz` | 250 | 155 | 95 | 0 | 62.0% | 0.0% | 353.3 |
| `boom` | 250 | 107 | 143 | 0 | 42.8% | 0.0% | 435.8 |
| `sosipolis` | 250 | 82 | 163 | 5 | 32.8% | 2.0% | 388.0 |
| `cm_hunter` | 250 | 43 | 203 | 4 | 17.2% | 1.6% | 402.8 |

## Round Elo (this grid only)

**Round-local ratings.** Fitted over this round's games only and anchored on the round's most-played bot, keyed on `bot_id` rather than on the content hash. They are **not comparable** to `data/ratings/leaderboard.md` or to any other round. Use the global fit for decisions.

Anchor: `blitz` pinned at 1500.0.

| Rank | Entity | Rating | 95% CI | Games | W | L | D | Prov. |
| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | `proteus` | 1576.3 | [1515, 1638] | 250 | 180 | 70 | 0 |  |
| 2 | `macaria` | 1571.5 | [1510, 1633] | 250 | 178 | 71 | 1 |  |
| 3 | `blitz` | 1500.0 | [1500, 1500] | 250 | 155 | 95 | 0 |  |
| 4 | `boom` | 1375.1 | [1315, 1435] | 250 | 107 | 143 | 0 |  |
| 5 | `sosipolis` | 1312.6 | [1250, 1375] | 250 | 82 | 163 | 5 |  |
| 6 | `cm_hunter` | 1187.8 | [1119, 1257] | 250 | 43 | 203 | 4 |  |

## Notable matchups

- **fast_win**: `blitz` beat opponent in 323 turns (macaria vs blitz, seed 12242791)
- **fast_win**: `proteus` beat opponent in 227 turns (proteus vs sosipolis, seed 13193186)
- **fast_win**: `proteus` beat opponent in 221 turns (boom vs proteus, seed 19439257)
- **fast_win**: `sosipolis` beat opponent in 292 turns (cm_hunter vs sosipolis, seed 36001441)
- **fast_win**: `proteus` beat opponent in 225 turns (sosipolis vs proteus, seed 37226351)
- **fast_win**: `proteus` beat opponent in 129 turns (cm_hunter vs proteus, seed 44919527)
- **fast_win**: `boom` beat opponent in 388 turns (sosipolis vs boom, seed 46306602)
- **high_castles**: boom vs sosipolis — 2 vs 5 castles, boom (seed 54789429)
- **fast_win**: `sosipolis` beat opponent in 392 turns (cm_hunter vs sosipolis, seed 62403564)
- **fast_win**: `sosipolis` beat opponent in 387 turns (cm_hunter vs sosipolis, seed 66036494)
- **fast_win**: `cm_hunter` beat opponent in 172 turns (cm_hunter vs macaria, seed 68473165)
- **fast_win**: `proteus` beat opponent in 159 turns (proteus vs boom, seed 86111009)
- **fast_win**: `sosipolis` beat opponent in 388 turns (sosipolis vs macaria, seed 92748255)
- **fast_win**: `proteus` beat opponent in 171 turns (sosipolis vs proteus, seed 95542928)
- **fast_win**: `macaria` beat opponent in 328 turns (macaria vs cm_hunter, seed 97026858)
- **fast_win**: `macaria` beat opponent in 387 turns (sosipolis vs macaria, seed 99086163)
- **fast_win**: `blitz` beat opponent in 287 turns (blitz vs cm_hunter, seed 99850712)
- **fast_win**: `proteus` beat opponent in 221 turns (boom vs proteus, seed 100627598)
- **fast_win**: `macaria` beat opponent in 276 turns (proteus vs macaria, seed 101200625)
- **fast_win**: `sosipolis` beat opponent in 345 turns (sosipolis vs boom, seed 103609336)

## Open questions

- Which bots separate on Elo with games-per-pair sampling?
- Are draw-heavy matchups truncating before strategic differences show?
- Should the next round add expander_python or cm_* anchors?

Machine-readable: [`round7.json`](round7.json)
