# Heuristic measurement — sosipolis-kubic-clocked-macaria-10

Generated: 2026-08-02T15:54:50Z
Games: 20 | Draw rate: 0.0% | Mean turns: 340.7

Experiment notes: [`sosipolis-kubic-clocked-theory.md`](sosipolis-kubic-clocked-theory.md)
(clocked Kubic shell + purpose MCTS).

## Winrate by bot

| Bot | Games | W | L | D | Winrate | Draw rate | Mean turns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `macaria` | 20 | 14 | 6 | 0 | 70.0% | 0.0% | 340.7 |
| `sosipolis` | 20 | 6 | 14 | 0 | 30.0% | 0.0% | 340.7 |

## Round Elo (this grid only)

**Round-local ratings.** Fitted over this round's games only and anchored on the round's most-played bot, keyed on `bot_id` rather than on the content hash. They are **not comparable** to `data/ratings/leaderboard.md` or to any other round. Use the global fit for decisions.

Anchor: `macaria` pinned at 1500.0.

| Rank | Entity | Rating | 95% CI | Games | W | L | D | Prov. |
| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | `macaria` | 1500.0 | [1500, 1500] | 20 | 14 | 6 | 0 |  |
| 2 | `sosipolis` | 1366.7 | [1211, 1522] | 20 | 6 | 14 | 0 |  |

## Notable matchups

- **fast_win**: `macaria` beat opponent in 326 turns (macaria vs sosipolis, seed 0)
- **fast_win**: `macaria` beat opponent in 228 turns (sosipolis vs macaria, seed 0)
- **fast_win**: `macaria` beat opponent in 372 turns (macaria vs sosipolis, seed 2)
- **fast_win**: `macaria` beat opponent in 268 turns (macaria vs sosipolis, seed 3)
- **fast_win**: `macaria` beat opponent in 277 turns (sosipolis vs macaria, seed 3)
- **fast_win**: `macaria` beat opponent in 197 turns (macaria vs sosipolis, seed 4)
- **fast_win**: `macaria` beat opponent in 379 turns (sosipolis vs macaria, seed 4)
- **fast_win**: `macaria` beat opponent in 178 turns (sosipolis vs macaria, seed 5)
- **fast_win**: `macaria` beat opponent in 275 turns (macaria vs sosipolis, seed 6)
- **fast_win**: `macaria` beat opponent in 270 turns (sosipolis vs macaria, seed 7)
- **fast_win**: `sosipolis` beat opponent in 312 turns (sosipolis vs macaria, seed 8)
- **fast_win**: `macaria` beat opponent in 129 turns (macaria vs sosipolis, seed 9)
- **fast_win**: `sosipolis` beat opponent in 345 turns (sosipolis vs macaria, seed 9)

## Open questions

- Which bots separate on Elo with games-per-pair sampling?
- Are draw-heavy matchups truncating before strategic differences show?
- Should the next round add expander_python or cm_* anchors?

Machine-readable: [`sosipolis-kubic-clocked-macaria-10.json`](sosipolis-kubic-clocked-macaria-10.json)
