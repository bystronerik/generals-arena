# Heuristic measurement — sosipolis-kubic-vs-macaria-10

Generated: 2026-08-02T15:18:30Z
Games: 10 | Draw rate: 0.0% | Mean turns: 238.0

Experiment notes: [`sosipolis-kubic-conveyor-theory-test.md`](sosipolis-kubic-conveyor-theory-test.md)
(pure Kubic conveyor, no live MCTS; bot code reverted after this grid).

## Winrate by bot

| Bot | Games | W | L | D | Winrate | Draw rate | Mean turns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `macaria` | 10 | 10 | 0 | 0 | 100.0% | 0.0% | 238.0 |
| `sosipolis` | 10 | 0 | 10 | 0 | 0.0% | 0.0% | 238.0 |

## Round Elo (this grid only)

**Round-local ratings.** Fitted over this round's games only and anchored on the round's most-played bot, keyed on `bot_id` rather than on the content hash. They are **not comparable** to `data/ratings/leaderboard.md` or to any other round. Use the global fit for decisions.

Anchor: `macaria` pinned at 1500.0.

| Rank | Entity | Rating | 95% CI | Games | W | L | D | Prov. |
| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | `macaria` | 1500.0 | [1500, 1500] | 10 | 10 | 0 | 0 |  |
| 2 | `sosipolis` | 1168.2 | [916, 1420] | 10 | 0 | 10 | 0 |  |

## Notable matchups

- **fast_win**: `macaria` beat opponent in 234 turns (macaria vs sosipolis, seed 0)
- **fast_win**: `macaria` beat opponent in 189 turns (sosipolis vs macaria, seed 1)
- **fast_win**: `macaria` beat opponent in 224 turns (macaria vs sosipolis, seed 2)
- **fast_win**: `macaria` beat opponent in 340 turns (sosipolis vs macaria, seed 3)
- **fast_win**: `macaria` beat opponent in 130 turns (sosipolis vs macaria, seed 4)
- **fast_win**: `macaria` beat opponent in 177 turns (sosipolis vs macaria, seed 5)
- **fast_win**: `macaria` beat opponent in 83 turns (sosipolis vs macaria, seed 7)
- **fast_win**: `macaria` beat opponent in 321 turns (macaria vs sosipolis, seed 8)
- **fast_win**: `macaria` beat opponent in 271 turns (macaria vs sosipolis, seed 9)

## Open questions

- Which bots separate on Elo with games-per-pair sampling?
- Are draw-heavy matchups truncating before strategic differences show?
- Should the next round add expander_python or cm_* anchors?

Machine-readable: [`sosipolis-kubic-vs-macaria-10.json`](sosipolis-kubic-vs-macaria-10.json)
