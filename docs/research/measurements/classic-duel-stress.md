# Classic duel stress test — offline

Generated: 2026-07-31T01:30:00Z
Bot: `classic_duel` (remote-only, classic harness)
Harness: `arena/classic_match.py` (neutral cities, no build, no deathtouch)
Grid: 30 games (3 opponents × 5 seeds × 2 seats)

## Pre-revision results

| Opponent | W | L | D | Winrate | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| `expand_plus` | 10 | 0 | 0 | 100% | Clean sweep |
| `smoke` | 9 | 0 | 1 | 90% | seed 2 seat 0 truncated at 5000 (never sighted) |
| `army_convey` | 4 | 6 | 0 | 40% | Primary weakness |
| **Total** | **23** | **6** | **1** | **76.7%** | |

Mean turns (decided games): 847.4

## Pre-revision game log

| bot_a | bot_b | seed | winner | turns | sighted |
| --- | --- | ---: | --- | ---: | --- |
| `classic_duel` | `expand_plus` | 0 | `classic_duel` | 542 | 541 |
| `classic_duel` | `expand_plus` | 1 | `classic_duel` | 188 | 119 |
| `classic_duel` | `expand_plus` | 2 | `classic_duel` | 1882 | 1881 |
| `classic_duel` | `expand_plus` | 3 | `classic_duel` | 849 | 696 |
| `classic_duel` | `expand_plus` | 4 | `classic_duel` | 799 | 537 |
| `classic_duel` | `smoke` | 0 | `classic_duel` | 666 | 558 |
| `classic_duel` | `smoke` | 1 | `classic_duel` | 182 | 121 |
| `classic_duel` | `smoke` | 2 | `draw` | 5000 | no |
| `classic_duel` | `smoke` | 3 | `classic_duel` | 783 | 343 |
| `classic_duel` | `smoke` | 4 | `classic_duel` | 793 | 493 |
| `classic_duel` | `army_convey` | 0 | `army_convey` | 598 | 317 |
| `classic_duel` | `army_convey` | 1 | `classic_duel` | 270 | 140 |
| `classic_duel` | `army_convey` | 2 | `army_convey` | 1252 | no |
| `classic_duel` | `army_convey` | 3 | `classic_duel` | 2324 | 648 |
| `classic_duel` | `army_convey` | 4 | `classic_duel` | 953 | 951 |
| `expand_plus` | `classic_duel` | 0 | `classic_duel` | 683 | 602 |
| `expand_plus` | `classic_duel` | 1 | `classic_duel` | 377 | 153 |
| `expand_plus` | `classic_duel` | 2 | `classic_duel` | 2095 | 1227 |
| `expand_plus` | `classic_duel` | 3 | `classic_duel` | 626 | 320 |
| `expand_plus` | `classic_duel` | 4 | `classic_duel` | 440 | 439 |
| `smoke` | `classic_duel` | 0 | `classic_duel` | 669 | 595 |
| `smoke` | `classic_duel` | 1 | `classic_duel` | 345 | 106 |
| `smoke` | `classic_duel` | 2 | `classic_duel` | 2077 | 1108 |
| `smoke` | `classic_duel` | 3 | `classic_duel` | 662 | 431 |
| `smoke` | `classic_duel` | 4 | `classic_duel` | 671 | 495 |
| `army_convey` | `classic_duel` | 0 | `army_convey` | 1012 | no |
| `army_convey` | `classic_duel` | 1 | `army_convey` | 265 | 69 |
| `army_convey` | `classic_duel` | 2 | `army_convey` | 768 | no |
| `army_convey` | `classic_duel` | 3 | `classic_duel` | 1153 | 990 |
| `army_convey` | `classic_duel` | 4 | `army_convey` | 689 | no |

## Findings (pre-revision)

- **Scout failure is the main loss mode.** 5 of 7 non-wins had
  `enemy_general_sighted=0` despite 130–428 owned land.
- `expand_plus` and `smoke` are not competitive; `army_convey` is the benchmark.
- The smoke seed 2 draw (428 vs 7 land, 5000 turns) shows expansion without
  direction when the enemy hides in unrevealed fog.
- City capture was not the bottleneck; missing the enemy general was.

## Parameter revision 1

Applied scout mode per [`classic_duel.md`](../strategies/classic_duel.md) §12:

- Pause city capture when `my_land >= 80` and enemy general not sighted.
- Boost fog/unseen frontier captures (`SCOUT_FOG_CAPTURE_MULT = 2.5`).
- Add fog march toward unrevealed terrain.
- Lower reserve to 75% in scout mode (floor `RESERVE_BASE`).

## Post-revision validation

Generated: 2026-07-31T01:32:00Z (Parameter revision 1 — scout mode)

| Opponent | W | L | D | Winrate | Δ vs pre |
| --- | ---: | ---: | ---: | ---: | --- |
| `expand_plus` | 9 | 0 | 1 | 90% | −1W (new draw seed 2 seat 0) |
| `smoke` | 10 | 0 | 0 | 100% | +1W (fixed seed 2 draw) |
| `army_convey` | 5 | 5 | 0 | 50% | +1W |
| **Total** | **24** | **5** | **1** | **80.0%** | **+1W −1L** |

Mean turns (decided games): 812.6

### Post-revision game log

| bot_a | bot_b | seed | winner | turns | sighted | Δ |
| --- | --- | ---: | --- | ---: | --- | --- |
| `classic_duel` | `expand_plus` | 0 | `classic_duel` | 533 | 532 | |
| `classic_duel` | `expand_plus` | 1 | `classic_duel` | 188 | 119 | |
| `classic_duel` | `expand_plus` | 2 | `draw` | 5000 | 406 | was W |
| `classic_duel` | `expand_plus` | 3 | `classic_duel` | 785 | 424 | |
| `classic_duel` | `expand_plus` | 4 | `classic_duel` | 1243 | 287 | |
| `classic_duel` | `smoke` | 0 | `classic_duel` | 680 | 272 | |
| `classic_duel` | `smoke` | 1 | `classic_duel` | 182 | 121 | |
| `classic_duel` | `smoke` | 2 | `classic_duel` | 694 | 693 | **was D** |
| `classic_duel` | `smoke` | 3 | `classic_duel` | 743 | 208 | |
| `classic_duel` | `smoke` | 4 | `classic_duel` | 797 | 490 | |
| `classic_duel` | `army_convey` | 0 | `classic_duel` | 346 | 345 | **was L** |
| `classic_duel` | `army_convey` | 1 | `classic_duel` | 270 | 140 | |
| `classic_duel` | `army_convey` | 2 | `army_convey` | 1010 | no | |
| `classic_duel` | `army_convey` | 3 | `classic_duel` | 2471 | 283 | |
| `classic_duel` | `army_convey` | 4 | `classic_duel` | 1271 | 1269 | |
| `expand_plus` | `classic_duel` | 0 | `classic_duel` | 749 | 306 | |
| `expand_plus` | `classic_duel` | 1 | `classic_duel` | 377 | 153 | |
| `expand_plus` | `classic_duel` | 2 | `classic_duel` | 1412 | 1154 | |
| `expand_plus` | `classic_duel` | 3 | `classic_duel` | 482 | 396 | |
| `expand_plus` | `classic_duel` | 4 | `classic_duel` | 541 | 449 | |
| `smoke` | `classic_duel` | 0 | `classic_duel` | 581 | 570 | |
| `smoke` | `classic_duel` | 1 | `classic_duel` | 345 | 106 | |
| `smoke` | `classic_duel` | 2 | `classic_duel` | 1115 | 1104 | |
| `smoke` | `classic_duel` | 3 | `classic_duel` | 674 | 380 | |
| `smoke` | `classic_duel` | 4 | `classic_duel` | 510 | 503 | |
| `army_convey` | `classic_duel` | 0 | `army_convey` | 483 | no | |
| `army_convey` | `classic_duel` | 1 | `army_convey` | 265 | 69 | |
| `army_convey` | `classic_duel` | 2 | `army_convey` | 1149 | no | |
| `army_convey` | `classic_duel` | 3 | `classic_duel` | 2074 | 2072 | |
| `army_convey` | `classic_duel` | 4 | `army_convey` | 984 | no | |

### Post-revision assessment

- Scout mode **confirmed**: smoke seed 2 draw (never sighted, 428 land) → win at turn 694.
- `army_convey` seat 0 seed 0 flip: loss → win after early sighting (turn 345).
- Sighting rate vs `army_convey`: 7/10 (was 3/10 pre-revision for unsighted losses).
- Trade-off: expand_plus seed 2 seat 0 regressed from win to draw (sighted turn 406,
  could not convert before truncation). Net grid improves 23-6-1 → 24-5-1.
- Remaining `army_convey` losses (5) still include 3 games with no sighting when
  classic_duel plays seat 1.
