# 023 — sosipolis: belief hunt in map_memory + contact_mcts

Bot: [`bots/sosipolis/`](../../../bots/sosipolis/).
Prior: [`022-sosipolis-macaria-tune.md`](022-sosipolis-macaria-tune.md).

## Hypothesis

Folding a Macaria-style belief hunt into `MapMemory.hunt_target` and driving
Contact/Search MCTS at that cell raises `% enemy_general_sighted` vs `macaria`
above the 0% baseline on seeds 0–4 (both seats), without a new module file.

## Design

- `map_memory.hunt_target`: score candidates by reveal prune, footprint prior
  decay, travel decay; after contact prefer pool within `HUNT_CONTACT_RADIUS`.
- `contact_mcts`: high-prior `step_toward` / `gather_toward` hunt roots; fog
  toward hunt outranks `CONTACT_ENEMY_BONUS` fights.
- `search_mcts`: mild `SEARCH_HUNT_BONUS` toward mirror/hunt before contact.
- Soft defense weights kept light so the stack is not recalled home.

## Gate

Seed 0 vs `smoke`: sosipolis win at turn 318, 0 castles, no faults.

## Macaria

| Round | Games | sosipolis W-L-D | Mean turns | Sight (recorded A-seat) |
| --- | ---: | --- | ---: | --- |
| pre-hunt (cmp) | 5 A-seat | 0-5 | ~393 | **0 / 5** |
| sosipolis-hunt1 | 10 alt | **0-10-0** | 363.9 | — |
| hunt1-rec | 5 A-seat | **1-4** (seed 0 win) | — | **2 / 5** (seeds 0, 4) |

Recorded probe detail (A seat):

| Seed | Sight turn | Strike turns | Candidates end |
| ---: | ---: | ---: | ---: |
| 0 | 316 | 1 | 1 |
| 1 | none | 0 | 132 |
| 2 | none | 0 | 94 |
| 3 | none | 0 | 41 |
| 4 | 306 | **59** | 1 |

## Decision

**Keep** the belief hunt. Sight rate moved from 0% → 40% on the 5-seed A-seat
panel; seed 4 spent 59 turns in `strike`; seed 0 won. Alternate 10-game grid
still 0 wins — convert after sight is the next bottleneck (Strike finish /
stack size), not discovery alone.

Reports: [`../measurements/sosipolis-hunt1.md`](../measurements/sosipolis-hunt1.md).
