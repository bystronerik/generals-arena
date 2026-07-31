# Strategy Specification: classic_duel

Remote-only champion for classic generals.io. This bot never enters the
competition roster, `data/games/`, or Elo. See
[`human-95-plan.md`](human-95-plan.md) §2.3 and
[`diversity-constraints.md`](diversity-constraints.md) §2.4.

## 1. Goal

Win classic 1v1 games on live generals.io by combining three pillars:

1. **Expansion engine** — interior-to-frontier logistics (the skill
   `army_convey` measures best in the arena).
2. **City policy** — capture neutral pre-placed cities for mid-game income.
3. **Kill conversion and general defense** — finish with a stack strictly larger
   than the enemy general; hold a reserve at home.

## 2. Classic-safe constraints

These are hard rules, not tunables:

- **Never emit `pass=2` (build).** Classic has no build action.
- **No deathtouch branch.** No reference to turn 800, turn 1200, or a one-unit
  lethal touch onto the enemy general.
- **No competition draw clock.** Do not gate behavior on turn 1200.
- **General capture needs strictly more army** than the defender on classic.

Prefer **land-keyed** and **army-keyed** gates over turn-keyed gates.

## 3. Observation features used

- `type_grid` — `3` = city, `4` = general.
- `owner_grid` — `0` = neutral, `1` = me, `2` = opponent.
- `army_grid` — garrison on cities and generals.
- `my_land`, `my_army`, `turn`, `H`, `W`.

Derived each turn:

- `frontier_cells` — owned cells adjacent to capturable unowned cells.
- `neutral_cities` — cells with `owner == 0` and `type == 3`.
- `own_general_pos` — owned cell with `type == 4`.
- `enemy_general_pos` — remembered after first sighting (`owner == 2`, `type == 4`).

## 4. Named thresholds

| Constant | Value | Role |
| --- | --- | --- |
| `RESERVE_BASE` | 8 | Minimum army kept on own general |
| `RESERVE_PER_LAND` | 0.04 | Extra reserve per owned land tile |
| `CITY_MIN_OWNED_LAND` | 12 | Do not fund city attacks below this land count |
| `CITY_MAX_GARRISON` | 35 | Skip cities with garrison above this (remeasure from live) |
| `CITY_CAPTURE_SCORE` | 800 | Base score for a legal neutral-city capture |
| `CITY_LAND_BONUS` | 4 | Added to city score per owned land |
| `CONVEY_MIN_ARMY` | 3 | Minimum stack size for interior convey |
| `FRONTIER_NEIGHBOR_WEIGHT` | 1.5 | Frontier capture bonus from adjacent owned army |
| `OPPONENT_CAPTURE_MULT` | 3.0 | Multiplier for opponent-tile captures |
| `ENEMY_GENERAL_SCORE` | 10000 | Score for a legal attack onto the remembered enemy general |
| `GENERAL_ATTACK_MIN_MARGIN` | 1 | Need `src_army > dest_army + margin` (one stays on source) |
| `SCOUT_UNSIGHTED_LAND` | 80 | Enter scout mode when enemy general not sighted and land ≥ this |
| `SCOUT_FOG_CAPTURE_MULT` | 2.5 | Frontier capture multiplier for fog/unseen targets in scout mode |
| `SCOUT_MARCH_MIN_ARMY` | 3 | Minimum stack for fog march toward unrevealed terrain |
| `SCOUT_RESERVE_FACTOR` | 0.75 | Reserve multiplier in scout mode (floor `RESERVE_BASE`) |

## 5. Action priority (strict order)

1. **General defense** — if the own general is a source and the move would
   break the reserve floor, skip that source.
2. **Enemy general kill** — if the enemy general is visible and a frontier or
   adjacent owned cell can capture it with `src_army > dest_army + 1`, take
   the highest-scoring legal attack.
3. **Neutral city capture** — when `my_land >= CITY_MIN_OWNED_LAND` and **not**
   in scout mode (`enemy_general_pos` sighted or `my_land < SCOUT_UNSIGHTED_LAND`),
   score captures onto `owner == 0`, `type == 3` with garrison
   `<= CITY_MAX_GARRISON`. Score =
   `CITY_CAPTURE_SCORE + my_land * CITY_LAND_BONUS - garrison`.
4. **Frontier capture** — `army_convey`-style greedy capture from frontier cells
   (including opponent tiles with `OPPONENT_CAPTURE_MULT`; fog/unseen targets
   get `SCOUT_FOG_CAPTURE_MULT` when in scout mode).
4b. **Fog march** — in scout mode only, move interior stacks toward nearest
   unrevealed passable cell (BFS distance field).
5. **Interior convey** — move idle interior stacks toward the frontier by
   `army / (distance + 1)`.
6. **Frontier gathering** — push interior neighbors onto frontier tips.
7. **Fallback** — first legal move or `PASS`.

## 6. City policy detail

A neutral city is `type_grid == 3`, `owner_grid == 0`, with garrison in
`army_grid`.

Decisions:

1. **Whether to take** — only when `my_land >= CITY_MIN_OWNED_LAND` and
   garrison `<= CITY_MAX_GARRISON`.
2. **Which city** — highest score among legal captures this turn (distance and
   garrison folded into the score; closer/weaker cities rank higher when scores
   tie).
3. **How to fund** — only frontier or adjacent owned cells with enough army;
   do not strip the general below the reserve floor.
4. **When to stop** — after a city capture, resume the normal priority stack;
   no separate "city phase."

## 7. General reserve

Reserve on the own general:

```text
required = RESERVE_BASE + floor(my_land * RESERVE_PER_LAND)
```

A move from the general is legal only if the army left behind is at least
`required` (for `split=0`, need `army - 1 >= required` when moving one army).

This is a flat land-scaled floor, not a threat model (that axis belongs to
`garrison` on the competition roster).

## 8. Pseudocode

```text
FUNCTION act(obs):
    locate own general; update enemy general memory if sighted

    reserve = RESERVE_BASE + floor(obs.my_land * RESERVE_PER_LAND)

    // Priority 2: kill conversion
    IF enemy_general visible:
        move = best_capture onto enemy general with src > dest + 1
        IF move: RETURN move

    // Priority 3: city capture
    IF obs.my_land >= CITY_MIN_OWNED_LAND:
        move = best neutral city capture (garrison <= CITY_MAX_GARRISON)
        IF move: RETURN move

    // Priority 4-6: army_convey expansion engine (respect reserve on general)
    RETURN convey_capture_or_fallback(obs, reserve)
```

## 9. Diversity claim

`classic_duel` is **remote-only** and exempt from the competition roster axis
table (§2.4 of `diversity-constraints.md`). It does not compete with roster
bots for an axis. No competition bot may import or copy from `classic_duel`
without the three tests in `diversity-constraints.md` §4.1.

## 10. Falsifiable hypothesis

**If** `classic_duel` captures at least one neutral city before turn 200 on the
classic-approximate harness (seeds 0–4, both seat orders) **then** its land count
at turn 200 will exceed `army_convey` on the same seeds by at least 5% on
average. **Falsified if** city capture rate rises but land-at-200 does not.

## 11. Local verification

Classic harness (not competition matchup):

```bash
python arena/classic_match.py bots/classic_duel/run.sh bots/expand_plus/run.sh --seed 0
python scripts/remote_play.py --mode dry-run --bot classic_duel
```

Remote adapter must load the bot with `builds_dropped == 0`.

## 12. Parameter revision 1 (scout mode)

**Evidence:** Classic stress grid (seeds 0–4, both seats) showed 7 non-wins
(6 losses vs `army_convey`, 1 draw vs `smoke`). In 5 of 7, telemetry reported
`enemy_general_sighted=0` — the bot never found the enemy general despite
owning 130–428 land tiles.

**Change:**

| Constant | Old | New | Axis |
| --- | --- | --- | --- |
| `SCOUT_UNSIGHTED_LAND` | — | 80 | scout |
| `SCOUT_FOG_CAPTURE_MULT` | — | 2.5 | scout |
| `SCOUT_MARCH_MIN_ARMY` | — | 3 | scout |
| `SCOUT_RESERVE_FACTOR` | — | 0.75 | reserve |

Scout mode pauses neutral city capture (city priority) and biases expansion
toward fog. Reserve drops to `max(RESERVE_BASE, floor(required * 0.75))`.

**Hypothesis:** Sighting rate before turn 600 vs `army_convey` rises from 3/10
to ≥ 7/10 without increasing losses where the enemy general was already sighted.
