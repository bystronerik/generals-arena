# Observation tensor

## Decision

The network input is a `49 × 21 × 21` float tensor. The bot places the actual
`H × W` board at the upper-left and sets all values outside it to zero.
`board_mask` distinguishes padding from real mountains.

This fixed shape matches every competition rectangle. A variable crop is
rejected because it changes edge meaning and prevents simple tree batching.

## Plane contract

| # | Plane | Value |
| ---: | --- | --- |
| 1 | `board_mask` | 1 inside `H × W` |
| 2 | `visible_now` | 1 when type is not fog `0` or hidden structure `5` |
| 3 | `fog_nonstructure_now` | 1 for type `0`; can hide the enemy general |
| 4 | `fog_structure_now` | 1 for type `5` |
| 5 | `known_mountain` | persistent mountain mask |
| 6 | `known_passable_base` | persistent non-mountain, non-castle terrain |
| 7 | `known_castle` | persistent castle mask |
| 8 | `own_general` | persistent own-general cell |
| 9 | `known_enemy_general` | latched enemy-general cell, or all zero |
| 10 | `owned_now` | current owner code `1` |
| 11 | `enemy_visible` | visible owner code `2` |
| 12 | `neutral_visible` | visible owner code `0` on a passable cell |
| 13 | `owned_army` | transformed army on owned cells |
| 14 | `enemy_army_visible` | transformed visible enemy army |
| 15 | `ever_visible` | 1 after the first direct sight |
| 16 | `sight_age` | 0 until seen; then turns since sight divided by 1200 |
| 17 | `remembered_owned` | owner at the last sight was Morpheus |
| 18 | `remembered_enemy` | owner at the last sight was the enemy |
| 19 | `remembered_neutral` | owner at the last sight was neutral |
| 20 | `remembered_enemy_army` | transformed enemy army at last sight |
| 21 | `remembered_own_castle` | last seen castle owner was Morpheus |
| 22 | `remembered_enemy_castle` | last seen castle owner was the enemy |
| 23 | `belief_enemy_owner` | particle probability of enemy ownership |
| 24 | `belief_enemy_army_mean` | transformed particle mean enemy army |
| 25 | `belief_enemy_army_std` | transformed particle army deviation |
| 26 | `belief_enemy_general` | particle probability of the general cell |
| 27 | `belief_enemy_castle_owner` | particle probability of enemy castle control |
| 28 | `belief_enemy_visibility` | probability that the enemy sees this cell |
| 29 | `belief_owner_entropy` | binary enemy-owner entropy divided by `ln(2)` |
| 30 | `previous_move_source` | 1 at the previous source |
| 31 | `previous_move_destination` | 1 at the previous destination |
| 32 | `previous_move_kind` | 1 at destination for all-but-one, 0.5 for half |
| 33 | `previous_build_cell` | 1 at the previous build cell |
| 34 | `row_coordinate` | row divided by 20 |
| 35 | `column_coordinate` | column divided by 20 |
| 36 | `row_from_own_general` | signed row delta divided by 20 |
| 37 | `column_from_own_general` | signed column delta divided by 20 |
| 38 | `turn_fraction` | turn divided by 1200 |
| 39 | `pre_deathtouch_fraction` | `clip((800-turn)/800, 0, 1)` |
| 40 | `deathtouch_active` | 1 from turn 800 |
| 41 | `structure_growth_next` | 1 when the next resolved turn produces |
| 42 | `bulk_growth_countdown` | turns to next 50-turn growth, divided by 49 |
| 43 | `own_land_fraction` | own land divided by 441 |
| 44 | `enemy_land_fraction` | enemy land divided by 441 |
| 45 | `own_army_total` | transformed exact own total |
| 46 | `enemy_army_total` | transformed exact enemy total |
| 47 | `land_margin` | `(own-enemy)/(own+enemy+1)` |
| 48 | `army_margin` | `(own-enemy)/(own+enemy+1)` |
| 49 | `belief_ess` | particle effective sample size divided by count |

Planes 38-49 are constant across the playable board. Padding stays zero.

## Numeric transform

For any army value `x`:

```text
army_value(x) = clip(log(1 + max(x, 0)) / log(1 + 4096), 0, 1)
```

`4096` is an **initial guess**. Training statistics can replace this scale
without changing the plane meanings.

`structure_growth_next` uses `(turn + 1) mod 2 == 0`.
`bulk_growth_countdown` uses
`((50 - ((turn + 1) mod 50)) mod 50) / 49`.

## Fog and memory

Competition maps start with no castles. At the first frame, type `5` cells are
therefore mountains. Type `0` proves that the cell has no mountain or castle,
but it can still hide the enemy general. A later type-`0` to type-`5` change
identifies a new castle even while its owner and army remain hidden.

Static terrain and both general locations, once seen, never expire. Dynamic
owner and army memory keeps the last value and its age. Current visible data
always replaces memory before the tensor is built.

`ever_visible` gates `sight_age`. Both planes are zero before first sight.

## Rejected alternative

A recurrent hidden state without explicit memory is rejected. It would make
fog facts, tree-node equality, and replay diagnosis depend on an opaque vector.
The network can still learn temporal use from the explicit memory and belief
planes.
