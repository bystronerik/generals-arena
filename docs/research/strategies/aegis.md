# Strategy spec — aegis

Bot: `bots/aegis/`. Migrated from the generals-bot repo as a grid-native
rewrite. Grounded idea: **turtle + counterattack — hold a compact,
defensible ring, destroy incoming attacks, then commit into the window
their spent attack opens.** Baselines for comparison: `garrison`,
`choke_control`, `smoke`, `blitz`.

Rule references: [`RULES.md`](../../../RULES.md) sections 03 (build castles),
06 (visibility), 07 (deathtouch). Source-tuned values transfer as
**hypotheses**.

## 1. The keep

The load-bearing idea: the reserve lives one cell *off* the general. On
the general it only fights the battle the opponent picks; beside it, the
same army covers the door (any attacker is seen one turn out — exactly
what the keep needs to step back in), hunts raiders, and leads the
counter. `man_the_door` is what makes this safe.

## 2. Defence sizing (the measured turtle)

`defense_need` plans against `wave_estimate`: the largest stack they have
*shown* (× `WAVE_GROWTH = 1.0`), or `WAVE_FRAC = 0.10` of their mobile
army, floored at `MIN_WAVE = 8` — capped by `max_strike_force`
(`opp_army − (opp_land − 1)`, exact from the aggregates). Attrition credit
(1/cell crossed, capped at 10) discounts distant threats; threat distance
decays one cell per turn after contact is lost. The garrison never exceeds
`MAX_DEFENSE_FRAC = 0.50` of our army (0.35 in econ mode) unless the
attack is imminent.

## 3. Active defence ladder

Interception (kill the stack with local superiority, never the general) →
general sortie (only when exactly one enemy is adjacent and survivors
still cover the next wave) → man the door → emergency garrison → hunt
(meet the raider within `HUNT_RADIUS = 7`, leashed to `HUNT_LEASH = 6`) →
pull-home rally on the keep.

## 4. Counterattack

Triggers (`counter_reason`): their army collapsed on our defence
(`drop ≥ max(12, 0.20 × their peak)` inside 25 turns, with army edge
≥ 1.15); the big stack we watched died; overwhelming ratio (≥ 1.5 after
200); land exhausted; or **late-game parity** (turn ≥ 350, army ≥ theirs)
— which under the arena's 1200-turn draw doubles as the draw guard. Once
triggered it holds `COUNTER_HOLD = 80` turns, aborts after losing 15% of
starting land, reinforces a designated strike for `GATHER_TICKS = 25`,
then collection-walks it at the target (known general > nearest enemy
cell > remembered mirror guess).

## 5. Growth and economy

Perimeter claims maximize compactness (nearest ring, most own/wall
neighbours, away from the enemy) — the differentiator vs `garrison`'s
reserve-holding and vs sprawl-happy expansion. Border raids (an enemy
castle outranks any plain cell) and short marches follow. **Castle builds
replace city grabs**: within `BUILD_RADIUS = 10` of the general, when
spare army (`my_army − defense_need`) covers the price, gather onto the
cheapest safe site and build (`BUILD_ANYTIME = True`, up to
`MAX_CASTLES = 3`). A turtle that compounds production while turtling is
the thesis intact. Chipping deleted (enemy castles regrow).

## 6. Arena-rule adaptations (vs the generals-bot source)

| Rule | Adaptation |
| --- | --- |
| No neutral cities | `city_step` → `build_step` (safe home castle builds); chipping deleted. |
| Deathtouch (800) | Kill shot accepts any 2-army neighbour; `chase_defence` from 780. |
| Draw at 1200 | `LATE_TURN = 350` parity commit is the draw guard (no forced commit when *behind* — a draw beats a loss). |
| Fog hides mountains | `free_land` counts only visible walls, over-estimating early; harmless (delays the land-exhausted trigger). |
| No AFK kick | Fallback shuffle kept for tempo. |

## 7. Diversity check

vs `garrison` (standing defensive reserves): aegis sizes its garrison
against a live wave estimate, parks it mobile on the keep, and actively
intercepts/hunts instead of holding. vs `choke_control` (corridor hold):
aegis holds a *ring*, not a corridor, and its win condition is the
counter window, not attrition. vs `late_rush`: aegis's commit is
event-driven (spent attack), not clock-driven.

## 8. Experiment hypothesis

**Hypothesis:** measured defence + event-driven counters beat aggressive
bots (their attack dies on the ring, the counter takes the empty general)
and the econ timeout + castle compounding avoids losing on points to
peaceful ones. Verification: beat `smoke` (capture, 528, 3 castles) and
`expand_plus` (capture, 357, 3 castles); lost to `blitz` (290) — the
source's known rush-vs-turtle weakness, carried over, to be measured on
the full grid.

Seed grid: opponents `smoke`, `garrison`, `late_rush`, `blitz`; seeds
0–2; `--mode competition`. Metrics: W-L-D, counter conversions (games won
inside a counter window), castles built, faults.

