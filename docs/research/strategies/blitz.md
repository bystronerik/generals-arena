# Strategy spec — blitz

Bot: `bots/blitz/`. Migrated from the generals-bot repo (live generals.io
ruleset) as a grid-native rewrite. Grounded idea: **general rush — one big
strike stack, launched early and repeatedly at the enemy general.**
Baselines for comparison: `smoke`, `expand_plus`, `army_convey`, `late_rush`.

Rule references: [`RULES.md`](../../../RULES.md) sections 04 (army growth), 07
(deathtouch). The turn cadence matches the source ruleset (production every
other turn, land bonus every 50), so the source-tuned thresholds transfer
as **hypotheses** — the numbers below were measured under the old ruleset
and must be re-measured here.

## 1. Priority ladder (every turn)

| Priority | Trigger | Behavior |
| --- | --- | --- |
| Finish | enemy general ever sighted and beatable now | Adjacent capture; from turn 800 (deathtouch) any 2-army neighbour takes it. |
| Defend | `home_threat` deficit > 0 | Kill the threat if reachable, else pull army home — preferring army that arrives before the threat does. From turn 780, `chase_defence` guards the general. |
| Opening | `turn < OPENING_END` (50) | Chain expansion biased toward the enemy anchor. |
| Assault | after opening | Wave cycle: rebuild → rally → push at the strike target. |
| Expand | nothing else | `expansion_step` — a whiffed rush must not also be idle. |

## 2. Opening — chain expansion

A move leaves one army behind, so walking a stack of `A` outward captures
`A-1` cells in `A-1` turns with zero waste. The launch rule is
`2*(army-1) >= turns_left`: commit once the affordable chain is long enough
to spend the production remaining before the turn-50 land bonus. Chains are
biased toward the enemy anchor so they double as the strike corridor (each
owned cell crossed later adds one army to the stack). A boxed-in chain head
may detour up to `CHAIN_DETOUR = 3` steps over owned land.

## 3. Assault — the wave cycle

- **Stack latch** (`ensure_stack`): sticky largest-stack tracking; jumps
  only when the stack died or another cell grew `RESTACK_RATIO = 1.5×`
  bigger (the signal that the wave is over).
- **Sizing** (`required_army`): `max(18, 0.7 × opponent_mobile) + 1/hop`.
  `opponent_mobile = opp_army − opp_land` — the aggregates are exact every
  turn, so this is a real read on their defence.
- **Launch**: reaching size launches early; the rally deadline
  (`RALLY_TICKS = 14`) launches undersized waves anyway — the turn 60–160
  window does not wait. The deadline must not scale with the opponent, or
  the wave never launches.
- **Pathing**: Dijkstra potential field, not BFS — own cells discounted
  (they feed the stack), defended enemy cells surcharged
  (`0.03/army`), neutral 1.15. **Arena change:** the source's
  `city_cost = 40` wall is gone — there are no neutral cities; an enemy
  castle is just a valuable enemy cell.
- **Rebuild**: between waves, chain-expand until the next land-bonus
  boundary (`((turn // 50) + 1) * 50`), then rally with a collection walk
  (move the stack forward through own surplus rather than ferrying army
  back).

## 4. Targeting

Seen general (latched by `BeliefState`) always wins. Before contact: fog
nearest the mirror-side anchor (not the fog farthest from us — that is a
corner). After contact: fog within `CONTACT_RADIUS = 6` of visible enemy
cells, farthest from our territory (their visible cells are their newest
land; the general is behind them). Hysteresis: a guess holds for
`RETARGET_INTERVAL = 20` turns, until scouted, or until stood upon.

## 5. Defence — the one thing that outranks the rush

Blitz empties its home by design, so the trigger is a real threat model:
biggest enemy stack within `DEFENSE_DIST = 12` of the general vs what we
can rally in time. Deliberately pessimistic — **one** reinforcing stack
counts, not the sum of everything in range (one move per turn means
scattered army cannot converge the way a naive sum implies). The general
regenerates `arrival // 2` while the threat walks (cadence identical to the
source ruleset).

## 6. Arena-rule adaptations (vs the generals-bot source)

| Rule | Adaptation |
| --- | --- |
| No neutral cities | `city_cost` path surcharge deleted; chain/expansion "avoid neutral city" guards moot. |
| Deathtouch (turn 800) | New: finish fires with any 2-army neighbour; `chase_defence` from turn 780. |
| Draw at 1200 | Rush identity is the anti-draw mechanism; no separate force-commit needed. |
| Build action | Unused. Blitz spends every army on tempo; building is a different bot's identity. |
| 150 ms/move | One Dijkstra + 2–3 BFS per turn at ≤441 cells — comfortably inside budget. |

## 7. Diversity check

vs `late_rush` (timed all-in at a fixed turn): blitz launches waves from
turn 50 onward, re-sizes each wave against the opponent's mobile army, and
rebuilds its economy between waves. vs `army_convey` (continuous frontier
funnel): blitz concentrates one stack and drives it at a *general* target
with a potential field, not at the frontier. vs `general_hunter`
(deathtouch beeline): blitz aims to win in the 60–300 window, long before
deathtouch matters.

## 8. Experiment hypothesis

**Hypothesis:** an early, repeatedly re-sized strike wave kills
expansion-oriented bots before their economy compounds, and beats
castle-economy bots by killing them before castle income pays back.
Source-ruleset numbers (90% winrate vs SimpleBot, mean kill ~turn 265) do
not transfer; the arena claim is: blitz ends most games against the
baseline pool by capture well before the 1200-turn draw, and never
draw-stalls against economy bots.

Seed grid per [`experiment-protocol.md`](../experiment-protocol.md):
opponents `smoke`, `expand_plus`, `army_convey`, `late_rush`; seeds 0–2;
`--mode competition` only. Metrics: W-L-D, mean turns to capture (primary
signal — target < 400), strikes per game from telemetry, faults.

