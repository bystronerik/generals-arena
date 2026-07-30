# Strategy spec — castle_rush

Bot target: `bots/castle_rush/` (not yet implemented). Fork from
`bots/smoke/` + `bots/expand_plus/` per the task brief. This file is the
strategy contract only; no code lives here.

Grounded idea: **more aggressive castle building than `castle_builder`;
spend early for income.** Baselines for comparison: `castle_builder`,
`expand_plus`, `smoke`.

Rule references: [`RULES.md`](../../RULES.md) sections 03 (build castles),
04 (army growth); build-cost detail in
[`docs/competition/build-castles.md`](../../competition/build-castles.md).

## 1. Phase boundaries

`castle_rush` does **not** use explicit phase switches (that is the
`phase_switch` bot's job). It runs one continuous policy with internal
gates. The "rush" is in the build parameters, not in a phase clock.

| Stage | Trigger | Behavior |
| --- | --- | --- |
| Setup | `turn < RUSH_START` (default 10) | Pure `expand_plus` expansion. No resting, no building. Just enough land to make the first castle defensible. |
| Rush | `turn >= RUSH_START` and `own_castles < RUSH_CAP` | Aggressive build: rest the general early and often, build at the cheapest reachable cell as soon as affordable, short cooldown, higher cap. |
| Sustain | `own_castles >= RUSH_CAP` | Stop resting; return to full `expand_plus` expansion. Existing castles keep producing. |

Tunables (defaults, to be confirmed by experiment):

- `RUSH_START = 10` — vs `castle_builder`'s `MIN_TURN_TO_BUILD = 20`. The rush
  starts as soon as the general has had ~5 army-growth ticks and a little
  land. Earlier risks building on a cell that gets raided; later loses income
  ticks.
- `RUSH_CAP = 4` — vs `castle_builder`'s `MAX_OWN_CASTLES = 2`. Each castle
  after the first raises the price of the next (surcharges stack), so 4 is
  the aggressive ceiling. Beyond 4 the price typically exceeds what the
  rested general can bank in the cooldown window.
- `RUSH_COOLDOWN = 25` — vs `castle_builder`'s `BUILD_COOLDOWN_TURNS = 40`.
  Shorter cooldown = more castles per game = more income, at the cost of
  resting the general more often (less expansion).
- `RUSH_MIN_LAND = 5` — vs `castle_builder`'s `MIN_LAND_TO_BUILD = 8`. Lower
  threshold so the first build can fire earlier.
- `RUSH_SURPLUS_MARGIN = 5` — vs `castle_builder`'s `BUILD_SURPLUS_MARGIN = 15`.
  Lower margin = build with a thinner surplus = earlier castle = earlier
  income, accepting a near-zero garrison on the new castle.

## 2. Castle cost / build policy (vs `castle_builder`)

Same build action and cost model (see
[`docs/competition/build-castles.md`](../../competition/build-castles.md)):

```
cost(cell) = 35 + sum_over_own_structures(max(0, 14 - 2 * manhattan(cell, s)))
```

Only own general + own castles count; enemy structures never affect the
price. Build action is `2 r c 0 0`, replaces the move, resolves before moves,
invalid build is a silent pass. Remainder army stays on the new castle (can
be 0).

Differences from `castle_builder`:

| Parameter | `castle_builder` | `castle_rush` (proposed) | Effect |
| --- | --- | --- | --- |
| Build window start | turn 20 | turn 10 (`RUSH_START`) | First castle ~10 ticks earlier → ~5 extra army-growth ticks of income per castle. |
| Build window end | turn 900 | none (no `MAX_TURN_TO_BUILD`) | Castles can be built all game; income compounds longer. |
| Min land to build | 8 | 5 (`RUSH_MIN_LAND`) | First build fires with less territory; riskier but earlier. |
| Max own castles | 2 | 4 (`RUSH_CAP`) | Up to 2 extra income sources. Each raises the next price; the cap exists because price outpaces banking past ~4. |
| Build cooldown | 40 turns | 25 turns (`RUSH_COOLDOWN`) | More builds per game; the general rests more often. |
| Build surplus margin | 15 | 5 (`RUSH_SURPLUS_MARGIN`) | Build with a thinner surplus; the new castle starts with less garrison. |
| Resting general | only if `own_castles < 2` and `turn in [20, 900]` and `land >= 8` | only if `own_castles < RUSH_CAP` (no turn window, lower land threshold) | Resting is the engine of building; resting more = building more. |

Net: every knob is shifted toward "build earlier, build more, accept a
thinner garrison". The bet is that income from 3–4 early castles outpaces the
land/expansion lost to resting the general.

## 3. Per-stage behavior

### Setup (turn < 10)

`expand_plus` expansion (greedy capture + BFS march fallback). No building,
no resting. The enemy-general sighting recorder runs every tick (cheap; see
§5) so a sighting during setup is not lost.

### Rush (turn >= 10 and own_castles < 4)

1. **Sighting step** (every tick): scan for an opponent-owned general; store
   its `(r, c)` permanently.
2. **Build attempt**: if `turn - last_build_turn >= RUSH_COOLDOWN`, scan all
   owned plain cells for the one with the largest `army - cost(cell)` surplus.
   If the best surplus is `>= RUSH_SURPLUS_MARGIN` (5), emit `2 r c 0 0`,
   record `last_build_turn`.
3. **Rest decision**: if `own_castles < RUSH_CAP` and
   `my_land >= RUSH_MIN_LAND`, mark the general as resting this tick.
4. **Relocate**: if resting and the general's stack can afford the cheapest
   reachable neighbor cell (`cost + RUSH_SURPLUS_MARGIN`), move the whole
   stack one hop onto that neighbor. The build fires next tick from there.
5. **Expand**: otherwise, run `expand_plus` expansion, excluding the general
   cell from the source list when it is resting.

### Sustain (own_castles >= 4)

`expand_plus` expansion with no resting and no building. Existing castles
keep producing (every other turn, like the general). The bot does not
deliberately defend its castles; it relies on expansion keeping the frontier
away from them.

## 4. Scoring summary

| Decision | Score / rule |
| --- | --- |
| Capture | `army * 10 * (2 if opp else 1)`, max over all owned-source × direction pairs (same as `expand_plus`) |
| March fallback | largest owned stack, step to neighbor with strictly smaller BFS distance to nearest capturable tile (same as `expand_plus`) |
| Build | `army - cost(cell)`, max over owned plain cells; fire if `>= 5` (vs `castle_builder`'s 15) |
| Relocate | neighbor with `min cost(cell)`; fire if `army - 1 >= cost + 5` (vs `castle_builder`'s 15) |

No deathtouch hunt. `castle_rush` is an economy bot, not a hunter; if the
enemy general is sighted, the bot does not switch behavior. (Adding a hunt
would make this a second `phase_switch`; the brief keeps them distinct.)

## 5. Pseudocode

```
Agent.act(obs):
    locate_general(obs)                      # once
    update_enemy_general_sighting(obs)       # every tick (kept for parity; not acted on)

    if own_castles(obs) < RUSH_CAP and obs.turn >= RUSH_START:
        structs, n_castles = own_structures(obs)
        b = maybe_build(obs, structs, n_castles)
        if b is not None:
            last_build_turn = obs.turn
            return b
        resting = (n_castles < RUSH_CAP and obs.my_land >= RUSH_MIN_LAND)
        if resting:
            re = maybe_relocate_general(obs, structs)
            if re is not None: return re
        return expand(obs, exclude_cell = general_pos if resting else None)

    return expand(obs)                       # setup and sustain


maybe_build(obs, structs, n_castles):
    if n_castles >= RUSH_CAP: return None
    if obs.turn - last_build_turn < RUSH_COOLDOWN: return None
    best_cell, best_surplus = None, -inf
    for (r, c) in owned_plain_cells(obs):
        surplus = army[r][c] - build_cost((r, c), structs)
        if surplus > best_surplus:
            best_cell, best_surplus = (r, c), surplus
    if best_cell is not None and best_surplus >= RUSH_SURPLUS_MARGIN:
        return (2, best_cell.r, best_cell.c, 0, 0)
    return None


maybe_relocate_general(obs, structs):
    gr, gc = general_pos
    if army[gr][gc] <= 1: return None
    best_dir, best_cost = None, None
    for d, (dr, dc) in DIRECTIONS:
        nr, nc = gr+dr, gc+dc
        if not in_bounds: continue
        if not passable(type[nr][nc]): continue
        if owner[nr][nc] == 2: continue        # do not fight to stage
        if type[nr][nc] in {3, 4}: continue    # not buildable
        c = build_cost((nr, nc), structs)
        if best_cost is None or c < best_cost:
            best_dir, best_cost = d, c
    if best_dir is None: return None
    if army[gr][gc] - 1 < best_cost + RUSH_SURPLUS_MARGIN: return None
    return (0, gr, gc, best_dir, 0)
```

`expand` is the `expand_plus` implementation (greedy capture + BFS march
fallback + any-legal-move last resort), with an optional `exclude_cell`
parameter to skip the resting general. See
[`bots/expand_plus/agent.py`](../../../bots/expand_plus/agent.py) for the
reference logic.

## 6. Edge cases

- **First build fires at turn ~10 with ~5 land.** The new castle sits near the
  general with a thin garrison (surplus margin 5). If the opponent raids that
  cell before it produces ~30 army, the investment is lost. This is the
  core risk of the rush; the bet is that the current opponent pool does not
  raid deep.
- **Surcharge stacking past castle 2.** Castle 3 pays the base 35 plus
  surcharges from the general and 2 castles; castle 4 adds a third castle's
  surcharge. With `RUSH_SURPLUS_MARGIN = 5`, the rested general must bank
  `cost + 5` before each build. If it cannot within `RUSH_COOLDOWN`, the build
  slips a cooldown — acceptable, the bot just builds fewer than 4.
- **General cell is the only owned cell with army.** In rush, if resting is
  on and no other cell can capture, the bot passes. Same as `castle_builder`.
- **All neighbors of the general are mountains / enemy / structures.**
  `maybe_relocate_general` returns `None`; the general keeps resting. The bot
  does not fight to stage a build.
- **Build target captured between relocate and build.** Next tick's
  `maybe_build` re-scans; the lost cell is not a candidate. No special
  handling.
- **Enemy general sighted.** Stored but not acted on. `castle_rush` is an
  economy bot; adding a hunt would duplicate `phase_switch`. The sighting is
  recorded only for parity with the sibling bots and to leave room for a
  future hunt without re-scanning.
- **Opponent is `general_hunter` and reaches turn 800 with a sighting.**
  `castle_rush` has no deathtouch defense. Its castles produce army but it
  does not chase or counter-chase. Expected to lose to `general_hunter` in
  any game where `general_hunter` sights its general — a known trade of the
  economy-rush design.
- **Map with the general boxed in by mountains.** Relocate fails; the bot
  rests indefinitely and never builds. The expand loop still runs from other
  cells, so the bot does not stall — it just gets 0 castles that game.
- **Two builds on the same tick.** Not possible; one action per tick.
  Relocate and build are sequential across ticks by design.

## 7. Experiment hypothesis

**Hypothesis:** Building castles earlier (turn 10 vs 20), more often
(cooldown 25 vs 40), and more of them (cap 4 vs 2), with a thinner surplus
margin (5 vs 15), yields higher late-game income than `castle_builder` — at
the cost of slower early expansion from resting the general more. Net
claim: over the same seed grid, `castle_rush` builds >= 3 castles per game
(vs `castle_builder`'s 2) and finishes at least as many games cleanly (no
fault regression), with winrate >= `castle_builder` against the
non-raiding opponent pool.

**One change vs `castle_builder`:** the build parameters
(`RUSH_START`, `RUSH_CAP`, `RUSH_COOLDOWN`, `RUSH_MIN_LAND`,
`RUSH_SURPLUS_MARGIN`). The build, relocate, and expand sub-procedures are
reused unchanged from `castle_builder` / `expand_plus`.

**Seed grid** (per [`experiment-protocol.md`](../experiment-protocol.md)):

- Opponents: `smoke`, `expand_plus`, `castle_builder`, `general_hunter`.
- Seeds: 0, 1, 2.
- Mode: `--mode competition` only.
- Store under `data/games/` before any rating update.

**Metrics:** W-L-D per matchup, mean turns, castles built per game (primary
signal — must be >= 3 on average), fault count. Expected outcome mirrors
experiments `001`–`003`: most games draw at 1200 turns against the
non-raiding pool; the positive evidence is "more castles built, no fault
regression".

**Failure mode that would force a revert:**

- `castle_rush` builds **fewer** castles per game than `castle_builder`
  (would mean the aggressive parameters backfire — the general cannot bank
  enough within the shorter cooldown, or the lower land threshold fires
  before the general can afford anything).
- `castle_rush` faults / crashes / fails to terminate more often than
  `castle_builder` on the same seeds.
- `castle_rush` loses more games than it wins against `general_hunter` by a
  margin worse than `castle_builder`'s, indicating the thinner garrisons are
  handing the hunter free runner stacks.

**Telemetry gap:** same as `002` / the `phase_switch` spec — the current
game-record schema stores winner/turns/terminated/truncated, not final
land/army. The income claim ("higher late-game income") cannot be measured
from stored games alone; it needs a schema follow-up (final land/army at
truncation, or per-tick army). The castle-count and winrate claims can be
measured today from the matchup logs and the stored record.
