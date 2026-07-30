# Strategy spec — phase_switch

Bot target: `bots/phase_switch/` (not yet implemented). Fork from
`bots/smoke/` + `bots/expand_plus/` per the task brief. This file is the
strategy contract only; no code lives here.

Grounded idea: **explicit early / mid / late phases** over the 1200-turn
competition game. Baselines for comparison: `expand_plus`, `castle_builder`,
`general_hunter`, `smoke`.

Rule references: [`RULES.md`](../../RULES.md) sections 03 (build castles),
04 (army growth), 07 (deathtouch from turn 800); build-cost detail in
[`docs/competition/build-castles.md`](../../competition/build-castles.md).

## 1. Phase boundaries

| Phase | Turn window | Name | Primary objective |
| --- | --- | --- | --- |
| Early | `0 <= turn < EARLY_END` | expand | Grow land as fast as possible; no castles, no resting. |
| Mid | `EARLY_END <= turn < LATE_START` | economy + castles | Convert land into income-producing castles while still expanding. |
| Late | `turn >= LATE_START` | hunt / deathtouch | If the enemy general is known, beeline a runner and execute on it. |

Tunables (defaults, to be confirmed by experiment):

- `EARLY_END = 80` — early phase ends once the general has had ~40 army-growth
  ticks (army generates every other turn) and the land bonus has fired once
  or twice. Earlier than 80 risks under-expanding; later delays castle income
  past the point where it amortizes within the 1200-turn cap.
- `LATE_START = 800` — fixed by the deathtouch rule (`DEATHTOUCH_TURN = 800`,
  see [`docs/competition/deathtouch.md`](../../competition/deathtouch.md)).
  Not a tunable.

Phase membership is a hard `if/elif` on `obs.turn`, evaluated every tick in
`Agent.act`. No hysteresis, no soft blend — the spec is "explicit phases", not
a weighted policy.

## 2. Castle cost / build policy (vs `castle_builder`)

Same build action and cost model as `castle_builder` (see
[`docs/bots/castle-builder.md`](../../bots/castle-builder.md) and
experiment [`002-castle-builder-early-investment.md`](../experiments/002-castle-builder-early-investment.md)):

```
cost(cell) = 35 + sum_over_own_structures(max(0, 14 - 2 * manhattan(cell, s)))
```

Only own general + own castles count; enemy structures never affect the
price. Build action is `2 r c 0 0` and replaces the move that turn. Builds
resolve before moves; invalid builds are a silent pass.

Differences from `castle_builder`:

| Parameter | `castle_builder` | `phase_switch` (proposed) | Rationale |
| --- | --- | --- | --- |
| Build window start | turn 20 | `EARLY_END` (80) | phase_switch does not build during the early expand phase at all; the mid phase owns all building. |
| Build window end | turn 900 | `LATE_START` (800) | Once deathtouch is live, the hunt phase owns the turn; building is paused to free every owned stack for the beeline. |
| Min land to build | 8 | 8 | Same — building with less land risks losing the cell to a raid before it pays back. |
| Max own castles | 2 | 2 | Same cap. Each new castle raises the price of the next (surcharges stack), and the mid window is only ~720 turns; a third castle rarely amortizes. |
| Build cooldown | 40 turns | 40 turns | Same — keeps the general from being permanently rested. |
| Resting general | from turn 20 if land >= 8 | only during mid phase | phase_switch never rests in early (land growth priority) and never rests in late (hunt priority). |

Net: the build mechanism is **identical to `castle_builder`**; the only
change is *when it is allowed to fire*. Confining builds to the mid window is
the testable claim — does gating builds by phase change income or winrate
versus `castle_builder`'s turn-20-to-900 window?

## 3. Per-phase behavior

### Early (turn < 80)

Pure `expand_plus` behavior:

1. Greedy capture: over all owned cells with `army > 1`, over all 4
   directions, score each adjacent capturable tile as
   `army * 10 * (2 if opponent else 1)`. Pick the highest score. A tile is
   capturable if `owner == 2` (opponent) or `owner == 0 and type not in {0,5}`
   (visible neutral).
2. Fallback when no capture is available: multi-source BFS seeded from every
   visible capturable tile, then move the largest owned stack one step along
   the gradient (decreasing BFS distance). This is the `expand_plus` march.
3. Last resort: any legal move; else `PASS = (1, 0, 0, 0, 0)`.

No castle building. No general resting. No enemy-general tracking beyond the
always-on sighting recorder (cheap; see §5).

### Mid (80 <= turn < 800)

`castle_builder` behavior, with the early-phase expansion as the fallback:

1. **Sighting step** (every tick, all phases): scan the visible grid for an
   opponent-owned general (`owner == 2`, `type == 4`); if found, store its
   `(r, c)` in `self.enemy_general` permanently. Generals never move, so one
   sighting is exact for the rest of the game.
2. **Build attempt**: if `own_castles < MAX_OWN_CASTLES` and
   `turn - last_build_turn >= BUILD_COOLDOWN_TURNS`, scan all owned plain
   cells (`type == 1`, `owner == 1`) for the one with the largest
   `army - cost(cell)` surplus. If the best surplus is `>= BUILD_SURPLUS_MARGIN`
   (15), emit `2 r c 0 0` and record `last_build_turn`.
3. **Rest decision**: if `own_castles < MAX_OWN_CASTLES` and
   `my_land >= MIN_LAND_TO_BUILD`, mark the general as resting this tick.
4. **Relocate**: if resting and the general's stack can afford the cheapest
   reachable neighbor cell (`cost + BUILD_SURPLUS_MARGIN`), move the whole
   stack one hop onto that neighbor (`0 gr gc dir 0`). The build fires next
   tick from the new cell.
5. **Expand**: otherwise, run the early-phase expansion, but exclude the
   general's cell from the source list when it is resting (so the resting
   stack is not spent on a capture).

### Late (turn >= 800)

`general_hunter` behavior, gated on a known enemy general:

1. If `self.enemy_general` is set, run the hunt:
   - **Immediate execute**: for each direction `d`, the cell *behind* the
     target relative to `d` (i.e. `(tr - dr, tc - dc)`) is an owned neighbor
     with `army >= 2`. If so, emit `0 r c d 0` — under deathtouch, one unit
     on the general wins regardless of the garrison.
   - **Beeline**: otherwise BFS from the target over passable tiles, then
     advance the owned cell with the smallest BFS distance one step along a
     decreasing-distance edge. Prefer the closest runner so the stack
     arrives fastest.
2. If `self.enemy_general` is **not** set (never sighted), fall through to
   the mid-phase loop. The bot does not chase into fog blind — a blind
   beeline has no target and would just wander.

## 4. Scoring summary

| Decision | Score / rule |
| --- | --- |
| Early capture | `army * 10 * (2 if opp else 1)`, max over all owned-source × direction pairs |
| Early fallback (march) | largest owned stack, step to neighbor with strictly smaller BFS distance to nearest capturable tile |
| Mid build | `army - cost(cell)`, max over owned plain cells; fire if `>= 15` |
| Mid relocate | neighbor with `min cost(cell)`; fire if `army - 1 >= cost + 15` |
| Late execute | any owned neighbor of the enemy general with `army >= 2` |
| Late beeline | owned cell with `min BFS distance` to target, step to a neighbor with strictly smaller distance |

## 5. Pseudocode

```
Agent.act(obs):
    locate_general(obs)                      # once
    update_enemy_general_sighting(obs)       # every tick, all phases

    if obs.turn >= LATE_START and enemy_general is not None:
        m = hunt(obs)
        if m is not None: return m

    if obs.turn >= EARLY_END:
        structs, n_castles = own_structures(obs)
        b = maybe_build(obs, structs, n_castles)
        if b is not None:
            last_build_turn = obs.turn
            return b
        resting = (n_castles < MAX_OWN_CASTLES and obs.my_land >= MIN_LAND_TO_BUILD)
        if resting:
            re = maybe_relocate_general(obs, structs)
            if re is not None: return re
        return expand(obs, exclude_cell = general_pos if resting else None)

    return expand(obs)                       # early phase


hunt(obs):
    tr, tc = enemy_general
    # immediate execute
    for d, (dr, dc) in DIRECTIONS:
        r, c = tr - dr, tc - dc
        if in_bounds(r, c) and owner[r][c] == 1 and army[r][c] >= 2:
            return (0, r, c, d, 0)
    # beeline via BFS from target
    dist = bfs_from(tr, tc, passable=type not in {2,5})
    best = None
    for each owned (r, c) with army >= 2 and dist[r][c] > 0:
        for d, (dr, dc) in DIRECTIONS:
            nr, nc = r+dr, c+dc
            if in_bounds and passable and owner[nr][nc] == 1 \
               and dist[nr][nc] >= 0 and dist[nr][nc] < dist[r][c]:
                if best is None or dist[r][c] < best.dist:
                    best = (0, r, c, d, 0)
    return best
```

`expand`, `maybe_build`, `maybe_relocate_general`, `own_structures`,
`build_cost` are the `castle_builder` / `expand_plus` implementations verbatim
(see [`bots/castle_builder/agent.py`](../../../bots/castle_builder/agent.py)
and [`bots/expand_plus/agent.py`](../../../bots/expand_plus/agent.py) for the
reference logic).

## 6. Edge cases

- **Enemy general sighted in early phase.** Store it; do not act on it until
  `turn >= LATE_START`. Acting early wastes expansion and the deathtouch rule
  is not active before turn 800 anyway.
- **Enemy general sighted, then fogged.** Keep the stored cell. Generals
  never move; the sighting stays exact. Late-phase hunt works off the
  remembered cell, not the live view.
- **Enemy general never sighted by turn 800.** Late phase falls through to
  the mid loop. The bot keeps building/expanding; it does not invent a target.
- **General cell is the only owned cell with army.** In mid, if resting is on
  and no other cell can capture, the bot passes (no relocate, no expand
  source). Acceptable — resting is the point.
- **Build target cell gets captured between the relocate and the build.**
  The next tick's `maybe_build` re-scans; the lost cell is simply not a
  candidate. No special handling.
- **All neighbors of the general are mountains / enemy / structures.**
  `maybe_relocate_general` returns `None`; the general keeps resting. The bot
  does not fight to stage a build.
- **Two bot builds on the same tick.** Not possible — one action per tick.
  Relocate and build are sequential across ticks by design.
- **Deathtouch draw.** Both players touching the enemy general on the same
  tick is a draw (`winner = -1`). The bot does not defend against this; it
  only attacks.
- **Chase defense.** If the opponent is also beelining and chases the bot's
  runner source, the touch may not execute. The spec does not include
  chase-aware routing; that is a follow-up if the hunt fires often enough to
  measure.

## 7. Experiment hypothesis

**Hypothesis:** Gating `castle_builder`'s build mechanism to a mid-phase
window (turn 80–800), with pure expansion before and a deathtouch hunt after,
yields at least as much late-game income as `castle_builder` (which builds
from turn 20–900) and converts any enemy-general sighting from turn 800 into
a win that `castle_builder` cannot get. Net claim: winrate >= `castle_builder`
over the same seed grid, with no regression in fault rate or match length.

**One change vs `castle_builder`:** the phase gates around the build
mechanism. The build, relocate, expand, and hunt sub-procedures are reused
unchanged from `castle_builder` / `expand_plus` / `general_hunter`.

**Seed grid** (per [`experiment-protocol.md`](../experiment-protocol.md)):

- Opponents: `smoke`, `expand_plus`, `castle_builder`, `general_hunter`.
- Seeds: 0, 1, 2.
- Mode: `--mode competition` only.
- Store under `data/games/` before any rating update.

**Metrics:** W-L-D per matchup, mean turns, castles built per game, hunt
activations (count of games where the late phase fired a beeline/execute).
Expected outcome mirrors experiments `001`–`003`: most games draw at 1200
turns because none of the opponents scout deep enough to expose a general;
the positive evidence is "no regression + the hunt mechanism is wired and
unit-checkable", same framing as `003-general-hunter-deathtouch-beeline.md`.

**Failure mode that would force a revert:** any matchup where phase_switch
finishes fewer games cleanly (more faults / crashes / non-termination) than
`castle_builder` on the same seeds, or where it builds fewer castles per game
than `castle_builder` (which would mean the phase gate broke the build
mechanism rather than only re-windowing it).

**Telemetry gap:** the current game-record schema stores
winner/turns/terminated/truncated, not final land/army or per-tick army. The
income claim cannot be measured from stored games alone; proving it needs a
schema follow-up (see `002-castle-builder-early-investment.md`'s open note).
The winrate / fault / castle-count claims can be measured today.

## Parameter revision 1

Source: [`round1.md`](../measurements/round1.md) + [`round1.json`](../measurements/round1.json). 14 games: 1 W, 3 L, 10 D. Winrate 7.1%, Elo 1473.7, mean turns 1063.5.

### What the data says

- **Castles built: 2 per game (every game).** The `MAX_OWN_CASTLES = 2` cap is the binding constraint. `castle_builder` builds 3 and `castle_rush` builds 4 in the same grid; `phase_switch` is the weakest economy in the economy cluster.
- **The one win fired from the late phase.** `phase_switch` beat `garrison` at turn 1089 — past `LATE_START = 800`. The deathtouch hunt mechanism works when it gets a sighting and reaches turn 800. This is positive evidence for the phase gate; do not weaken it.
- **The three losses are all mid-phase, to aggressive scouts.** `fog_scout` won at turn 442, `army_convey` at 549, `late_rush` at 809. All three close before the late hunt can fire. `phase_switch` is economically behind in mid-phase because it builds only 2 castles while the opponents out-scout or out-economy it.
- **Economy-cluster games all draw at 1200.** vs `castle_builder` (×2), vs `castle_rush` (×4). The phase gate does not break the build mechanism (no faults), but the 2-castle cap leaves `phase_switch` unable to press an economy advantage.

### Tweaks (preserve gated early/mid/late phases)

| Parameter | Was | Now | Rationale |
| --- | --- | --- | --- |
| `MAX_OWN_CASTLES` | 2 | 3 | The mid window (turn 80–800 = 720 ticks) with a 40-turn cooldown fits a third castle. Matches `castle_builder`'s economy while keeping the phase gate as the defining feature. The 2-castle cap was the conservative default; the data shows the mid window has unused capacity. |
| `EARLY_END` | 80 | 60 | Start mid-phase economy 20 ticks earlier so the third castle amortizes within the window. Still a hard phase gate; still pure expand before turn 60. The 20-tick shift gives ~10 extra army-growth ticks of castle income by mid-game, addressing the economic deficit behind the aggressive scouts. |

Unchanged: `LATE_START = 800` (rule-fixed), `BUILD_COOLDOWN = 40`, `BUILD_SURPLUS_MARGIN = 15`, `MIN_LAND_TO_BUILD = 8`. The hunt, build, relocate, and expand sub-procedures stay verbatim from `castle_builder` / `expand_plus` / `general_hunter`.

### What this tests

Does a third castle in the mid window convert any of the 3 mid-phase losses into draws or wins, without breaking the clean-draw record in the economy cluster? The hypothesis is unchanged — the phase gate is still the one change vs `castle_builder`; the cap raise only lets the existing mid-phase build mechanism fire one more time.
