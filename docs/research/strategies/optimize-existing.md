# Strategy analysis — optimizing the existing four bots

Scope: `bots/smoke/`, `bots/expand_plus/`, `bots/castle_builder/`,
`bots/general_hunter/`. This file is analysis and an implementation contract.
No code lives here. Implementation belongs to the bot-author agents.

Rule references: [`RULES.md`](../../../RULES.md) sections 02 (move order),
03 (build castles), 04 (army growth), 05 (combat), 06 (visibility),
07 (game end / deathtouch).

Baseline evidence: all 31 stored games in `data/games/` are draws at the
1200-turn cap. Elo is stuck at 1500 for every bot. The sections below
explain why that happens and give the concrete rules that break it.

---

## 1. Root cause of the all-draw season

Five defects, all present in the current code. D1 and D2 are the ones that
matter.

### D1 — Every bot spends its own general's army, so every general is a 1-army tile

`_expand` in all four bots iterates *every* owned cell with `army > 1` as a
move source. The general is an owned cell. Greedy expansion therefore drains
the general on almost every tick it has 2+ army.

Growth arithmetic (`global_update` in `competition-module/generals/core/game.py`):

| Source | Rate |
| --- | --- |
| General or castle | +1 army on every even tick = **0.5 army/turn** |
| Every owned cell | +1 army when `turn % 50 == 0` = **0.02 army/cell/turn** |

A general that is never spent holds `1 + 0.5 * turn` army — about 400 by turn
800. A general that is spent whenever it exceeds 1 holds **1 or 2 army for the
whole game**.

Consequence: every current bot can be killed at *any* turn by one enemy stack
of 3 army that reaches its general tile. Deathtouch is not even needed. The
pool is mutually one-shot-able and simply never arrives.

**This is the single highest-value fix and the cheapest.**

### D2 — Sighting the enemy general requires standing next to it, so the hunt is dead code

Vision is a Chebyshev radius of 1 around owned cells (`Observation`
docstring in `competition-module/generals/core/observation.py`: "players can
only see cells within a 3x3 radius of cells they own"). Movement is
4-connected; vision is 8-connected.

`general_hunter._update_enemy_general_sighting` only records a general that is
*currently visible*. To make that fire, the bot must already own a cell within
the 3×3 box around the enemy general. Greedy expansion stops long before
that (see D3). So `self.enemy_general` stays `None`, `_hunt` never runs, and
`general_hunter` is behaviourally identical to `expand_plus` minus the march.

Experiment note `003` records the hunt as "wired but unmeasured". It is worse
than unmeasured: under the current expansion policy it is unreachable.

### D3 — The capture gate stops expansion at the first defended tile

Every bot requires `src_army > dest_army + 1` on a *single* source cell. No
bot ever pools army. Once the two territories meet, each frontier tile is
held by a stack the neighbouring single stack cannot beat, so both sides stop
advancing and the game is decided by the turn cap.

### D4 — The `first_valid` fallback wastes turns and leaks the general

`smoke`, `castle_builder` and `general_hunter` fall back to `first_valid`,
which is the first owned cell in row-major order with `army > 1` and a
passable neighbour. That is frequently the general, and the destination is
arbitrary. Hundreds of late-game ticks are spent shuffling army into already
owned cells in the top-left of the map.

`expand_plus` is better (BFS march first) but still ends with the same
fallback.

### D5 — No bot ever uses `split=1`

Action field 5 chooses "all but one" (`split=0`) or "half" (`split=1`). With
only `split=0` available, a bot cannot detach a scout without stripping the
source cell. Every reserve rule below needs `split=1`.

---

## 2. Mechanics that produce decisive games

These follow from `RULES.md` and the modifier source. They are the levers.

### L1 — Deathtouch makes the endgame pure geometry

From turn 800 any move that *executes* onto the enemy general's tile wins,
whatever the garrison is (`competition-module/generals/modifiers/deathtouch.py`).
The endgame reduces to two facts: **do you know the cell**, and **do you have
any 2-army stack adjacent to it at a tick ≥ 800**. Army size is irrelevant.

### L2 — Before turn 800 a spent general dies to 3 army

See D1. `army > dest_army + 1` with `dest_army = 1` means a source of 3 army
captures the general outright and wins instantly. Against the current pool
this works from roughly turn 20 onward.

### L3 — Income: one castle ≡ 25 land cells, but castle income is *usable*

0.5 army/turn per structure against 0.02 army/cell/turn per land cell. One
castle equals 25 land cells of income. Land is usually easier to acquire than
35–47 army, so **land beats castles on raw income**.

The real argument for a castle is *concentration*: a castle's income appears
as one growing stack you can march, while the land bonus appears as +1 on
sixty scattered cells that cost sixty convey moves to pool. Castles buy
army you can actually use in an attack. State the castle case that way, not
as an income claim.

Amortization: a castle built at turn `T` returns `0.5 * (1200 - T)` army. It
costs 35–47 army plus the turns of expansion given up while banking. Break-even
on the army alone is about turn 1130, so the `MAX_TURN_TO_BUILD = 900` gate in
`castle_builder` is not the binding constraint — the opportunity cost of
banking is.

### L4 — Vision radius 1 means one probe sweeps a 3-wide corridor

A single moving stack reveals a 3×3 box each tick, so it clears a 3-wide lane
along its path. Scouting is cheap: 2 army buys a lane.

### L5 — Deathtouch defence is a chase, and you get exactly one turn to use it

From `deathtouch.py`: a counter-move from a **third** tile onto the attacker's
*source* cell is a chase, so it resolves first; capture that source outright
and the touch never executes. A counter from the general itself does not work
(head-on clash, attacker wins unless armies are exactly equal).

The reaction window exists because observations are computed before actions:
an enemy stack that steps adjacent to your general on turn `T` is visible in
your turn `T+1` observation, and its touch also happens on `T+1`. Your chase
on `T+1` pre-empts it.

### L6 — Both sides touching on the same tick is a draw

So the goal is to be able to touch **at** turn 800, and to deny the opponent
the same tick. If you can touch and you can also see that the opponent can
touch, chasing is better than touching: a draw scores 0.5, a successful chase
keeps the win available next tick.

### L7 — The spawn rule gives a free prior on the enemy general's location

`min_generals_distance = 17` with sides 18–21. Run one BFS over passable
cells from your own general at turn 0; every cell with distance `< 17` is
excluded. The candidate set is typically the far corner block. This costs one
BFS for the whole game and is the only sound way to aim a scout before first
contact.

Caveat: the generation floor is measured on the generated grid before padding,
and a bot's BFS treats `type == 5` (structure in fog) as impassable, so
distances can be over-estimated. Treat `>= 17` as a prior to be intersected
with "never seen", not as a proof. If the candidate set empties, fall back to
"passable and never seen".

---

## 3. Shared building blocks

Implement once, reuse in all bots. Cost is trivial: a 21×21 board is 441
cells, so a full-board scan is ~1.8k neighbour checks and a BFS is similar.
Several of each per tick sits far inside the 150 ms budget.

### 3.1 General reserve (fixes D1)

```
RESERVE_OPENING_END = 60          # spend freely before this
def general_reserve(turn):
    if turn < RESERVE_OPENING_END:
        return 0
    return min(30, 3 + (turn - RESERVE_OPENING_END) // 25)
```

Rule: the general cell may be used as a move source only when the army left
behind is at least `general_reserve(turn)`.

- `split = 0` leaves 1 behind → allowed only while `general_reserve(turn) <= 1`.
- `split = 1` leaves `ceil(army/2)` behind → allowed when
  `army - army // 2 >= general_reserve(turn)` and `army // 2 >= 1`.

The opening exemption matters: at turn 0 the general is the only owned cell,
so a hard "never spend" rule would forfeit the opening. After turn 60 the
reserve grows and the general becomes a real garrison.

### 3.2 Enemy-general belief

State kept across ticks:

- `enemy_general`: `(r, c)` once seen (`owner == 2 and type == 4`). Generals
  never move, so one sighting is exact forever. Already implemented in
  `general_hunter` and `castle_builder`.
- `ever_seen[r][c]`: set for every cell not in fog this tick.
- `candidates`: initialised at the first tick to
  `{cells : passable and bfs_from_own_general >= 17}`; a cell is removed once
  `ever_seen` is set and it was not the general.

### 3.3 Smallest-sufficient-source capture scoring (fixes part of D3)

The current formula `army * 10 * (2 if opp else 1)` picks the **largest**
stack, so a 40-army stack is spent taking a 1-army neutral while the
frontier keeps small stacks. Invert it:

```
value(dest) = 12  if dest_owner == 2            # take enemy land and deny it
              6   if visible neutral plain      # owner 0, type in {1, 3}
              0   otherwise
frontier_gain(dest) = count of passable 4-neighbours of dest that are not owned
score = 100 * value(dest) + 15 * frontier_gain(dest) - src_army
```

Maximise `score` over `(source, direction)` pairs that satisfy
`src_army > dest_army + 1` and the reserve rule. Subtracting `src_army`
selects the **smallest adequate** source, which is the standard efficiency
rule for this game family: it keeps large stacks intact for a breakthrough and
still expands every tick.

### 3.4 Replace the `first_valid` fallback (fixes D4)

Fallback order when no capture scores above zero:

1. **Convey**: BFS distance field from the frontier over owned cells only;
   move the largest interior stack one step down the gradient. (`army_convey`
   owns this idea; `expand_plus` already has the shape of it.)
2. **Probe**: if `candidates` is non-empty, send a `split=1` detachment from
   the nearest stack with `army >= 4` toward
   `argmax(reveal_gain(cell) / (1 + steps))`, where `reveal_gain` counts
   candidate cells inside the destination's 3×3 box.
3. **Pass** (`1 0 0 0 0`). Passing is strictly better than a random shuffle:
   it keeps army where it is and cannot walk the general out of its own tile.

### 3.5 Win checks, evaluated before anything else, every tick

```
# W1 pre-800 snipe: works at any turn, wins instantly
if enemy_general is visible or remembered:
    for each owned 4-neighbour s of enemy_general:
        if army[s] > army[enemy_general] + 1:      # unknown army -> assume 1 if fogged
            return move(s -> enemy_general, split=0)

# W2 deathtouch: army-blind from turn 800
if turn >= 800 and enemy_general is not None:
    for each owned 4-neighbour s of enemy_general:
        if army[s] >= 2:
            return move(s -> enemy_general, split=0)
```

If the target is remembered but fogged, its army is unknown. Assume it is
large before turn 800 (do not gamble a breakthrough stack on W1), and assume
nothing after 800 because W2 ignores army.

### 3.6 Chase defence (implements L5)

Active from turn `DEFEND_FROM = 780`:

```
for each enemy-owned cell e that is 4-adjacent to my general:
    for each owned cell s != my_general that is 4-adjacent to e:
        if army[s] > army[e] + 1:
            return move(s -> e, split=0)      # chase: resolves before their touch
```

Maintain the means to do it: from turn 700, keep the highest-army owned
neighbour of your own general topped up by conveying into it, target
`sentry_army >= 12`. Never chase from the general itself (L5).

Ordering against W2 (L6): if W2 is available **and** an enemy cell is adjacent
to your own general, prefer the chase. Touching into a mutual touch scores a
draw; chasing keeps the win for the next tick.

---

## 3.7 Overlap with the sibling bot specs — keep identities distinct

Eight new bots are specced in this directory. Three of them own ideas that
also appear above, so the shared parts must land as *defect fixes* in the
existing bots, not as strategy identity, or the pool ends up with duplicates
and the leaderboard stops being informative.

| Idea | Owning bot | What the existing four get |
| --- | --- | --- |
| Defense as a whole strategy: growing reserve floors, threat model, intercepts | [`garrison`](garrison.md) | only the minimal reserve in §3.1 and the chase in §3.6 — enough to not lose by accident, no threat model |
| Mass one stack and commit it | [`late_rush`](late_rush.md) | nothing; `general_hunter` sends cheap probes, not a main stack |
| Broad map revelation | [`fog_scout`](fog_scout.md) | `general_hunter` gets *targeted* search only: the ≥17 spawn prior aimed at one cell, not whole-map coverage |
| `split=1` as a policy | [`splitter`](splitter.md) | `split=1` used only for the general reserve (§3.1) and to detach probes (§3.4) |
| Corridor control | [`choke_control`](choke_control.md) | nothing |
| Interior conveying as a policy | [`army_convey`](army_convey.md) | conveying used only as the idle fallback (§3.4 step 1) |
| Aggressive build parameters | [`castle_rush`](castle_rush.md) | `castle_builder` keeps the conservative cap; the funding and placement fixes in §4.3 are corrections, not aggression |

The distinction that matters most: `general_hunter` and `late_rush` both end
in a deathtouch execution, but the *targeting* differs. `general_hunter`
locates the general with cheap probes against the spawn prior;
`late_rush` walks a main stack at a remembered or guessed target. If both end
up doing the same thing, delete one.

---

## 4. Per-bot contracts

### 4.1 `smoke` — freeze it, do not optimize it

Recommendation: **make no strategy change.**

- It is the Phase 1 stdio smoke test. Its documented purpose is proving the
  wire protocol, not competing (`docs/bots/smoke.md`).
- The arena needs a frozen weak anchor. Elo is only meaningful relative to
  something; if every bot improves together, the leaderboard stays flat for
  the same reason it is flat now.
- A weak bot is a *source* of decisive games. Every optimized bot should beat
  `smoke` outright once §3.5 lands. If `smoke` still draws against an
  optimized bot, that is a bug in the optimized bot, and the signal is only
  readable because `smoke` did not move.

If a stronger baseline is wanted, fork `bots/smoke_v2/` and leave `smoke`
alone. The only acceptable edit to `smoke` is emitting the telemetry line in
§5.2, which does not change its policy.

### 4.2 `expand_plus` — make it a competent expander that can win

Keep the identity ("expand"). Apply, in this order:

1. §3.5 win checks (W1, W2). This alone converts every game against a bot
   with a spent general into a win instead of a draw.
2. §3.1 general reserve. This alone stops it losing to any opponent that
   adopts W1.
3. §3.3 smallest-sufficient-source scoring, replacing
   `army * 10 * (2 if opp)`.
4. §3.4 fallback chain: keep the existing BFS march as step 1 (it is already
   the convey idea), add the probe, and replace `first_valid` with `PASS`.
5. §3.2 belief state (`ever_seen`, `candidates`), needed by the probe.

Do **not** add castle building or a hunt phase — those are
`castle_builder` / `general_hunter` / `phase_switch`.

Falsifiable prediction: `expand_plus` v2 beats `smoke` on ≥ 80% of seeds with
a terminated (not truncated) result, and beats `expand_plus` v1 on ≥ 60%.

### 4.3 `castle_builder` — stop resting the general, fix the placement

The current design has two structural errors.

**Error 1: resting is the wrong funding mechanism.** Resting the general banks
0.5 army/turn, so a 47-army castle needs ~94 turns of resting. Conveying the
land bonus instead moves ~1 army per turn (one move per tick), so the same
castle is funded in ~40 turns without giving up the general's garrison.

**Error 2: relocating the general puts the castle in the most expensive
possible cell.** `_maybe_relocate_general` moves the whole general stack one
hop and builds there. A cell adjacent to the general pays the maximum
surcharge (+12, price 47), and moving the stack out empties the general to 1
army — re-creating D1 deliberately at the worst moment.

Replacement policy:

```
BANK_MIN_SPACING = 7        # Manhattan distance to every own structure
BUILD_GARRISON   = 10       # army the new castle must retain

choose_bank_cell():
    among owned plain cells (type == 1) with
        min_manhattan_distance_to_own_structures >= BANK_MIN_SPACING
    prefer the cell minimising BFS distance to the frontier
    (income appears where it is needed; see L3)
    -> cost(cell) is exactly 35, the floor price

each tick:
    if no capture scores above zero:
        convey one step along the BFS gradient toward bank_cell
    if army[bank_cell] >= 35 + BUILD_GARRISON:
        return (2, bank_r, bank_c, 0, 0)
```

Other parameter changes:

| Parameter | Now | Recommended | Reason |
| --- | --- | --- | --- |
| Funding | rest the general | convey the land bonus into a bank cell | ~2× faster, keeps the garrison |
| Placement | neighbour of the general (price 47) | ≥ 7 Manhattan from every own structure (price 35) | 12 army cheaper per castle, no surcharge stacking |
| `BUILD_SURPLUS_MARGIN` | 15 | 10 as a hard garrison floor | a castle built at 0 army is sniped by one unit (`build_castles.py`) |
| `MAX_OWN_CASTLES` | 2 | 3, all at ≥ 7 spacing | a 7-spaced lattice on an 18–21 board holds ~9 points, so all three still cost 35 |
| `MAX_TURN_TO_BUILD` | 900 | 1050 | amortization break-even is ~turn 1130 (L3) |
| General as move source | unrestricted | §3.1 reserve | D1 |
| Win checks | none | §3.5 | it currently cannot win even when handed the chance |

Castles are also the natural chase reserve (§3.6): they accumulate army
without attention, so a castle placed between the general and the frontier
doubles as the sentry supply.

Falsifiable prediction: ≥ 3 castles per game at price 35 each, mean turn of
first castle earlier than v1, and no game lost to a general snipe.

### 4.4 `general_hunter` — the decisiveness bot, needs a working scout

The hunt logic is sound; the targeting is missing. Add, in order:

1. **Candidate prior** (§3.2, L7). One BFS at the first tick.
2. **Probe economy** (§3.4 step 2). Detach `split=1` scouts of 2–4 army from
   any stack with `army >= 4`, route them to
   `argmax(reveal_gain / (1 + steps))` over `candidates`. Two live probes at
   once is enough; more starves expansion.
3. **Arrival timing.** Keep `eta = BFS distance from the best runner to
   enemy_general`. Start the final approach at `turn >= 800 - eta - 5`, park
   on any cell 4-adjacent to the target, and touch on the first tick ≥ 800.
   Parking early is free — the runner needs only 2 army.
4. **W1 before W2.** If the general is sighted before 800 and its army is
   visible and beatable, snipe immediately; there is no reason to wait for
   deathtouch against the current pool (L2).
5. **Chase defence** (§3.6) from turn 780, with the sentry top-up from 700.
   Without this, `general_hunter` loses to the first sibling bot that copies
   its own hunt.
6. **General reserve** (§3.1).

Ordering inside `act`, highest first:

| Rank | Rule | Section |
| --- | --- | --- |
| 1 | chase defence (enemy adjacent to own general, turn ≥ 780) | 3.6 |
| 2 | W1 pre-800 snipe | 3.5 |
| 3 | W2 deathtouch execute (turn ≥ 800) | 3.5 |
| 4 | final approach to a known general (`turn >= 800 - eta - 5`) | 4.4.3 |
| 5 | capture by smallest sufficient source | 3.3 |
| 6 | convey toward the frontier | 3.4 |
| 7 | probe toward the best candidate | 3.4 |
| 8 | `PASS` | 3.4 |

Falsifiable prediction: `general_hunter` v2 sights the enemy general before
turn 800 on ≥ 60% of seeds, and converts ≥ 80% of those sightings into a
terminated win.

---

## 5. Measurement requirements

### 5.1 Why the current records cannot settle these claims

`data/games/*.json` stores `winner`, `turns`, `terminated`, `truncated` and
nothing else. Every claim above about land, army, income, or sighting is
unmeasurable from the store today. Experiment notes `001`, `002`, `003` and
the `castle_rush` / `phase_switch` specs all record the same gap.

### 5.2 Requested schema addition (backward compatible)

Add to `arena/store.py` as **optional, nullable** fields. Missing means "not
measured", never zero. Records written before the change must keep loading, so
read every new key with `.get()` and default `schema_version` to 1.

| Field | Type | Source |
| --- | --- | --- |
| `schema_version` | int | 2 for new records, 1 when absent |
| `duration_seconds` | float or null | wall clock in `run_and_store` |
| `castles_built_a` / `_b` | int or null | parse `[matchup] castles built: N (label) vs M (label)` — `matchup.py` already prints this under build-castles modes |
| `final_land_a` / `_b` | int or null | see the telemetry convention below |
| `final_army_a` / `_b` | int or null | same |
| `metrics` | object | free-form per-experiment counters, default `{}` |

`matchup.py` never prints land or army, so those need an opt-in bot-side
convention. Each bot writes **one** line to stderr when its stdin reaches EOF
(matchup inherits bot stderr, so `run_match.py` captures it):

```
[telemetry] player=<0|1> turn=<int> my_land=<int> my_army=<int> opp_land=<int> opp_army=<int>
```

Values come from the last observation frame the bot parsed. `run_match.py`
takes the last line per player, maps player 0 to bot A and player 1 to bot B,
and fills the opposite side from the reporting bot's `opp_*` view when only
one bot reports. One line per bot per game keeps the logs clean and cannot
affect gameplay.

Do not add `trajectory_path` or any training field. Phase 4 is not in scope.

### 5.3 Metrics each experiment note must report

Beyond winrate and mean turns: decisive rate, final land differential, final
army differential, castles built per side, turn of first enemy-general
sighting, hunt/probe activation count, and fault count. The first three are
what make a draw-heavy season rankable at all — see
[`tournament-plan.md`](tournament-plan.md).

---

## 6. Expected effect on the draw rate

| Change | Effect on decisiveness |
| --- | --- |
| §3.5 win checks alone, one bot only | that bot beats every unpatched bot as soon as a probe reaches a spent general |
| §3.1 reserve alone, one bot only | that bot stops losing to §3.5, and games return to draws |
| §3.5 + §3.1 on all four | decisive games come from *scouting speed*, which is the intended skill axis |
| §3.6 chase defence | prevents the endgame collapsing into whoever probes first |

The intended equilibrium is: reserves make generals hard to kill by accident,
so the deciding skill is finding the enemy general before turn 800 and denying
the same to the opponent. That is a real skill ladder and it produces
decisive games. The current equilibrium — nobody defends and nobody attacks —
produces only draws.

## 7. Risks

- **150 ms per move, 50 faults forfeits** (`RULES.md` §08). The added work is
  a handful of full-board BFS passes per tick in pure Python on 441 cells;
  measured cost is well inside budget, but any new per-tick loop over
  `candidates × cells` must be avoided.
- **Passing instead of shuffling** (§3.4) looks idle in logs. It is correct.
  Do not "fix" it by restoring `first_valid`.
- **The probe leaks vision both ways.** A probe standing in enemy territory
  is a cell the enemy can see. Against an opponent with §3.6 it will be
  chased. Expect probe attrition; that is why they cost 2 army.
- **One change per experiment** (`experiment-protocol.md`). §3.1 and §3.5
  interact strongly — land them as two separate measured steps, reserve
  first, so the win-check result is not attributed to the reserve.

---

## Parameter revision 1

Source: [`../measurements/round1.md`](../measurements/round1.md) and
[`round1.json`](../measurements/round1.json) — 58 games, generated
2026-07-30T23:34:29Z.

Read [`diversity-constraints.md`](diversity-constraints.md) before you apply
any item here. Every revision below stays inside its bot's own axis.

### R1.0 What round 1 changed about §1–§7

The §3 helpers were already in the working tree when round 1 ran. Commit
`499f7bd` landed 14 s after the last game, and `castle_builder` built 3
castles per game, which only the revised `MAX_OWN_CASTLES = 3` allows. So
round 1 measured **`expand_plus` v2, `castle_builder` v2 and `smoke` v1**, not
the pre-helper code. `general_hunter` v2 played **zero** games.

Confirmed by the data:

| §1 claim | Round 1 verdict |
| --- | --- |
| The pool cannot produce decisive games | **Falsified.** Decisive rate is 24/58 = 41.4%, inside the 30–70% healthy band in [`tournament-plan.md`](tournament-plan.md) §3. Elo now separates: 1616.4 down to 1453.5. |
| The §3 helper set makes a bot competent | **Not supported.** `expand_plus` v2 finished 0-3-5 and lost to `fog_scout`, `army_convey` and `late_rush` — three bots that carry none of the helpers. |
| §4.3 placement fix produces ≥ 3 castles at price 35 | **Confirmed on castle count.** `castle_builder` built 3 in every game. |
| L3: castle income buys a usable attack | **Not supported.** Castle count and result are inversely ordered: `castle_rush` 4 castles → 0 wins, `castle_builder` 3 → 0 wins, `phase_switch` 2 → 1 win, and the three winners built 0. |

New population facts that drive the revisions:

| Fact | Value | Consequence |
| --- | --- | --- |
| Decisive games | 24 of 58 | The pool is rankable. Stop optimizing for "produce a decisive game". |
| Decisive games ending before turn 800 | 22 of 24 = 91.7% | Every turn-800 mechanism is dead in ~92% of decided games. |
| Mean turn of a decisive game | 558 | Earliest 341, median near 528. |
| Bots that could not beat `smoke` | 5 of 8 new bots | `smoke` is doing its anchor job. |
| Winners' shared trait | 0 castles, army concentration or map coverage | Land-to-one-stack conversion decides games; income does not. |

### R1.1 The two defects round 1 exposes in the shared helper module

Both live in `bots/*/strategy_common.py`, which is duplicated in
`expand_plus`, `castle_builder` and `general_hunter`. Fix them in all three
copies or the copies drift.

**Defect H1 — a fogged enemy general is treated as a 1-army cell.**

`enemy_general_army` returns `1` when `type_grid[r][c] == 0`. `win_check_w1`
then fires whenever any owned neighbour holds ≥ 3 army, even though an unspent
general holds about `1 + 0.5 * turn` army — roughly 225 at turn 448. The
attack cannot succeed, the source cell is emptied, and W1 fires again the next
tick. §3.5 states the opposite rule ("assume it is large before turn 800"), so
this is an implementation inversion, not a design choice.

This is the most probable cause of `expand_plus` v2's three losses. It is
reachable only after a sighting, which is exactly when the bot has a stack near
the enemy general to throw away.

| Parameter | Now | Revision 1 | Reason |
| --- | --- | --- | --- |
| Assumed army of a fogged general | `1` | `1 + turn // 2` | The unspent-general growth model from §D1. Blocks W1 unless the bot really can win. |

**Defect H2 — `general_hunter` cannot probe.**

`Agent._update_active_probes` marks **every** owned cell with `2 <= army <= 4`
as an active probe, and `probe_move` returns `None` once `len(active_probes)
>= 2`. Ordinary land cells hold 2–4 army all game, so the cap is met on almost
every tick and the probe path never runs. This recreates defect D2: the hunt
has no targeting, exactly the failure §4.4 was written to remove.

| Parameter | Now | Revision 1 | Reason |
| --- | --- | --- | --- |
| Active-probe membership | any owned cell with `2 <= army <= 4` | only cells the bot dispatched as a probe, carried forward by the step it took | An army-size test cannot identify a probe. |
| Active-probe cap | 2 | 2, unchanged | The cap was never the problem. |

**Cost warning attached to H2.** Once probing works, `probe_move` runs one
`bfs_distances` pass **per candidate cell**, plus a 441-cell scan inside each.
With 100–200 candidates that is 10⁵–10⁶ operations of pure Python per tick,
against a 150 ms budget (`RULES.md` §08). Cache the candidate loop or evaluate
a fixed sample of candidates per tick before round 2. A fault storm would void
the round.

### R1.2 `smoke` — freeze, confirmed by measurement

**Revision: no parameter change. `smoke` has no tunable constants and gains
none.**

§4.1 argued for freezing `smoke` from first principles. Round 1 supplies the
evidence:

- 16 games, the largest sample of any bot, 0 W – 5 L – 11 D.
- It separates the field cleanly. `fog_scout`, `army_convey` and `late_rush`
  beat it; `garrison`, `splitter`, `choke_control`, `phase_switch` and
  `castle_rush` could not. A frozen anchor that splits the field 3–5 is
  working exactly as intended.
- Its round Elo, 1453.5, is last. [`tournament-plan.md`](tournament-plan.md)
  §5 makes "`smoke` finishes last" the control condition. The control held.

The only permitted edit stays the §5.2 telemetry line, which does not change
the policy. `bots/smoke/main.py` already carries it and `bots/smoke/agent.py`
is untouched. Keep it that way.

**Read the 5 bots that cannot beat `smoke` as a finding about those bots**, not
as a reason to weaken the anchor.

### R1.3 `expand_plus` — stop paying twice for enemy land

Round 1: 0 W – 3 L – 5 D over 8 games, round Elo 1464.7, lost at turns 448,
554 and 554 to `fog_scout`, `army_convey` and `late_rush`.

**Primary revision (apply alone, measure, then continue):**

| Parameter | Now | Revision 1 | Reason |
| --- | --- | --- | --- |
| `dest_value(owner == 2)` | `12` | `6` | Equal to the neutral-plain value. |

Mechanism. The capture score is
`100 * dest_value + 15 * frontier_gain - src_army`, so an enemy tile scores
1200 and a neutral plain scores 600. The gap is 600, while `frontier_gain`
contributes at most 60 and `src_army` a few tens. The enemy tier therefore wins
every comparison it enters, and the bot grinds one defended tile at a time
while free neutral land sits next to it. Setting both tiers to 6 hands the
decision to `frontier_gain` and `src_army`, which is the land-rate rule that
defines this bot's axis. This is not aggression tuning; it removes a hidden
tier that overrides the stated scoring rule.

**Secondary revisions, in order, one measured step each:**

| Order | Parameter | Now | Revision 1 | Reason |
| --- | --- | --- | --- | --- |
| 2 | Defect H1 | fogged general = 1 army | `1 + turn // 2` | Stops the repeated suicide attack that most likely caused the three losses. |
| 3 | `general_reserve` cap | `min(30, ...)` | drop the cap, or delete the parameter | The cap is inert in both branches. For `split = 0` the check is `reserve <= 1`, which fails from turn 60 onward whatever the cap is. For `split = 1` the check is `army - army // 2 >= reserve`, and an unspent general holds about `1 + 0.5 * turn` — roughly 225 army by turn 448 — so a 30-army floor is met with room to spare. The parameter reads as a garrison size but sets nothing. Delete it or make it bind, and do not cite it as the general's defense. |

**Do not raise the reserve to answer the losses.** The arithmetic says it
cannot work: an unspent general gains 0.5 army/turn, while an opponent that
funnels the 50-turn land bonus from 80 cells into one stack gains about
1.6 army/turn. A static garrison loses that race by more than 3×. Defense
against a massed stack belongs to [`garrison`](garrison.md), not here.

**Falsifiable prediction.** With the primary revision alone, `expand_plus` v3
holds more land at truncation than v2 on ≥ 7 of 10 paired seeds, and does not
lose more games than v2. **Revert if** it wins no more games *and* holds no
more land, or if the loss count rises.

### R1.4 `castle_builder` — build inside the horizon that decides games

Round 1: 4 games, all draws at 1200 turns, 3 castles per game, round Elo
1495.2. Its decisive rate is 0%, which
[`tournament-plan.md`](tournament-plan.md) §3 defines as "not being measured".
It met only `castle_rush` and `phase_switch`; it never played a bot that wins.

**Primary revision:**

| Parameter | Now | Revision 1 | Reason |
| --- | --- | --- | --- |
| `MAX_TURN_TO_BUILD` | `1050` | `600` | The 1050 value came from a 1200-turn amortization. Round 1 says decisive games end at mean turn 558 and 92% finish before 800, so the effective horizon is ~800, not 1200. A castle built at turn `T` returns `0.5 * (800 - T)` usable army; at cost 35 that breaks even at turn 730 and needs roughly another 100 turns for the army to reach a frontier. 600 is the last turn at which a castle can still pay for itself and be spent. |

After turn 600 the bot stops banking and spends castle income on captures. The
conservative-economy identity is unchanged: it still builds few castles, at the
floor price, with a full garrison.

**Secondary revisions:**

| Order | Parameter | Now | Revision 1 | Reason |
| --- | --- | --- | --- | --- |
| 2 | Defect H1 | fogged general = 1 army | `1 + turn // 2` | Same shared-module fix as R1.3. |
| 3 | `MIN_TURN_TO_BUILD` | `20` | `20`, held | The first castle is not the problem; the last one is. |
| 4 | `MAX_OWN_CASTLES` | `3` | `3`, held under review | See the diversity note below. |
| 5 | `BANK_MIN_SPACING`, `BUILD_GARRISON` | `7`, `10` | held | Confirmed working: 3 castles per game, no castle sniped. |

**Diversity note, must be checked before round 2.** `castle_builder` at cap 3
sits one castle away from `castle_rush` at cap 4, and the two drew every game
with a 21-point round-Elo gap. That is the parameter-drift alarm in
[`diversity-constraints.md`](diversity-constraints.md) §4. The
`MAX_TURN_TO_BUILD` revision restores separation on the timing axis instead —
`castle_builder` stops at 600, `castle_rush` has no end gate at all. If round 2
still cannot separate the two bots, cut the cap back to 2 rather than tuning
them closer.

**Falsifiable prediction.** `castle_builder` v3 records a non-zero decisive
rate over a grid that includes `army_convey`, `fog_scout` and `late_rush`, and
its mean turns fall below 1200. **Revert if** it builds fewer than 3 castles
per game or its loss count exceeds `castle_rush`'s on the same grid.

### R1.5 `general_hunter` — unmeasured, and its clock is set for a game that ends first

Round 1: **0 games.** No result exists for this bot. Every item here comes from
the population statistics and from code, and every one is provisional until
`general_hunter` plays a grid.

The population fact that matters: 22 of 24 decisive games ended before turn
800, at a mean of turn 558. `general_hunter`'s whole endgame — `win_check_w2`,
the final approach at `turn >= 800 - eta - 5`, `chase_defence` from 780, the
sentry from 700 — activates after most games are already over. With a typical
`eta` near 20 the approach begins at turn 775. In round 1 conditions the bot
would be eliminated, on average, 217 turns before its first hunt move.

**Primary revision:**

| Parameter | Now | Revision 1 | Reason |
| --- | --- | --- | --- |
| Final-approach trigger | `turn >= DEATHTOUCH_TURN - eta - 5` | `turn >= APPROACH_START` with `APPROACH_START = 450`, keeping the existing trigger as an upper bound | 450 sits below the mean decisive turn of 558, so the runner is parked next to the enemy general before the game is typically decided. Parking is nearly free: the runner needs 2 army, and it touches on the first tick at or after 800. |

`DEATHTOUCH_TURN = 800` is fixed by the engine and is not a tunable. The kill
condition does not change, so the bot's identity does not change — only the
travel schedule does.

**Secondary revisions, in order:**

| Order | Parameter | Now | Revision 1 | Reason |
| --- | --- | --- | --- | --- |
| 2 | Defect H2 | probes disabled by an army-size test | dispatch-tracked probes | Without this the bot has no targeting and `APPROACH_START` has nothing to approach. Apply **before** the primary revision if only one change fits the schedule. |
| 3 | `probe_move` cost | one BFS per candidate | sample or cache candidates | 150 ms budget, `RULES.md` §08. |
| 4 | Defect H1 | fogged general = 1 army | `1 + turn // 2` | Same shared-module fix. |
| 5 | `SENTRY_FROM`, `DEFEND_FROM` | `700`, `780` | `450`, `500` | Both defend against an attack that, per round 1, arrives near turn 558. Keeping them at 700 and 780 defends an empty board. |

**Falsifiable prediction.** `general_hunter` v3 sights the enemy general before
turn 800 on ≥ 60% of seeds — the §4.4 prediction, still untested — and its
probe path fires at least once per game. **Revert if** the probe path still
never fires, or if per-move time exceeds 150 ms on a 21×21 board.

### R1.6 Order of work

Defects first, because they invalidate any measurement taken around them.

| Step | Change | Bots touched | Gate |
| --- | --- | --- | --- |
| 1 | Defect H2 — probe dispatch tracking | `general_hunter` | Probe path fires at least once per game. |
| 2 | `probe_move` cost reduction | all three helper copies | Per-move time under 150 ms on 21×21. |
| 3 | Defect H1 — fogged-general army assumption | all three helper copies | W1 no longer fires against a fogged general with a small stack. |
| 4 | R1.3 primary — `dest_value(enemy) = 6` | `expand_plus` | Paired seed grid vs v2. |
| 5 | R1.4 primary — `MAX_TURN_TO_BUILD = 600` | `castle_builder` | Paired seed grid vs v2. |
| 6 | R1.5 primary — `APPROACH_START = 450` | `general_hunter` | Paired seed grid vs v2. |

Steps 1–3 are defect fixes and may share one experiment note. Steps 4–6 are
strategy changes and need one note each, per
[`experiment-protocol.md`](../experiment-protocol.md).

### R1.7 Unresolvable with the current store

Round 1 added `castles_a` / `castles_b` and the tag grid, which is what made
the economy conclusion possible. Still missing, and still blocking:

- Final land and army at truncation. Without them the 34 drawn games carry no
  ranking information, and the R1.3 land prediction cannot be scored.
- Turn of first enemy-general sighting. This is the number that decides whether
  R1.5 worked, and no bot reports it.

Both are the §5.2 telemetry line. Land it before round 2 or R1.3 and R1.5 stay
unfalsifiable.
