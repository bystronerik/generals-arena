# 048 — sosipolis: charge the tip for its route

Bot: [`bots/sosipolis/`](../../../bots/sosipolis/) — follows 047.
**Do not auto-revert**; wait for GUI.

## Defect

047 fixed where the probe aims; it still arrived with nothing. On seed 2 the
assault tip peaked at 30 army and was spent 18 turns later, twice in one game,
and sat under 6 army for **73%** of all post-contact turns.

The accounting, from a per-move trace of that game:

| tip step onto | steps | net tip army |
| --- | --- | --- |
| our own land | 71 | +41 |
| neutral | 24 | −24 |
| enemy tile | 30 | **−76** |

An enemy tile costs the unit left behind plus its garrison — about 2.5 army a
step here — while our own land is free. Turns 93–103 walked the tip through
eleven consecutive enemy tiles, 28 army down to 6, to advance eleven cells.

Cause: routing was hop count. `_ensure_waypoint_bfs` built an unweighted BFS
from the waypoint and `_shallow_action_score` rewarded any step that lowered
it, so a route straight through a snake scored the same as walking round it
for a fraction of the mass. `tip_mass_target` / `path_finish_need` already
model this cost exactly, but only gate the strike-phase feed — in contact they
merely scale a scoring weight, never stopping or redirecting a march the tip
cannot pay for.

## Revision

| Change | Intent |
| --- | --- |
| `army.march_cost` — own 0, neutral 1, enemy `1 + min(garrison, cap)` | price a step in the currency the tip actually spends |
| `army.march_dist` — Dijkstra over that cost | cheapest route, not shortest |
| `_ensure_waypoint_bfs` builds the cost map, cached per turn | the scorer's "am I closer" becomes "am I closer per army spent" |
| `MARCH_COST_CAP = 3` | see below — the target is *inside* their land |

`_UNREACHABLE` replaces the old `99` / `H+W` defaults, since costs are no
longer bounded by board size.

## The cap is the whole story

The cap trades tip mass against actually closing. Seed 2, one game each:

| routing | game turns | median tip | tip < 6 | tip ≥ 23 |
| --- | --- | --- | --- | --- |
| hop distance (before) | 272 | 3 | 73% | 12% |
| cost, cap 3 | 363 | 7 | 43% | 16% |
| cost, cap 6 | 334 | 4 | 63% | 22% |
| cost, cap 12 | 626 | 18 | **31%** | **45%** |

Cap 12 fixes the collapse outright — and then the bot stops winning. The
target sits inside their territory, so with a large cap a fat garrison reads
as a wall worth a ten-step detour: the tip circles, games stretch to 626
turns, and seed 1 turned from a win into a 1200-turn draw. Cap 3 keeps the
routing honest without letting a detour beat the objective.

## Measurement

**20 seeds vs `macaria`, one game per seed, clean harness:**

| arm | wins | mean game length |
| --- | --- | --- |
| 047 (hop routing) | 4 / 20 | 293 |
| this change (cost, cap 3) | 4 / 20 | 366 |

Four seeds flipped, two each way. **The win rate is unchanged.** An earlier
8-seed read showed 4 wins against 2 and did not survive twelve more seeds —
one game per seed does not separate these arms.

So: the collapse is real, it is now partly fixed (starved turns 73% → 43% on
seed 2), and it was **not** the binding constraint on beating macaria. Fixing
it completely, at cap 12, costs more games than it wins.

Latency is unaffected: move_ms median 17, max 81 against the 100 ms cap, with
contact turns at median 13 / max 23.

## Rejected

A tip-selection rule that refused to name a spent stack "the tip" while a much
larger one waited further back (`select_mass_tip` prefers stacks within
`STRIKE_TIP_MAX_DIST` of the goal, so a 2-army remnant kept the designation
while a 32-army stack sat at home). Measured alone it went 1 win / 8 against
2 for the unchanged bot, and combined with cap-12 routing it went 0 / 8 with a
draw. Not kept.

## Still open

The wave still spends its whole mass per trip and rebuilds from scratch; the
tip is fed by the mod-50 gather clock rather than by what the route to the
objective costs. `tip_mass_target` computes that number already and nothing in
the contact phase gates on it.

## Gate

```bash
pytest bots/sosipolis/tests tests -q
python competition-module/competition/matchup.py \
  bots/sosipolis/run.sh bots/macaria/run.sh \
  --mode competition --seed 1
```

Tests: [`bots/sosipolis/tests/test_march_cost.py`](../../../bots/sosipolis/tests/test_march_cost.py)
— per-tile cost by owner, the garrison cap, walking round a snake when both
routes are the same length, a target behind enemy land staying reachable, and
walls staying out of the map.

## GUI

```bash
python competition-module/competition/matchup.py \
  bots/sosipolis/run.sh bots/macaria/run.sh \
  --mode competition --seed 2 --gui --fps 8
```
