# 050 — sosipolis: take the kill that is already on

Bot: [`bots/sosipolis/`](../../../bots/sosipolis/) — follows 049.
**Do not auto-revert**; wait for GUI.

## Defect (seed 6, traced turn by turn)

Kubic's sight→kill is 20–24 ticks (§6). Ours was a median of 51, and one
sighted game spent 186 ticks in strike and lost. The trace of seed 6 shows
both reasons in ten turns:

```
t     branch    tip     tipA  d(tip,gen)  gen army
141   tip_feed  (3,1)   7     2           2      <- kill is on; we feed
...
146   tip_feed  (3,1)   7     2           7      <- general has grown
147   mcts      (12,6)  17    14          7      <- tip moves 14 cells away
...
169   mcts      (12,6)  30    14          19     <- never came back
```

1. **The sight floor blocked a winning kill.** `TIP_AT_SIGHT_FLOOR = 10` is a
   Kubic *operating* level, and `_tip_feed_wave` applied it unconditionally: a
   tip holding 7 stood two steps from a general holding 2 and gathered instead.
   `tip_is_ready` could not release it either — `tip_mass_target` floors at
   `STRIKE_MIN_TIP = 23`, and `path_finish_need` adds `STRIKE_PATH_BUFFER`
   per hop plus a regen allowance, pricing this 2-hop finish near 10.
2. **Mass-first reselection abandoned the position.** `select_mass_tip` ranks
   by army, so a 17-army stack fourteen cells away took the tip from the one
   two cells from the kill.

## Revision

| Change | Intent |
| --- | --- |
| `tip.finish_cost_exact` — path cost with no buffers | price a kill that is already on, as opposed to planning a long march |
| `tip.can_finish_now` | one predicate: can this stack, as it stands, walk in and take the general |
| `_tip_feed_wave` returns None when the held tip can finish | the floor stops being a reason to stand next to a general we beat |
| ...and keeps the tip in that case | a winning stack is not handed over to a bigger one further away |

`path_finish_need` is untouched — its buffers are right for planning a march.

## Measurement

20 seeds vs `macaria`, clean harness:

| arm | wins |
| --- | --- |
| session start | 4/20 (20%) |
| + 049 economy | 5/20 (25%) |
| + this change | **7/20 (35%)** |

Seeds 1 and 3 flipped to wins and **no seed flipped the other way** — every
win the previous arm had was kept. Post-sight strike length median 51 → 32
(Kubic 20–24); one game now kills on the first strike tick.

## Where the win rate still goes

Funnel over 10 games: contact 10/10, **sight 5/10**, win-given-sight 3/5.
Sight means owning a cell next to their general, and the closest we ever get
is a median of **4 cells** — twice with 50+ army on the tip, so it is not a
mass problem at the end. Tried against that and measured *not* to work, all
recorded in 049: expanding during contact, a home muster, and a local sweep
that finishes the approach once the tip arrives (closest 4 → 5, sight
unchanged).

For scale: `macaria` tops the arena leaderboard at 2376 Elo against
sosipolis's 2034 — a 342-point gap whose expected win rate is 12%, so 35% is
already well above par. A 70% target means putting sosipolis ~147 Elo *above*
the strongest bot in the pool, i.e. ~490 Elo of improvement. That is a new
bot, not a debugging pass.

## Gate

```bash
pytest bots/sosipolis/tests tests -q
python competition-module/competition/matchup.py \
  bots/sosipolis/run.sh bots/macaria/run.sh \
  --mode competition --seed 3
```

Tests: [`bots/sosipolis/tests/test_finish_now.py`](../../../bots/sosipolis/tests/test_finish_now.py)
— the two-step kill is affordable, the exact cost undercuts the march
planner, a losing stack is not ready, enemy tiles on the path are charged,
and unreachable goals never finish.

## GUI

```bash
python competition-module/competition/matchup.py \
  bots/sosipolis/run.sh bots/macaria/run.sh \
  --mode competition --seed 6 --gui --fps 8
```
