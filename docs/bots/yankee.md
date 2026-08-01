# yankee

`bots/yankee/`. A fork of [`proteus`](proteus.md) that keeps the classifier and
the switcher and adds two things proteus does not have: a Monte Carlo tree
search scoped to tactical windows, and a third strategy core that plays
[`RULES.md`](../../RULES.md) §07's endgame as its own game.

Spec and measurements:
[`../research/strategies/yankee.md`](../research/strategies/yankee.md),
[`../research/measurements/yankee.md`](../research/measurements/yankee.md).

## Strategy

1. **Classify and switch** exactly as proteus does — aggressor → blitz,
   economy → boom, unknown → the blitz spine, with asymmetric hysteresis.
2. **From turn 800, play the deathtouch core** instead. Selected on the clock,
   outside the hysteresis, because §07 is a clock rule and does not flicker.
3. **Search, under blitz only.** When an enemy stack that could actually take
   our general is inside the window, or a kill is landable, run a 40 ms MCTS
   over a shortlist whose first entry is always the core's own move.

## The three cores

| core | when | what it optimises |
| --- | --- | --- |
| blitz | the spine, and against anything that walks a fist at us | racing to their general |
| boom | against opponents that spend army on land | economy, castles, one late fist |
| deathtouch | from turn 800 | touching their general and chasing off theirs |

The blitz and boom cores are **vendored copies** of `bots/blitz/agent.py` and
`bots/boom/agent.py`, not imports. Those files are in proteus's source closure,
and yankee is measured against proteus, so editing them would move the baseline
under the comparison.

## Why the endgame is a core and not a clause

Six bots already carry `deathtouch_turn = 800` as an extra clause in a
finishing check. From turn 800 three things change that a pre-800 core is
actively working against:

- army on our own general defends nothing (one unit is lethal);
- the only defence is a chase that **captures** the attacker's source, from a
  tile that is not the general — which puts the endgame garrison at distance 2
  from home, not on it;
- a 2-stack that arrives beats a 40-stack that is walking, so routing a touch
  is a path problem over friendly ground.

## What the search will not touch

It runs only under the blitz core and never overrides a castle build. Both
restrictions are measured: overriding boom cost 5.5 and 10.0 points of winrate
against fog_scout and metro while gaining 2.1 against cm_hunter. blitz re-derives
its strike stack from the board each turn and so survives an override; boom's
build plan and strike latch do not.

## Telemetry

proteus's keys, plus `searched` / `overrode` / `search_iters` / `move_ms`.
`searched` and `overrode` are separate because "the search never ran" and "it
ran and agreed" look identical from outside and are different defects.
