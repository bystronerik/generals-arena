# choke_control

`bots/choke_control/`. Grounded idea: **corridor hold**. Baseline for comparison: `expand_plus`, `smoke`.

## Strategy

1. Detect choke cells: narrow passable tiles (`narrow >= 2`) that pass a
   local gate test (passable neighbors split into disconnected components).
2. Weight capture destinations by `choke_score` (frontier-relevant gates
   only; fog-decayed near unexplored edges).
3. Hold rule: stacks on `held_choke` cells do not drain sideways unless
   shifting the hold forward, attacking with overwhelming force
   (`army - 1 >= 3 * threat`), or the stack is below `HOLD_MIN_ARMY`.
4. Deny rule: reinforce merges into held chokes outrank non-overwhelming
   forward attacks through the corridor.
5. Two-deep hold: `split=1` into a corridor mouth when the source sits on
   the corridor line and half army still captures.
6. BFS frontier march fallback — unchanged from `expand_plus`, with a
   preference to march toward held chokes under threat.

## Experiment

[`docs/research/experiments/009-choke-control-corridor-hold.md`](../research/experiments/009-choke-control-corridor-hold.md)

## Verification

```bash
source .venv/bin/activate
python competition-module/competition/matchup.py \
  bots/choke_control/run.sh \
  bots/smoke/run.sh \
  --mode competition --seed 0
```
