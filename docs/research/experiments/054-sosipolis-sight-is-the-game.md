# 054 — sosipolis vs macaria: sight is the whole game

Bot: [`bots/sosipolis/`](../../../bots/sosipolis/) — follows 053. Nothing in
this note is in the bot; it is a reframing plus four negative results.

## The funnel

Over 20 seeds at the 053 build, with true general positions taken from the
engine (`matchup.make_board(env, seed).general_positions`):

| | |
| --- | --- |
| P(sight their general) | **9/20 = 45%** |
| P(win \| sighted) | **7/9 = 78%** |
| P(win \| never sighted) | **0/11** |

We are already good at killing a general we can see. We find one in fewer than
half our games, and we have never once won without finding it.

**70% winrate needs P(sight) ~= 90%**, holding the 78% conversion. Every other
lever is second order until that moves: an extra point of conversion is worth
0.45 points of winrate, an extra point of sight is worth 0.78.

## What sight is not blocked by

Four things that looked like the cause and are not.

**The candidate set never loses the general.** Replaying each game and feeding
the true fogged observation to a fresh agent, the true general cell is in
`memory.candidates` at the end of *every* failed game. The set shrinks to
39-166 cells and we simply never go and look at them. It is not an elimination
bug.

**The belief is not biased deep.** Median `d(home, objective) - d(home, true
general)` is **+0** in games we sighted and **+0** in games we did not. The
waypoint is not systematically overshooting or undershooting.

**The economy is not the difference.** Our land share against macaria, split by
outcome:

| turn | won | lost |
| --- | --- | --- |
| 50 | 0.49 | 0.49 |
| 100 | 0.51 | 0.49 |
| 150 | 0.47 | 0.49 |
| 200 | 0.46 | 0.45 |
| 250 | 0.47 | 0.44 |

Identical. This also explains 051 row 12: directed expansion was a wash because
expansion was never what separated our wins from our losses.

**The assault is not being spent on neutrals.** Reconstructing the true board
under every one of our contact moves, the tip's army goes almost entirely into
*enemy* cells, not neutral ones — seed 2: 397 army into 116 enemy captures
against 10 army into neutrals. The stack is ground down chewing frontier, and
the volume of that grinding tracks the outcome (wins fight through 15-57 enemy
cells, losses 47-116). Direction does not: forward share is 90% in one loss and
63% in one win.

## What was tried and failed

| change | seeds 0-19 |
| --- | --- |
| baseline (053) | **8** |
| `MARCH_COST_CAP` 3 -> 8 | 6 |
| `MARCH_COST_CAP` 3 -> 15 | 6 |
| stall release, 40 turns without closing on the waypoint | 5 |
| stall release, 100 turns | 6 |
| stall release, 200 turns | 7 |

The stall release targets a real defect. A commitment is only released by
*arrival* or by *vision*, and both require reaching the waypoint, so one we
cannot reach is held for the rest of the game:

| seed | waypoint | turns held | closest the tip came |
| --- | --- | --- | --- |
| 1 | (0,19) — a corner | **853** | 7 |
| 13 | (20,18) | 351 | 12 |
| 17 | (8,16) | 295 | 2 |

Releasing it still does not help, and the win rate climbs monotonically back to
baseline as the release is weakened (40 -> 5, 100 -> 6, 200 -> 7, off -> 8).
The replacement waypoint is drawn from the same diffuse belief, so switching
targets buys nothing. **Do not re-try this without changing what the
replacement is chosen from.**

## Where this points

Sight needs us to *own a cell adjacent to* their general — visibility is a 3x3
block around owned cells, so being 3 cells away reveals nothing, and we were
within 8 cells on 31% of contact turns on seed 1 while never seeing it.

So the probe target is being chosen as a point estimate (belief argmax) when
what the situation calls for is **coverage**: with 39-166 candidates left and
hundreds of turns available, the question is which move eliminates the most
candidates, not which single cell is likeliest. That is a change to how the
objective is chosen, not a parameter, and is not attempted here.

## Reproduction

Diagnostics used, all against `data/trajectories/<round>/`:

```bash
for s in $(seq 0 19); do
  python arena/matches/run_match.py bots/sosipolis/run.sh bots/macaria/run.sh \
    --mode competition --seed $s --round bleed --record
done
```

True general positions come from `matchup.make_board(env, seed)`; the true
board under each recorded move comes from
`arena.records.trajectories.replay_states`.
