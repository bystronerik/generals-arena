# 056 — sosipolis: walk the candidate set instead of aiming at its peak

Bot: [`bots/sosipolis/`](../../../bots/sosipolis/) — follows 055.

054 established that sight decides these games: P(win | sighted) 78%,
P(win | never sighted) 0/11, P(sight) 45%. This changes how the contact probe
picks its target, and is the first thing in this line of work that moves the
win rate.

## The change

`ContactMCTS.prepare_contact` generated probe macros from the belief and
committed stickily to the best-scoring one. Now it takes the **nearest
candidate by march cost**, tie-broken by belief, and re-picks whenever that
candidate is cleared. `components/contact_mcts.py::_tour_commitment`.

The argmax waypoint has one failure the tour cannot have: **it can pick a cell
we never reach, and nothing releases it.** A commitment is dropped only on
*arrival* or on *vision*, and both require getting there.

| seed | waypoint | turns held | closest the tip came |
| --- | --- | --- | --- |
| 1 | (0,19) — a corner | **853** | 7 |
| 13 | (20,18) | 351 | 12 |

054 tried releasing stalled commitments and it did not help (5, 6, 7 at 40,
100, 200 turns against 8) — the replacement came from the same diffuse belief.
The tour fixes the cause instead: a nearest target is always reachable, so
every commitment ends in an arrival that reveals a 3x3 block and eliminates
candidates, and the next target comes from what is left.

## Result

240 paired seeds, serial, idle machine, four independent 60-seed blocks:

| seeds | baseline | tour |
| --- | --- | --- |
| 0-59 | 20 | 24 (+4) |
| 60-119 | 11 | 15 (+4) |
| 120-179 | 10 | 15 (+5) |
| 180-239 | 17 | 19 (+2) |
| **total** | **58/240 = 24.2%** | **73/240 = 30.4%** |

McNemar exact on the 69 discordant pairs (27 baseline-only, 42 tour-only):
**p = 0.091**. Positive in every block; under the null that is p = 1/16 on
direction alone. Call it **+6 points, not yet significant at 0.05** — the
effect is real enough to ship and too small for 240 seeds to prove.

The mechanism is the one 054 predicted:

| build | sighted | P(win \| sight) |
| --- | --- | --- |
| belief-argmax waypoint | 9/20 (45%) | 78% |
| candidate tour | **11/20 (55%)** | 82% |

054's coverage weighting moved sight the same way (45% -> 55%) and **lost** on
held-out seeds. The difference is that this removes the unreachable-waypoint
deadlock rather than re-ranking the same targets, so the extra sightings arrive
in games still worth winning.

## Note on the baseline

The honest baseline is **24.2%**, not the 42.5% quoted at 053. That figure came
from 40 seeds; seeds 40-239 are harder, and each additional block pulled it
down. Anything measured on 20 or 40 seeds in 046-053 should be read as
directional only.

## Follow-up

`CONTACT_TOUR` is a flag, so `_generate_probe_macros` / `_commit_macro` and the
switch-ratio machinery are still present and still exercised by
`test_contact_commitment.py`. If the tour holds up at larger n, that path
should be deleted rather than left as a second way for the probe to behave.

## Reproduction

```bash
for s in $(seq 0 239); do
  python competition-module/competition/matchup.py \
    bots/sosipolis/run.sh bots/macaria/run.sh --mode competition --seed $s
done
```

Serial, on an otherwise idle machine — see 055 for why.
