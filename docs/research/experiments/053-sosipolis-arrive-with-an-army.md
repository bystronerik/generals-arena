# 053 — sosipolis vs macaria: arriving with 3% of our army

Bot: [`bots/sosipolis/`](../../../bots/sosipolis/) — follows 052.

052 cleared the rungs that were spending turns walking the assault backwards.
This one asks what the assault was carrying when it did arrive.

## The measurement

True general positions come from the engine (`matchup.make_board(env, seed)`,
`state.general_positions`), so "how close did we get" is measurable even in
games where we never sight it. For each of 20 seeds, at the turn where our tip
is physically closest to macaria's general:

| | value |
| --- | --- |
| median tip army | **7** |
| median our total army on the board | **260** |
| median tip share of our army | **3%** |
| games with tip share <= 10% | **12 / 20** |

Split by outcome, the separation is total. Games where macaria's army hits 0 —
i.e. we killed the general — against the rest:

| outcome | tip army at closest approach |
| --- | --- |
| won | 69, 110, 74, 75, 53 (10-15% of ours) |
| lost | 5, 8, 6, 4, 3, 2, 5, 4, 4, 2, 5 (0-3%) |

We were never short of army. We had 150-850 on the board and brought 2-8 of it
to the kill. This is the measured form of the seed-1 GUI observation: *"it's not
even gathering bigger army"*.

Sight is not the binding constraint either — we reached within 8 cells of their
general on 31% of contact turns on seed 1 and within 3 cells exactly once in
788 turns. Visibility is a 3x3 around owned cells, so proximity reveals nothing.

## Cause

`brain.py` ran the tip feed only under `phase == "strike"`, and strike requires
the general to be **sighted** — which never happened in 8 of 20 games and
happened late in the rest. Through the whole contact phase nothing concentrated
army. The mod-50 gather window in contact went to `far_haul_capture` (which
takes land rather than hauling) and then to `ContactMCTS`.

The gather half of the clock did not gather.

## Fix, and why the bar is a fraction

Feed the tip on contact gather ticks. The first version used a constant
(`CONTACT_FEED_TARGET = 35`) and moved the median arrival 7 -> 12 — but games
arriving with 50+ fell from **6 to 1**, because the constant is a cap as well as
a floor, and the 50+ arrivals are the ones that kill generals. Its parameter
sweep was correspondingly chaotic:

| constant | 25 | 35 | 40 | 55 | 80 |
| --- | --- | --- | --- | --- | --- |
| wins /20 | 5 | 8 | 5 | 7 | 3 |

A share of our own army is scale-free — it keeps rising as the board grows, so a
500-army midgame stages a 100-army assault instead of walking in with 35 — and
it behaves far better:

| `CONTACT_FEED_FRAC` | 0.15 | 0.25 | 0.35 |
| --- | --- | --- | --- |
| wins /20 | 8 | 8 | 5 |

Shipped at **0.20**, with a floor of `CONTACT_ASSAULT_STACK` so early contact is
not paralysed.

## Result, on held-out seeds

Seeds 0-19 are the set every parameter above was chosen on, so they cannot
support the claim. Seeds 20-39 were run once, after the choice:

| seeds | 052 committed | + contact feed |
| --- | --- | --- |
| 0-19 (tuned on) | 7/20 | 8/20 |
| **20-39 (held out)** | **5/20** | **9/20** |
| **all 40** | **12/40 = 30%** | **17/40 = 42.5%** |

## Caveats

- The win/loss response to a single parameter is **chaotic**: neighbouring
  constants gave 8 and 5. Individual 20-seed grids are worth about +/-2, so
  treat the 40-seed figure as the number and never tune on a 20-seed delta
  again.
- The feed still has not restored the tail. Median arrival is up, but the 6
  games that used to deliver 50+ are not yet 6 games that deliver 100+. Where
  the next gain is: **what the tip spends on the way in**, not what it leaves
  home with.

## Follow-up that failed: committing the staged stack to a march

The obvious next step, and a wrong one. Over 20 seeds, the turns between the
tip's peak mass and its closest approach to their general split the outcomes:

| outcome | gap (turns) |
| --- | --- |
| won | 1, 3, 4, 23, 24, 72, 80 |
| lost | 14, 29, 30, 55, **104, 115, 123, 173** |

And in the losing games the stack does not die, it *dissolves*: on seed 2 the
tip fell 93 -> 25 across those 173 turns while our own army grew 333 -> 598.

So: latch a staged tip and march it at the objective every tick until spent.
Implemented with `march_dist` routing and a release at `CONTACT_ASSAULT_STACK`.
**3/20.** Reverted.

The correlation is a survivorship artifact. When we win, the game *ends* a few
turns after the peak, so the peak is necessarily near the arrival; when we
lose, play continues for hundreds of turns and the peak is early by
construction. The gap measures game length after the peak, not commitment.
The dissolution is real, but nothing here shows that marching sooner fixes it —
and marching blind at an objective that is often wrong spends the stack on fog.

## Reproduction

```bash
for s in $(seq 0 39); do
  python competition-module/competition/matchup.py \
    bots/sosipolis/run.sh bots/macaria/run.sh --mode competition --seed $s
done
```
