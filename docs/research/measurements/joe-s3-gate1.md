# S3 Gate 1: the depth-1 value re-rank dies offline

**Claim tested** (selection-plan S3, Gate 1): a successor frame fabricated
from the current frame plus one own move carries a value signal a depth-1
re-rank could read. The gate is free and offline by design: fabricate the
successor for the move joe actually played at every corpus turn, compare
`value(fabricated)` against `value(real next frame)`, and kill S3 here if
the fabrication noise drowns the differences a re-rank would read.

**Verdict: killed.** Gate 2 (the slow-fleet x86 budget check) never needed
to run.

## Setup

`bots/joe-rs/tools/s3_gate1.py` over the step-10000 parity corpus
(14 games, 5,632 scored turns; `synthetic-long` excluded). The fabricator
mirrors `competition-module/generals/core/game.py`: split = ⌊A/2⌋, full =
A−1, strict-greater combat with |diff| remaining, build subtracts the live
cost, growth applied after the time increment (all owned cells at
`t % 50 == 0`, structures at `t % 2 == 0`). Held static, per the plan: the
opponent's move and growth, and every fog cell the move does not enter
(a move into fog is modeled as capturing an empty plain; the reveal around
a captured cell is not modeled). Values come from the same net and obs
state the bot would hold, with the fabricated frame augmented on a scratch
copy of the post-frame state — exactly what deployed S3 would do.

## Results

| quantity | p50 | p90 | p99 |
| --- | ---: | ---: | ---: |
| fabrication error \|v_fab − v_real_next\| | 0.0037 | 0.0737 | 0.2927 |
| move-to-move value movement \|v(t+1) − v(t)\| | 0.0051 | 0.0284 | 0.1040 |
| top-2 fabricated value gap, all turns | 0.0039 | 0.0257 | 0.1218 |
| top-2 gap, near-tie turns (margin < 0.1) | **0.0000** | 0.0092 | 0.0523 |
| top-2 gap, near-tie **distinct** successors | 0.0033 | 0.0223 | 0.0897 |

corr(v_fab, v_real_next) = **0.9823** pooled (per-game 0.879–0.9997; the
minimum is a joe mirror, where the opponent-static assumption bites
hardest). Near-tie turns are **5.21%** of the corpus — the plan's ~5%
claim, confirmed. **68.7%** of near-tie top-2 pairs fabricate
**byte-identical successors** (the half/full twins of one move: from a
2-army cell both push 1 unit).

## Reading

The high correlation is the game-scale value trend, not per-move
discrimination — the re-rank reads within-turn differences, and those are
the numbers that matter:

1. **On the turns S3 would act (near-ties), there is mostly nothing to
   rank.** Two thirds of the top-2 pairs are the same action in effect;
   no evaluator could or needs to split them.
2. **Where there is something to rank, the signal sits at the noise
   floor.** The median distinct-pair gap (0.0033) is below the median
   fabrication error (0.0037); the sign of such a gap is close to a coin
   flip.
3. **The tail drowns.** p90 fabrication error (0.0737) is 2.6× the p90
   move-to-move movement (0.0284): exactly the "fabrication noise drowns
   typical move-to-move value differences" condition the plan set as the
   kill criterion.

This also confirms the plan's stated suspicion: the value head, trained on
real trajectories, is near-indifferent between adjacent candidate moves.
The confinement diagnostic showed it knew the *game* was being lost, not
that move A beat move B — and per-move discrimination is what a re-rank
needs.

The fabricator is not the culprit: median error 0.0037 on a ±1 scale and
0.9997 correlation on the quietest game bound any systematic modeling bug
to be far below the decision threshold. The tail error lives where the
plan chose to hold the world static — opponent moves and fog reveals —
which is inherent to depth-1 fabrication, not fixable within it.

## Consequence

S3 is dead in all its variants (plain top-k and margin-gated both read the
same per-move value differences). With S2 already dead (its trigger was an
S1 decisive-play cost that [joe-selection-s1](joe-selection-s1.md) did not
show) and S1 proven flat, **the selection layer is complete: S1 Gumbel
T = 1 is the shipped end state.** Any future value-guided selection needs
a value head trained to discriminate moves (a training project, out of
joe-rs scope per the plan's search assessment), not a better fabricator.

The gate tool stays (`tools/`, outside the content hash) — it is the
cheap first check for any future one-step-lookahead idea against this
lineage's checkpoints.
