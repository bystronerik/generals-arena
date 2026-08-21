# joe-rs soft trail penalty (S4)

Anti-circuit penalty for the joe-rs selection layer — the follow-up to the
S1 oscillation residual measured on 2026-08-21.

## The measured problem

On the diagnostic seed (macaria vs joe-rs, seed 1, joe-rs seat 1), joe-rs
wins where argmax joe lost, but 8.8% of its moves immediately undo the
previous move, peaking at **26% of moves in turns 300–400**, with the
moving stack confined to ~11 source cells while owning 83. The undos are
**not** Gumbel re-flips: 86% of return legs are the argmax itself at a
median top-2 margin of **1.85 logits**. The network's preference genuinely
oscillates as growth ticks and the stack's own position perturb the board;
training's sampling masked this, so the gradient never met it.

**The v1 arc penalty displaced, not dissolved.** The first cut taxed only
the exact inverse of last turn's move. Its grid halved the undo rate — and
seed-1 play then showed **period-4 circles** (observed 2026-08-21): the
walk routed around the one taxed arc. Any fix must catch circuits of any
short period, and it must catch them **fast** — a trailing-window detector
(the S2 confinement shape) needs ~50 turns to fire, which is 50 lost
steps per episode.

## Mechanism

Remember the last `K` move **source** cells (the stack's trail — a pass or
a build pushes nothing, and a pause forgives nothing). Before Gumbel
selection, subtract `δ` from every move that would **land on a trail cell
that is still ours** — all four approach directions, full and half
channels. A twice-departed cell is taxed once, never stacked.

Why this catches everything the symptom contains, at zero detector lag: a
circuit of period ≤ K must land on its own trail at the closing move,
while a forward march never re-enters its trail — so normal expansion and
corridor marches are untouched, and a forming circle is taxed at the exact
move that would close it, the first time.

**Two guard rules (load-bearing, each pinned by a unit test and a
mutation plant):**

- *Retake:* departing always leaves army behind, so a trail cell stays
  ours unless the enemy captures it. If it flips, the tax lifts with the
  ownership bit — **retaking a lost cell is combat, not shuffling**, and
  it fights at full logit.
- *Encirclement:* a source whose every other passable exit is also taxed
  trail — mountains, the board edge, or trail on all remaining sides —
  has only the step back, and **the only way out is never taxed**. A
  stack boxed in by hills retreats at full logit instead of dithering
  against a flat tax until the window ages out.

Contrast with the removed Python penalty (the measured-bad family): that
one distorted a cell's whole 10-channel action column on every revisit
with decaying counters, always-on. This taxes only re-entry arcs into a
bounded trail, flat, with the ownership guard, and composes with the S1
noise (a genuinely necessary return whose margin beats `δ` still wins the
draw).

## Knobs and sizing

`JOE_RS_NOUNDO` (δ, f32, **default 0 = off**) and `JOE_RS_NOUNDO_WINDOW`
(K, usize, default 8, 0 disables the memory). δ must beat the measured
~1.9-logit return margins; the shortlist arms are δ = 2 and δ = 4. K = 8
covers the observed period-2/4 orbits with margin; K = 1 degenerates to
(a slightly stronger form of) the v1 arc penalty. Per the repo rule no
constant ships by hand: the unrated diagnostic grid
(`bots/joe-rs/tools/osc_grid.py`) shortlists, a rated round against the
frozen δ=0 bot prices the winner per
[decision-rule](../../arena/decision-rule.md) before any default flips.

## Diagnostic grids (2026-08-21, unrated, seeds 0–4 × both seats vs macaria)

**v1 arc penalty** (superseded — shown for the displacement evidence):

| δ | wins | mean end | undo% | strict-cycle% | worst confinement |
| ---: | :---: | ---: | ---: | ---: | ---: |
| 0 | 10/10 | 423 | 2.9 | 5.9 | 11 |
| 2 | 10/10 | 392 | 1.6 | 5.8 | 17 |
| 4 | 10/10 | 429 | 0.9 | 5.0 | 18 |

Undo down, strict-cycle coverage flat — the circles changed period instead
of dying, which is what seed-1 play showed.

**v2 trail penalty without the encirclement exemption** (superseded;
δ ∈ {2, 4}) exposed the **saturation failure mode**: at δ = 2, one game
(seed 4 seat 0) collapsed to 6 distinct sources with period-2/4 spans
back — when the stack is boxed in so tightly that *every* candidate move
lands on the trail, a flat tax cancels within the pocket while still
suppressing those moves against pass/build, and the stack dithers. The
exemption (and the retake guard) came out of exactly this observation
plus review.

**v3 trail penalty, both guards (K = 8)** — `revisit%` is the direct
any-period symptom (moves landing on one of the last 8 sources); the
cycle metric counts move spans only:

| δ | wins | mean end | undo% | revisit% | move-cycle% | worst confinement |
| ---: | :---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 10/10 | 423 | 2.9 | 10.8 | 0.8 | 11 |
| 4 | 10/10 | 392 | 2.0 | 7.6 | 0.3 | 14 |
| 6 | 10/10 | **355** | **1.0** | **4.8** | **0.0** | **19** |

**Reading: δ = 6 is the shortlist pick.** Monotone improvement on every
oscillation metric, zero strict move-cycles anywhere, no confinement
collapse, and the fastest conversions (mean −68 turns vs baseline). The
diagnosed game (seed 1 seat 1) goes from 28.5% revisit / 8.8% undo /
confinement 11 at δ = 0 to **4.7% / 0.4% / 24 with zero cycle spans,
winning 65 turns sooner** (t = 532 vs 597). δ = 4 still shuffled on that
trajectory (18.1% revisit) — its tax sits below some of the measured
return margins.

Single games are chaotic in these knobs (the repo's standing warning), so
the grid shortlists only; strength needs the rated round. **Round
`s4-trail-r1` ran (2026-08-21): both arms proven flat against the δ = 0
baseline — [joe-s4-trail-round](../measurements/joe-s4-trail-round.md) —
so the behavioral gains are free; the default flips to δ = 6 after the
r2 replication.** Latency is unchanged (`bench`
p50 26.65 ms with the penalty on, vs the 26.2–26.6 ms reference — the
tax touches at most 8 × K logit entries).

## Properties preserved

- The bot stays a deterministic function of the frame stream (a bounded
  ring of its own past sources; replays and self-goldens reproduce byte
  for byte). With the default off, shipped behavior and the recorded
  goldens are unchanged.
- The network path, parity tiers, and the `decide` surface are untouched.
- Escalation if the grid shows displacement into longer wander orbits
  (period > K): the S2 confinement detector — not a bigger δ, and not an
  unbounded window.
