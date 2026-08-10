# bca loss profile

Part of [leaderboard-top-loss-analysis.md](leaderboard-top-loss-analysis.md);
method, corpus, and caveats live there. Observational, read-only.

135 played losses vs 919 played wins. Loses down as well as up: 2–21 vs
Kubic, 9–33 vs ResBot, 24–27 vs nanomena — the weakest head-to-head record
of the three.

Drill-down sample (12, by match id): 152642, 164171, 198778, 108699, 108680,
108636, 152235, 194967, 148205, 162256, 148179, 153167.

## Batch aggregate (losses)

| metric | value |
| --- | --- |
| median ticks | **577** — the longest losses of the three (wins: 279) |
| median gather waves | 7, but only **1.3 waves per 100 ticks** (Kubic: 2.0) |
| median toward-fraction | 0.81 (wins: 0.86) |
| never saw enemy general | 98/135 (73%) |
| median peak stack in top-cluster losses | 141 — the biggest stacks of the three |

## How the losses go

- **72% are grinds (≥450 ticks)** — bca does not get punched often; it gets
  starved. It hoards its opening (1 tile at t24 in all sampled losses,
  catching up to ~22 by t48), survives contact, then stalls.
- **Passivity is the signature.** In 849-tick 162256 bca ran 2 gather waves
  to nanomena's 23. In 1035-tick 148179 it ran 4 to ResBot's 3 — but with
  toward-fraction 61%, the lowest in the whole sample, despite knowing
  ResBot's general location for 897 ticks.
- **Big stacks that don't convert.** bca builds the largest piles of any
  loser (peak 279 in 153167, 317 at one frame) and spends them on the
  nearest visible tile or a contested castle, not on a push. In 153167 the
  central castle at (6,10) changed hands **eight times** while bca's land
  sat ≤94 tiles for 784 ticks and ResBot's rose 141→166. Sunk-cost castle
  fighting while losing the map is the cleanest single mechanism in the
  sample.
- **Fast collapses exist but are rare** (14%): 152642 died in 86 ticks to a
  nanomena rush with zero gather waves of its own.

## When it became irreversible

Latest of the three players in absolute ticks (median last army-parity
~t480) but *earliest as a fraction of the game* — the median 577-tick loss
is decided with a third of it still to play. The ResBot games continue
hundreds of ticks past any plausible recovery.

## What the winning opponent did differently

Kept expanding while bca fought (every sampled winner's tile count rose
through bca's stall window); pressed at 94–95% toward-fraction (ResBot in
153167) against bca's 81–83%; and — against the hoard opening — was simply
on the map earlier: sampled winners held a median 7 tiles at t24 to bca's 1
(4 of 12 also hoarded), and bca's t48 catch-up never translated into a
contact-time lead.
