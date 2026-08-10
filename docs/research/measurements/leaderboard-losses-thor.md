# thor loss profile

Part of [leaderboard-top-loss-analysis.md](leaderboard-top-loss-analysis.md);
method, corpus, and caveats live there. Observational, read-only.

217 played losses vs 1,342 played wins — the most losses and the highest
share against the top cluster (175/217, 81%): 6–54 vs ResBot, 3–37 vs Kubic,
41–49 vs nanomena.

Drill-down sample (12, by match id): 112499, 153055, 99773, 152223, 187818,
179350, 163832, 159813, 159811, 180354, 152274, 153148.

## Batch aggregate (losses)

| metric | value |
| --- | --- |
| median ticks | 440 (wins: 324) |
| median gather waves | 7 (1.5 per 100 ticks) |
| median toward-fraction | 0.81 (wins: 0.87) |
| never saw enemy general | 162/217 (75%) |
| no-contact games | **4** — bot-failure shape, see below |

## How the losses go

- **Bot failures.** 112499: thor held 1 tile for all 56 ticks and never
  moved while eggtart expanded to 30 — a no-op game, not play. 4 of 217
  losses have no contact at all; they belong in a reliability bucket, not a
  strategy one.
- **The punch (24% ≤250 ticks, mostly nanomena/Kubic).** 153055 is the
  archetype: tile parity at t96 (40 vs 45), nanomena assembles a 37-stack at
  t116 against thor's 14–36, takes thor's fresh castle at t132, and drives
  blind at the general — dgen already 9 by t128, sight at t170, capture
  t171. Dead ~70 ticks after first contact.
- **The grind (49% ≥450 ticks, mostly ResBot/bca).** Same shape as the
  others but with the largest deficits: winner/loser army ratio at the last
  contested frame is median 1.83 (p75 2.50), land 1.48. In 152274 thor
  stalled ≤86 tiles for 372 ticks; in 153148 it knew ResBot's general for
  766 ticks, aimed 4 waves at it, and never got through a 981-army wall.
- **Stall in every sampled loss.** `expansion_stall` fires for thor in 12 of
  12 sampled played losses — the most consistent stall signature of the
  three players.

## When it became irreversible

Median last army-parity tick ~t290 of median-430-tick sampled games; in the
punches, within ~50 ticks of first contact — the first oversized enemy wave
is already the loss.

## What the winning opponent did differently

Concentrated earlier (in 153055 nanomena's 37-army breakthrough wave faced
thor stacks of 11–17 at the line) and kept gaining land through the fight —
toward-fraction barely separates the seats in thor's losses (winners ~0.83
to thor's ~0.80), so this is a concentration-and-economy loss, not a
wandering one. Castle
timing again does not separate the seats — but in the punches the *loser's*
fresh castle becomes the winner's forward base (153055 t126 build, t132
captured), so an undefended early castle near the frontier reads as a
liability.
