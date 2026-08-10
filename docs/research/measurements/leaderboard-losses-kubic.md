# Kubic loss profile

Part of [leaderboard-top-loss-analysis.md](leaderboard-top-loss-analysis.md);
method, corpus, and caveats live there. Observational, read-only.

104 played losses vs 3,536 played wins — the strongest of the three players.
ResBot alone accounts for 44 losses (42%); the whole top cluster for 73 (70%).
Kubic dominates everyone else (166–7 vs nanomena, 181–7 vs thor, 166–12 vs
bca) and loses only up: 25–44 vs ResBot.

Drill-down sample (12, by match id): 40409, 96219, 148039, 153088, 139385,
152219, 177676, 77773, 194957, 176533, 178483, 153084.

## Batch aggregate (losses)

| metric | value |
| --- | --- |
| median ticks | 348 (wins: 252) |
| median gather waves | 6.5 |
| median toward-fraction | 0.82 (wins: 0.84) |
| never saw enemy general | 78/104 (75%) |
| median closest approach | 4 |

## How the losses go

- **Openings are fine.** Kubic has 7–8 tiles at t24 and ~22 at t48 in every
  sampled loss — it does not lose in the opening.
- **The grind (41% ≥450 ticks, mostly ResBot).** Kubic matches ResBot wave
  for wave (36 vs 30 in 153084) and even matches its toward-fraction
  (0.83–0.88 both sides in the ResBot samples) — what it cannot match is
  land: stall ≤77 tiles t184–1145 in 153084, ≤75 t375–973 in 178483, ≤81
  t364–781 in 176533, while ResBot keeps gaining. Against ResBot, Kubic
  loses on economy alone.
- **The mid-game break (e.g. 152219 vs thor).** Parity to t160, thor takes a
  central castle fight and a 25-army big capture at t184, Kubic's land peaks
  at t258 and never recovers; irreversible ~60 ticks before the end.
- **Losing while fully informed (177676 vs bca).** Kubic sighted bca's
  general at t88, knew it for 308 ticks, aimed 11 of its 12 waves at it —
  and still lost 78 tiles to 145. Intel without an economy behind it did not
  help. This is the strongest single counterexample to "sighting the general
  is what wins".
- **Fast collapses (31% ≤250 ticks).** 96219 (vs bca): zero gather waves in
  149 ticks — expanded and defended, never assembled an attack, was 61 vs
  136 tiles when it died.

## When it became irreversible

Median last army-parity (≥80%) tick in the sample: ~t300 of median-395-tick
games. The tell is the pair (`expansion_stall` for Kubic, continued tile gain
for the opponent) — present in 8 of 12 sampled losses; once both hold for
~50 ticks no sampled game turns around.

## What the winning opponent did differently

Not more waves, not earlier castles (both ~t120–140), and — against ResBot —
not even straighter pressure. They kept expanding during the fight (ResBot
141→166 tiles across Kubic's stall in 153084) and held the land-and-army
ratio above 1.2×/1.4× from mid-game on. In the punch losses the winner's
directness does show up (thor 0.90 vs Kubic 0.75 in 152219; bca 0.91 vs
0.68 in 96219): when Kubic loses fast, it is the side wandering.
