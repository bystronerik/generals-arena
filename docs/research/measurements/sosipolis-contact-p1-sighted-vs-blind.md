# contact-p1 sighted vs blind trace contrast

Date: 2026-08-02  
Round: [`sosipolis-contact-p1-diag`](../measurements/sosipolis-contact-p1-diag.md)  
(4 sighted / 6 blind sosipolis seats)

## Main contrast

| Signal (contact window median) | Sighted | Blind |
| --- | ---: | ---: |
| Contact turn | 82.5 | 82.5 |
| Tip army (median) | **16.5** | **9.5** |
| Tip army (max) | 97 | 42.5 |
| Tip ready frac | 0.17 | 0.07 |
| Toward frac | 0.82 | 0.84 |
| Distinct waypoints | 10 | 6 |
| **Macro mix: cluster** | **58%** | **33%** |
| **Macro mix: split** | **16%** | **59%** |
| Macro mix: mid_edge | 26% | 8% |

Contact clock and toward-rate do **not** separate the groups. Macro kind does.

## Per-game split fraction (contact turns)

| Tag | seed/seat | split_frac | recall turns | tip@goal without sight |
| --- | --- | ---: | ---: | ---: |
| SIGHT | 3b | **0.06** | 0 | 47 |
| SIGHT | 0b | 0.17 | 0 | 43 |
| SIGHT | 4a | 0.17 | 56 | 0 |
| SIGHT | 1a | 0.25 | 0 | 65 |
| BLIND | 3a | 0.32 | 2 | 28 |
| BLIND | 2b | 0.46 | 3 | 34 |
| BLIND | 2a | 0.48 | 75 | 82 |
| BLIND | 1b | **0.64** | **514** | 35 |
| BLIND | 4b | **0.72** | 3 | 16 |
| BLIND | 0a | **0.90** | 6 | 112 |

## Named defects

1. **Split-macro dominance (primary).** Blind games spend most contact turns on
   `split` waypoints. Sighted games stay on `cluster` (+ some `mid_edge`).
2. **False waypoints.** Tip often reaches `tip_dist_goal==0` without sighting.
   The committed cell is usually not the enemy general; the tip parks on empty fog.
3. **Thin tip on blinds.** Median tip army ~9.5 vs ~16.5 when sighted.
4. **Recall thrash (one game).** seed1 seat B: 514 recall vs 95 MCTS turns.

## What not to do next

- Do not re-run 030’s broad push/stickiness bundle (already 0% sight).
- Do not deepen fog belief alone (031 already 0% sight).
- Do not disable `mid_edge` together with `split` in the same arm (030 did both).

## Next revision (one defect)

Disable **`split` macros only**. Keep `cluster` / `mid_edge`, land bonuses, tip
hold, and sticky thresholds unchanged. Measure sight rate vs contact-p1 on the
same seeds 0–4 alternate panel.
