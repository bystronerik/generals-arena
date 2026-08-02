# sosipolis-contact-p1-diag — ContactMCTS hunt ownership (shallow)

Date: 2026-08-02  
Bot: ContactMCTS owns post-contact probe (belief + sticky macros + shallow scorer)  
Opponent: `macaria`  
Grid: seeds 0–4, `--seat-policy alternate`, `--record` (10 games)  
Analyzer: [`scripts/analyze_sosipolis_sight.py`](../../../scripts/analyze_sosipolis_sight.py)

## Targets (Kubic wins)

| Metric | Target |
| --- | ---: |
| Contact median | ≤ 82 |
| Sight median (when sighted) | ≤ 181.5 |
| Contact→sight median | ≤ 95.5 |
| Sight rate | ~100% in wins |

## Results

| Round | Sight rate | Median contact | Median sight (if any) | Median contact→sight | W-L |
| --- | ---: | ---: | ---: | ---: | ---: |
| sight3 | 1/10 (10%) | 80.0 | 503 | 437 | 1-9 |
| **contact-p1-diag** | **4/10 (40%)** | **82.5** | **447** | **373** | **3-7** |

Per-game contact: 99, 82, 66, 85, 97, 80, 61, 87, 83, 79.

Extra (p1 trajectories):

| Signal | Value |
| --- | ---: |
| Median distinct contact waypoints | 7.5 |
| Median contact switches | 6.5 |
| Median commitment age (per-game median) | 18.5 |
| Macro kinds | cluster 1342, split 1203, mid_edge 486 |

## Verdict

**Sight conversion improved** (10% → 40%) and wins rose (1 → 3). Contact tempo
stays near the ≤82 gate (82.5). Contact→sight improved (437 → 373) but remains
far from ≤95.5. Sticky commitment reduces waypoint churn vs sight3 (12.5 → 7.5).

## Reproduce

```bash
python -m arena.tournaments.competition \
  bots/sosipolis/run.sh bots/macaria/run.sh \
  --round sosipolis-contact-p1-diag \
  --seeds 0-4 --seat-policy alternate --no-ratings --record --jobs 4
PYTHONPATH=. python scripts/analyze_sosipolis_sight.py --round sosipolis-contact-p1-diag
```
