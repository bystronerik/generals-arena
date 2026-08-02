# sosipolis-contact-p2-diag — ContactMCTS hunt ownership (macro MCTS)

Date: 2026-08-02  
Bot: same belief + sticky macros as p1; path layer = multi-depth macro MCTS  
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
| contact-p1-diag (shallow) | 4/10 (40%) | 82.5 | 447 | 373 | 3-7 |
| **contact-p2-diag (macro MCTS)** | **1/10 (10%)** | **84.0** | **487** | **421** | **1-9** |

Per-game contact: 100, 82, 66, 86, 101, 88, 61, 58, 87, 79.

Extra (p2 trajectories):

| Signal | Value |
| --- | ---: |
| Median distinct contact waypoints | 5.0 |
| Median contact switches | 4.0 |
| Median commitment age (per-game median) | 21.0 |
| Macro kinds | cluster 1223, split 731, mid_edge 703 |

## Verdict

**Phase 2 loses to Phase 1 on this panel.** Sight rate falls back to 10%; wins
fall to 1-9. Keep `CONTACT_PATH_MODE=shallow`. Contact→sight remains far from
≤95.5 under both path layers — next work is belief/probe quality, not deeper
path search.

## Reproduce

```bash
# Temporarily set CONTACT_PATH_MODE = "macro_mcts" in bots/sosipolis/params.py
python -m arena.tournaments.competition \
  bots/sosipolis/run.sh bots/macaria/run.sh \
  --round sosipolis-contact-p2-diag \
  --seeds 0-4 --seat-policy alternate --no-ratings --record --jobs 4
PYTHONPATH=. python scripts/analyze_sosipolis_sight.py --round sosipolis-contact-p2-diag
```
