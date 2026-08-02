# sosipolis-sight3 — contact→sight tempo (clocked shell)

Date: 2026-08-02  
Bot: clocked Kubic shell + Search/Contact tempo revision  
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
| sight1 (tip4) | 1/10 (10%) | 130.5 | 592 | 467 | — |
| sight2 (tip-feed unlock) | 0/10 (0%) | 132.0 | — | — | — |
| **sight3 (this)** | **1/10 (10%)** | **80.0** | 503 | 437 | **1-9** |
| Kubic wins | ~100% | 82 | 181.5 | 95.5 | — |

Per-game contact: 102, 82, 66, 85, 81, 83, 61, 59, 79, 79.
Only seed 1 seat A sighted (turn 503; contact→sight 437) and won.

Extra (sight3 trajectories):

| Signal | Value |
| --- | ---: |
| toward_frac search | 0.86 |
| toward_frac contact→sight | 0.67 |
| MCTS override rate contact→sight | 0.006 |
| Median contact window (to sight or end) | 242.5 |
| Median distinct hunts in window | 12.5 |

## Verdict

**Contact tempo is fixed.** Median contact **80** meets the Kubic ≤82 gate
(was ~130–132 on sight1/sight2).

**Sight conversion is still broken.** Sight rate stays **10%**; the one sight
is late (503) with contact→sight **437** (target ≤95.5). Long contact windows
(~240 turns) with many hunt retargets and almost no MCTS override mean the
prior chain marches under clock masks but does not close on the general.

Gate: smoke seed 0 — sosipolis win at turn 300, first castle 128 (≥116),
no faults.

## Reproduce

```bash
python -m arena.tournaments.competition \
  bots/sosipolis/run.sh bots/macaria/run.sh \
  --round sosipolis-sight3 \
  --seeds 0-4 --seat-policy alternate --no-ratings --record --jobs 4
PYTHONPATH=. python scripts/analyze_sosipolis_sight.py --round sosipolis-sight3
```
