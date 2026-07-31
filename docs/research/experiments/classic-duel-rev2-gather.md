# Experiment: classic_duel parameter revision 2 — gather concentration

Date: 2026-07-31  
Bot: `classic_duel`  
Harness: `arena/classic_match.py` (seeds 0–4, both seats)  
Opponents: `army_convey`, `expand_plus`, `smoke`

## Change

| Constant | Old | New |
| --- | --- | --- |
| `GATHER_DEST_ARMY_WEIGHT` | — (implicit 0) | 1.0 |

`_frontier_gathering` scores `src_army + dest_army * GATHER_DEST_ARMY_WEIGHT`
instead of `src_army` alone.

## Results

| Opponent | W | L | D | Winrate |
| --- | ---: | ---: | ---: | ---: |
| `army_convey` | 6 | 4 | 0 | 60% |
| `expand_plus` | 9 | 0 | 1 | 90% |
| `smoke` | 10 | 0 | 0 | 100% |
| **Total** | **25** | **4** | **1** | **83.3%** |

Baseline after revision 1: 24-5-1 (80.0%); vs `army_convey` 5-5 (50%).

## Verdict

**Keep.** +1W −1L on total grid; +1W −1L vs `army_convey`. No smoke regression.
Expand_plus unchanged. Broader convey-weight experiments (FRONTIER_NEIGHBOR 2.0+,
CONVEY_MIN 2) regressed to 30–40% vs `army_convey` and were reverted.

Full grid: [`classic-duel-stress.md`](../measurements/classic-duel-stress.md).
