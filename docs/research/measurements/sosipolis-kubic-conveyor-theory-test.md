# Sosipolis pure-conveyor theory test (reverted)

Date: 2026-08-02  
Behavioural source: [`../strategies/kubic-behavior-spec.md`](../strategies/kubic-behavior-spec.md)  
Grid report: [`sosipolis-kubic-vs-macaria-10.md`](sosipolis-kubic-vs-macaria-10.md)

## Hypothesis

A rule-based Kubic conveyor (no live Search / Contact / Strike MCTS) can match
enough of the observed Kubic decision order to compete with `macaria`.

Locked proxies for UNKNOWN scorers: nearest never-seen frontier (BFS),
`MapMemory.hunt_target` after contact, `RECALL_PROX_D=3`,
`TIP_AT_SIGHT_FLOOR=10`, `CASTLE_EARLIEST=116`, `OPEN_FLOOD_START=27`.

## What was built (then reverted)

Live `bots/sosipolis/brain.py` priority stack:

1. lethal kill  
2. PASS only when no leave-1 move  
3. rare tip recall  
4. opening ≤50  
5. castle ≥116  
6. post-sight tip floor then march  
7. mod-50 gather `[10,27]` / wave `[28,49]∪[0,9]`

New modules (removed on revert): `components/opening.py`, `conveyor.py`,
`recall.py`. MCTS files were kept on disk during the test but unused on the
live path.

Smoke gate: sosipolis vs smoke, `--mode competition --seed 0`, finished with a
sosipolis win (turn 296); first castle at 152 (≥116).

## Result vs macaria

| Item | Value |
| --- | --- |
| Round | `sosipolis-kubic-vs-macaria-10` |
| Games | 10 (seeds 0–9, random seats) |
| Outcome | **macaria 10–0** |
| Draw rate | 0% |
| Mean turns | 238 |
| Fastest loss | seed 7, 83 turns |

Contrast: prior MCTS sosipolis tip5 grid vs macaria was **7–33** (17.5% WR) on
40 games ([`sosipolis-tip5.md`](sosipolis-tip5.md)). The pure conveyor scored
**0/10**.

## Verdict

The pure conveyor **failed** this theory test against `macaria`. Matching the
kubic priority shell and named clocks was not enough to reach competitive
outcomes. Restore work should keep MCTS (or another planner) on the live path
and treat the kubic spec as behavioural targets / constraints, not as a
complete rule-only policy.

## Revert

`bots/sosipolis/` was restored to the pre-conveyor tree (triple MCTS + tip feed).
Strategy spec [`../strategies/sosipolis.md`](../strategies/sosipolis.md) was
restored with the bot. Raw games remain under
`data/games/sosipolis-kubic-vs-macaria-10/` (gitignored). Do not feed
leaderboard replays into `data/games/` or ratings.
