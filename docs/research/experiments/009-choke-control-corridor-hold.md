# 009 — choke_control: corridor claim and hold

Bot: [`bots/choke_control/`](../../../bots/choke_control/). Baseline: `expand_plus`, `smoke`.

## Hypothesis

Choke-aware capture scoring plus hold/deny rules on `expand_plus` should
lose less land to opponent expansion on corridor-rich maps and convert
opponent attacks into failed pushes. On open maps the bot should behave
like `expand_plus` (choke scores near zero).

One change only: choke detection, scoring, and hold/deny per
[`docs/research/strategies/choke_control.md`](../strategies/choke_control.md).
The BFS frontier-march fallback is unchanged except for a reinforce-toward-
held-choke preference under threat.

## Seed grid

- Opponents: `smoke`, `expand_plus` (add `splitter` when available).
- Seeds: 0, 1, 2.
- Mode: `--mode competition` only.

## Metrics (initial verification)

| Matchup | Games | choke_control W-L-D | Mean turns |
| --- | --- | --- | --- |
| choke_control vs smoke | 1 | 0-0-1 | 1200.0 |

Seed 0 vs `smoke`: draw at 1200 turns, no faults, no truncation fault.
Game stored at `data/games/20260730T233129Z_choke_control_vs_smoke_s0_d5b2ccb9.json`.

Full seed grid not yet run. Final land/army telemetry is not in the
game-record schema — same gap as experiments 001 and 008.

## Decision

**Keep**, provisionally. No regression on the verification gate (finishes
clean, no faults). Run the full seed grid and synthetic detector tests
before rating updates. Revert or retune `W_CHOKE` if losses or faults
appear.
