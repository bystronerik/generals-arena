# 008 — splitter: half-army multi-front capture

Bot: [`bots/splitter/`](../../../bots/splitter/). Baseline: `expand_plus`, `smoke`.

## Hypothesis

Adding a split-flag decision layer on top of `expand_plus` greedy capture
lets large stacks capture with half army while garrisoning the source or
opening a second front. This should convert some 1200-turn draws into wins
without increasing faults or losses.

One change only: split scoring plus the guards in
[`docs/research/strategies/splitter.md`](../strategies/splitter.md). The
BFS frontier-march fallback is unchanged.

## Seed grid

- Opponents: `smoke`, `expand_plus`.
- Seeds: 0, 1, 2.
- Mode: `--mode competition` only.

## Metrics (initial verification)

| Matchup | Games | splitter W-L-D | Mean turns |
| --- | --- | --- | --- |
| splitter vs smoke | 1 | 0-0-1 | 1200.0 |

Seed 0 vs `smoke`: draw at 1200 turns, no faults, no truncation fault.
Game stored at `data/games/20260730T233120Z_splitter_vs_smoke_s0_447c341d.json`.

Full seed grid (vs `smoke` and `expand_plus`, seeds 0–2) not yet run.
Land/army growth from efficient capture is not in the current game-record
schema — same gap as experiment 001.

## Decision

**Keep**, provisionally. No regression on the verification gate (finishes
clean, no faults). Run the full seed grid before rating updates. Revert if
losses or faults appear on any seed.
