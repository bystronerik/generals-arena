# 001 — expand_plus: frontier-directed march

Bot: [`bots/expand_plus/`](../../../bots/expand_plus/). Baseline: `expander_python`, `smoke`.

## Hypothesis

Greedy capture (`expander_python`, `smoke`) falls back to an arbitrary
"first legal move" when no adjacent tile is capturable this turn. Replacing
that fallback with a BFS march — move the largest owned stack one step
toward the nearest capturable tile — should raise land/army growth and
winrate, because idle turns stop wandering and start consolidating toward
the frontier.

One change only: the fallback move when no capture is available. The
primary greedy-capture scoring is unchanged from `expander_python`.

## Seed grid

- Opponents: `smoke`, `expander_python`.
- Seeds: 0, 1, 2.
- Mode: `--mode competition` only.

## Metrics (from `data/games/`, see tournament summary)

| Matchup | Games | expand_plus W-L-D | Mean turns |
| --- | --- | --- | --- |
| expand_plus vs smoke | 3 | 0-0-3 | 1200.0 |
| expand_plus vs expander_python | 3 | 0-0-3 | 1200.0 |

All 6 games ran the full 1200-turn cap and drew. Under this seed grid,
neither side's expansion ever reaches the other's general through fog of
war, so this experiment cannot show a winrate delta — the mechanism itself
was unit-tested separately (BFS gradient picks the correct march step when
no capture is adjacent) but full-game telemetry here only confirms it does
not destabilize the match (still finishes cleanly, no faults, no crash).

## Decision

**Keep**, provisionally. No regression vs. baseline (still draws cleanly
every seed tried), and the fallback is strictly more directed than "first
legal move". Land/army totals at truncation were not captured by the
current game-record schema (only winner/turns/terminated/truncated) — a
follow-up would need per-turn or final-state telemetry to measure the
intended effect (land/army growth rate) rather than only winrate. Revert
only if a later run shows worse decisive-outcome or fault rates.
