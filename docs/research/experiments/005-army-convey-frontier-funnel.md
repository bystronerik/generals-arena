# 005 — army_convey: interior-to-frontier funnel

Bot: [`bots/army_convey/`](../../../bots/army_convey/). Baseline: `smoke`, `expand_plus`.

## Hypothesis

`smoke` and `expand_plus` leave interior armies idle when no adjacent
unowned cell is capturable. `expand_plus` only marches the single largest
stack globally. Continuously conveying all interior stacks along a
frontier distance field should concentrate army force at expansion tips,
raising captured land and winrate against static expanders.

One change cluster: frontier-mask detection, per-turn conveyance scoring
(`army / (distance + 1)`), and frontier gathering before fallback.

## Seed grid

- Opponents: `smoke`.
- Seeds: 0 (verification gate).
- Mode: `--mode competition` only.

## Metrics

| Matchup | Games | army_convey W-L-D | Mean turns |
| --- | --- | --- | --- |
| army_convey vs smoke | 1 | 1-0-0 | 409 |

Seed 0: army_convey (player 0) captured smoke's general on turn 409. Match
finished cleanly, no faults, 0 castles built.

## Decision

**Keep**, provisionally. Verification gate passed; decisive win vs.
`smoke` supports the frontier-funnel hypothesis on this seed.
