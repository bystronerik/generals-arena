# 007 — late_rush: timed commitment

Bot: [`bots/late_rush/`](../../../bots/late_rush/). Baseline: `smoke`, `expand_plus`.

## Hypothesis

`smoke` and `expand_plus` never commit a main stack toward the enemy on a
fixed schedule. An accumulation window followed by an irreversible turn-700
rush should produce more decisive games and more wins than passive
expanders, especially when the rush reaches enemy territory before turn 800.

One change cluster: rally-cell selection, main-stack designation,
commitment turn thresholds (650 / 675 / 700), rush scoring toward a
selected target, and deathtouch contact priority after turn 800.

## Seed grid

- Opponents: `smoke`.
- Seeds: 0 (verification gate).
- Mode: `--mode competition` only.

## Metrics

| Matchup | Games | late_rush W-L-D | Mean turns |
| --- | --- | --- | --- |
| late_rush vs smoke | 1 | 1-0-0 | 685 |

Seed 0: late_rush (player 0) captured smoke's general on turn 685. Match
finished cleanly, no faults, 0 castles built.

## Decision

**Keep**, provisionally. Verification gate passed; decisive win vs.
`smoke` supports the timed-commitment hypothesis on this seed.
