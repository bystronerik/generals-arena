# 004 — fog_scout: systematic fog probing

Bot: [`bots/fog_scout/`](../../../bots/fog_scout/). Baseline: `smoke`, `expand_plus`.

## Hypothesis

`smoke` and `expand_plus` ignore fog tiles (`type == 0`) when scoring
expansion moves. Actively incentivizing fog entry and marching idle stacks
toward never-seen terrain should reveal the enemy general earlier, giving
actionable targeting data before turn-800 deathtouch.

One change cluster: persistent `ever_seen_grid`, fog score multiplier (3×),
and BFS march toward unrevealed cells. Primary greedy capture scoring
otherwise follows the expander pattern.

## Seed grid

- Opponents: `smoke`.
- Seeds: 0 (verification gate).
- Mode: `--mode competition` only.

## Metrics

| Matchup | Games | fog_scout W-L-D | Mean turns |
| --- | --- | --- | --- |
| fog_scout vs smoke | 1 | 1-0-0 | 401 |

Seed 0: fog_scout (player 0) captured smoke's general on turn 401. Match
finished cleanly, no faults, 0 castles built.

## Decision

**Keep**, provisionally. Verification gate passed; early general capture
vs. `smoke` supports the fog-probing hypothesis on this seed.
