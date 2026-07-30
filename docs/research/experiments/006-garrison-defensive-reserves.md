# 006 — garrison: defensive reserves

Bot: [`bots/garrison/`](../../../bots/garrison/). Baseline: `smoke`, `expand_plus`.

## Hypothesis

`smoke` and `expand_plus` expand without a general reserve or route-aware
reinforcement. A phase-based army floor on the own general, plus pressure
scoring and deathtouch intercept logic after turn 800, should reduce losses
against bots that reach the home area without increasing draws by more than
20 percentage points against passive expanders.

One change cluster: general reserve floors by phase, own-territory
reinforcement, visible-pressure threat model, emergency defense scoring,
and deathtouch chase/intercept after turn 800.

## Seed grid

- Opponents: `smoke`.
- Seeds: 0 (verification gate).
- Mode: `--mode competition` only.

## Metrics

| Matchup | Games | garrison W-L-D | Mean turns |
| --- | --- | --- | --- |
| garrison vs smoke | 1 | 0-0-1 | 1200 |

Seed 0: match truncated at turn 1200 (draw). Match finished cleanly, no
faults, 0 castles built.

## Decision

**Keep**, provisionally. Verification gate passed; the reserve and
reinforcement logic run without faults. The draw vs. `smoke` is expected
for a defensive bot against a passive expander on a single seed — extend
the seed grid against active opponents before changing thresholds.
