# 010 — phase_switch: phase-gated castle builds + late hunt

Bot: [`bots/phase_switch/`](../../../bots/phase_switch/). Baselines: `castle_builder`, `expand_plus`, `general_hunter`, `smoke`.

## Hypothesis

Gating `castle_builder`'s build mechanism to a mid-phase window (turn 80–800), with pure `expand_plus` before and a deathtouch hunt after turn 800, yields at least as much late-game income as `castle_builder` (which builds from turn 20–900) and converts any enemy-general sighting from turn 800 into a win that `castle_builder` cannot get.

**One change vs `castle_builder`:** the phase gates around the build mechanism. Build, relocate, expand, and hunt sub-procedures are reused unchanged from `castle_builder` / `expand_plus` / `general_hunter`.

## Seed grid

- Opponents: `smoke`, `expand_plus`, `castle_builder`, `general_hunter`.
- Seeds: 0, 1, 2.
- Mode: `--mode competition` only.

## Metrics (verification match, seed 0 vs smoke)

| Matchup | Games | phase_switch W-L-D | Mean turns | Castles built |
| --- | --- | --- | --- | --- |
| phase_switch vs smoke | 1 | 0-0-1 | 1200.0 | 2 (turns 130, 186) |

Full seed grid pending tournament run. Expected outcome mirrors experiments `001`–`003`: most games draw at 1200 turns because none of the opponents scout deep enough to expose a general; positive evidence is no regression plus wired hunt/build mechanisms.

## Decision

**Keep**, provisionally. Revert if any matchup shows more faults than `castle_builder` on the same seeds, or if it builds fewer castles per game than `castle_builder` (would mean the phase gate broke the build mechanism).

**Telemetry gap:** game-record schema stores winner/turns/terminated/truncated only — income claim needs final land/army at truncation (schema follow-up).
