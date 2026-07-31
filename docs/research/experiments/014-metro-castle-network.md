# 014 — metro: castle network + pressure

Bot: [`bots/metro/`](../../../bots/metro/). Baselines: `castle_builder`,
`castle_rush`, `phase_switch`, `expand_plus`. Migration + redesign of the
generals-bot Metro (city control) strategy — neutral cities do not exist
in the arena, so the production programme builds castles and hunts enemy
castles (see [`strategies/metro.md`](../strategies/metro.md)).

## Hypothesis

Three gather-funded forward castles + mandatory pressure waves from turn
140 out-produce the pool *and* convert the lead into captures — unlike the
existing castle-economy cluster, whose mutual games all drew. The
diversity differentiators (field-funded builds, forward placement,
garrisoned bastions, enemy-castle capture priority, mandatory waves) are
the experiment.

**One change vs the source:** the city programme became a castle-build
programme (chipping and scout-bias deleted, enemy-castle cash-in added);
defence, garrison, and pressure logic carried over with their thresholds.

## Seed grid

- Opponents: `smoke`, `expand_plus`, `castle_builder`, `castle_rush`.
- Seeds: 0, 1, 2.
- Mode: `--mode competition` only.

## Metrics (verification matches, seed 0)

| Matchup | Games | metro W-L-D | End turn | Castles (metro vs opp) |
| --- | --- | --- | --- | --- |
| metro vs smoke | 1 | 1-0-0 | 378 (capture) | 3 vs 0 |
| metro vs expand_plus | 1 | 1-0-0 | 382 (capture) | 3 vs 0 |
| metro vs castle_builder | 1 | 1-0-0 | 382 (capture) | 3 vs 0 |

No faults. The castle_builder game is the key datum: a win by capture
against a home-builder, where the existing cluster only ever drew.

## Decision

**Keep**, provisionally. Revert if the full grid shows metro drawing its
castle-cluster games (diversity failure), building fewer than 2 castles
per game (programme starvation), or losing its general during the saving
phase (the source's signature weakness resurfacing).
