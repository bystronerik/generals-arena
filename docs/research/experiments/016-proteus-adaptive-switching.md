# 016 — proteus: adaptive strategy switching

Bot: [`bots/proteus/`](../../../bots/proteus/). Baselines: the four pure
cores (`blitz`, `boom`, `metro`, `aegis`) and `phase_switch`. Migration of
the generals-bot adaptive bot (see
[`strategies/proteus.md`](../strategies/proteus.md)).

## Hypothesis

Evidence-driven switching (classifier + debounced switcher over one
shared opponent model) keeps blitz's strong matchups while patching its
weak ones: the defensive fast-path answers rushes, and boom out-economies
passive turtlers. Proteus should beat the *worst-case* pure-core matchup
against each opponent archetype.

**One change vs the source:** cities → castles in the classifier
(`enemy_castles_seen`); everything else inherited from the migrated cores.

## Seed grid

- Opponents: `smoke`, `blitz`, `garrison`, `castle_builder`.
- Seeds: 0, 1, 2.
- Mode: `--mode competition` only.

## Metrics (verification matches, seed 0)

| Matchup | Games | proteus W-L-D | End turn | Switches |
| --- | --- | --- | --- | --- |
| proteus vs smoke | 1 | 1-0-0 | 347 (capture) | 1 (→ boom, "turtler") |
| proteus vs expand_plus | 1 | 1-0-0 | 344 (capture) | — |
| proteus vs blitz | 1 | 0-1-0 | 289 (blitz capture) | — |

No faults. The smoke game exercised the full pipeline: classified smoke
as a turtler, switched blitz → boom, built a castle, won. The blitz game
is a near-mirror (proteus opens as blitz) decided by tempo — an expected
coin-flip matchup, not a regression.

## Decision

**Keep**, provisionally. Revert if the full grid shows proteus
underperforming its own worst pure core against any archetype (the
switching would then be pure overhead), switching more than ~3 times per
game (hysteresis failure), or faulting.
