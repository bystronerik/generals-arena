# 015 — aegis: migrated turtle + counterattack

Bot: [`bots/aegis/`](../../../bots/aegis/). Baselines: `garrison`,
`choke_control`, `blitz`, `smoke`. Migration of the generals-bot Aegis
strategy (grid-native rewrite; see
[`strategies/aegis.md`](../strategies/aegis.md)).

## Hypothesis

A measured garrison (wave estimate from exact aggregates) plus active
interception destroys attacks cheaply, and the event-driven counter
window converts the destroyed attack into the game. Safe home castle
builds (replacing the source's city grabs) keep the turtle from losing on
production to peaceful opponents.

**One change vs the source:** rules adaptation only — city grabs →
castle builds, deathtouch kill/defence added, `free_land` counted from
visible walls. Thresholds carried over unchanged as hypotheses.

## Seed grid

- Opponents: `smoke`, `garrison`, `late_rush`, `blitz`.
- Seeds: 0, 1, 2.
- Mode: `--mode competition` only.

## Metrics (verification matches, seed 0)

| Matchup | Games | aegis W-L-D | End turn | Castles built |
| --- | --- | --- | --- | --- |
| aegis vs smoke | 1 | 1-0-0 | 528 (capture, counter) | 3 |
| aegis vs expand_plus | 1 | 1-0-0 | 357 (capture) | 3 |
| aegis vs blitz | 1 | 0-1-0 | 290 (blitz capture) | 3 |

No faults. The blitz loss is the source's known rush-vs-turtle weakness
carried over — blitz's third wave arrived before the turtle's counter
window opened. Whether the anti-rush knobs (`WAVE_FRAC`, `MIN_WAVE`,
`PULL_RADIUS`) should firm up against rushers is a measurement-round
question, ideally via proteus-style classification rather than blanket
paranoia.

## Decision

**Keep**, provisionally. Revert if the full grid shows aegis losing to
non-rush opponents (defence model failure), never converting a counter
window (win-condition failure), or faulting.
