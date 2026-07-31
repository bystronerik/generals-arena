# 012 — blitz: migrated general rush

Bot: [`bots/blitz/`](../../../bots/blitz/). Baselines: `late_rush`,
`army_convey`, `expand_plus`, `smoke`. Migration of the generals-bot Blitz
strategy (grid-native rewrite; see
[`strategies/blitz.md`](../strategies/blitz.md)).

## Hypothesis

An early, repeatedly re-sized strike wave (first launch ≈ turn 50–64, waves
re-sized against `opp_army − opp_land`) kills expansion- and economy-
oriented bots before their advantage compounds. Source-ruleset numbers do
not transfer; the arena claim is that blitz ends games by capture well
before the 1200-turn draw.

**One change vs the source:** rules adaptation only — neutral-city path
walls removed, deathtouch finish/defence added. Thresholds carried over
unchanged as hypotheses (turn cadence is identical).

## Seed grid

- Opponents: `smoke`, `expand_plus`, `army_convey`, `late_rush`.
- Seeds: 0, 1, 2.
- Mode: `--mode competition` only.

## Metrics (verification matches, seed 0)

| Matchup | Games | blitz W-L-D | End turn | Strikes |
| --- | --- | --- | --- | --- |
| blitz vs smoke | 1 | 1-0-0 | 327 (capture) | 3 |
| blitz vs expand_plus | 1 | 1-0-0 | 226 (capture) | 4 |

Both games ended by general capture, no faults, no castle builds (by
design). Full seed grid pending measurement round.

## Decision

**Keep**, provisionally. Revert if the full grid shows blitz drawing at
1200 against the baseline pool (the rush failing to close is an identity
failure), or faulting, or losing the early game to `late_rush`-style timed
pushes on a majority of seeds.
