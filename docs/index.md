# Docs index

Small topic files for the Generals Arena research repo.

## Start here

- [Competition rules (root)](../RULES.md) — processed official rules + code cross-check
- [Sources](sources.md) — DeepWiki, generals.bot, submodule

## Competition

- [vs classic](competition/vs-classic.md)
- [stdio protocol](competition/protocol.md)
- [build castles](competition/build-castles.md)
- [deathtouch](competition/deathtouch.md)
- [map generation](competition/map-generation.md)

## Engine

- [unified bot API](engine/unified-bot-api.md) — one observation/action shape for stdio and live generals.io
- [local matchup](engine/local-matchup.md)
- [classic matchup (remote practice)](engine/classic-matchup.md)
- [remote generals.io](engine/remote-generalsio.md)
- [remote eval of heuristics](engine/remote-eval-heuristics.md) — playing humans on classic generals.io, and why it is not a competition result
- [remote play setup](engine/remote-play-setup.md) — env vars, CLI, and logging for live generals.io

## Bots

- [adding a bot](bots/adding-a-bot.md)
- [smoke](bots/smoke.md) — Phase 1 stdio smoke test, not competitive
- [expand_plus](bots/expand-plus.md) — greedy capture + BFS frontier march
- [castle_builder](bots/castle-builder.md) — rested-general castle investment
- [general_hunter](bots/general-hunter.md) — deathtouch beeline once the enemy general is sighted
- [garrison](bots/garrison.md) — phase-based general reserve and route-aware defense
- [late_rush](bots/late-rush.md) — rally accumulation and turn-700 committed rush

## Arena

- [game record schema](arena/game-record-schema.md)
- [match runner](arena/match-runner.md)
- [ratings (elote)](arena/ratings.md)
- [tournament](arena/tournament.md)
- [classic tournament](arena/classic-tournament.md)

## Research

- [experiment protocol](research/experiment-protocol.md)
- `research/experiments/` — one note per measurable strategy choice
- `research/strategies/` — one strategy spec per bot, written before the code
- [optimize the existing four bots](research/strategies/optimize-existing.md) — root causes and `Parameter revision 1`
- [tournament plan](research/strategies/tournament-plan.md) — seed grid, staging, metrics, and the round 2 schedule
- [diversity constraints](research/strategies/diversity-constraints.md) — hard rules that keep the roster from converging
- [human 95/100 plan](research/strategies/human-95-plan.md) — Phase 3 goal: 95 wins in 100 logged games against humans on classic generals.io
- `research/measurements/` — one result set per tournament round
- [learned bot plan (Phase 4 scaffold)](research/learned-bot-plan.md)
