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
- [leaderboard replays](engine/leaderboard-replays.md) — scraped generals.bot games: format, layout, and why they are not arena matches
- [replay analysis](engine/replay-analysis.md) — `scripts/replay.py`: timeline, events, fog vs action, and flaw aggregates over scraped replays

## Bots

- [adding a bot](bots/adding-a-bot.md)
- [benchmark agents](bots/benchmark-agents.md) — fixed `cm_*` wrappers over competition-module JAX agents
- [smoke](bots/smoke.md) — stdio smoke test, not competitive
- [expand_plus](bots/expand-plus.md) — greedy capture + BFS frontier march
- [castle_builder](bots/castle-builder.md) — rested-general castle investment
- [castle_rush](bots/castle-rush.md) — early castle investment rush
- [general_hunter](bots/general-hunter.md) — deathtouch beeline once the enemy general is sighted
- [garrison](bots/garrison.md) — phase-based general reserve and route-aware defense
- [late_rush](bots/late-rush.md) — rally accumulation and turn-700 committed rush
- [fog_scout](bots/fog-scout.md) — fog-aware frontier probing
- [army_convey](bots/army-convey.md) — interior-to-frontier logistics
- [splitter](bots/splitter.md) — half-army dual-front expansion
- [choke_control](bots/choke-control.md) — corridor hold and denial
- [phase_switch](bots/phase-switch.md) — gated build phases
- [classic_duel](bots/classic-duel.md) — remote-only classic champion (no Elo)
- [blitz](bots/blitz.md) — general rush with re-sized strike waves (migrated from generals-bot)
- [boom](bots/boom.md) — fast-expand economy with an endgame latch (migrated from generals-bot)
- [metro](bots/metro.md) — castle network + mandatory pressure waves (migrated from generals-bot)
- [aegis](bots/aegis.md) — turtle + event-driven counterattack (migrated from generals-bot)
- [proteus](bots/proteus.md) — adaptive switcher over blitz/boom/metro/aegis (migrated from generals-bot)
- [macaria](bots/macaria.md) — vendored blitz core + a scoped tactical search
- [yankee](bots/yankee.md) — proteus plus a scoped MCTS and a RULES.md §07 endgame core
- [sosipolis](bots/sosipolis.md) — dual-mode MCTS general search with pocket skip and section priors

## Arena

- [game record schema](arena/game-record-schema.md)
- [match runner](arena/match-runner.md)
- [ratings](arena/ratings.md) — batch Bradley–Terry + Davidson draws + seat term
- [decision rule](arena/decision-rule.md) — the keep/revert thresholds, in one place
- [bot version registry](arena/bot-version-registry.md) — content hash → source, commit, git ref
- [trajectories](arena/trajectories.md) — per-turn recording, probes, replay
- [tournament](arena/tournament.md)
- [classic tournament](arena/classic-tournament.md)

## Research

- [experiment protocol](research/experiment-protocol.md)
- `research/experiments/` — one note per measurable strategy choice
- `research/strategies/` — one strategy spec per bot, written before the code
- [optimize the existing four bots](research/strategies/optimize-existing.md) — root causes and `Parameter revision 1`
- [tournament plan](research/strategies/tournament-plan.md) — seed grid, staging, metrics, and the round 2 schedule
- [diversity constraints](research/strategies/diversity-constraints.md) — hard rules that keep the roster from converging
- [sosipolis strategy](research/strategies/sosipolis.md) — dual-mode MCTS find-and-strike research bot
- [Kubic behavior (merged)](research/strategies/kubic-behavior-spec.md) — reverse-engineered leaderboard bot; defense present-but-rare
- [human 95/100 plan](research/strategies/human-95-plan.md) — remote goal: 95 wins in 100 logged games against humans on classic generals.io
- `research/measurements/` — one result set per tournament round
- [learned bot plan](research/learned-bot-plan.md) — not started; scaffold notes only
