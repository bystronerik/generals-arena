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
- [sprint replays](engine/sprint-replays.md) — whole generals.bot tournaments: the results asset, the blob layout, and the organizers' ruleset
- [replay analysis](engine/replay-analysis.md) — `scripts/replay.py`: timeline, events, fog vs action, and flaw aggregates over scraped replays
- [Joe vast.ai training](engine/joe-vast-train.md) — interruptible GPU runs; durable state in R2, not the Modal Volume
- [Modal jobs](engine/modal-jobs.md) — how a remote job fails silently, the two import traps, and how to read the container's log

## Bots

- [adding a bot](bots/adding-a-bot.md)
- [benchmark agents](bots/benchmark-agents.md) — fixed `cm_*` wrappers over competition-module JAX agents
- [joe](bots/joe.md) — deployed Average Joe self-play PPO policy (EMA, greedy)
- [joe-rs port plan](bots/joe-rs/port-plan.md) — Rust sibling of joe: constraints, milestones, risks
- [joe-rs export](bots/joe-rs/export.md) — ema.eqx → safetensors through the deployment template, and the schema the loader enforces
- [joe-rs parity](bots/joe-rs/parity.md) — corpus, tiers, pinned bounds, wire replay, and the mutation pass that checks the proof
- [joe-rs XLA semantics](bots/joe-rs/xla-semantics.md) — the reciprocal-multiply rewrite and XLA's log1p with FMA contraction, mirrored bit-for-bit
- [joe-rs refactor](bots/joe-rs/refactor-plan.md) — the flat module list into io/board/nn subdirectories: the grouping, the pure-move rule, and the blast radius
- [joe-rs latency](bots/joe-rs/latency.md) — full-path percentiles on one x86 core; why candle stays
- [joe-rs packaging](bots/joe-rs/packaging.md) — J5 plan: the shared Rust packager, the measured zip budget, and what a submission still needs
- [joe-rs A/B sanity](research/measurements/joe-rs-ab-sanity.md) — matched arms vs the panel: 100 identical game pairs, verdict `no change`
- [joe argmax limit cycle](research/measurements/joe-argmax-limit-cycle.md) — why joe walks a stack around its own castle: PPO sampled, deployment takes the argmax, and the escape never had to be learned; plus the repetition penalty that now ships and what it does **not** establish
- [unclejoe fork plan](bots/unclejoe/fork-plan.md) — the joe-rs fork: copy strategy, own rating identity, and the U1 equality proof (built 2026-08-15)
- [unclejoe tactics plan](bots/unclejoe/tactics-plan.md) — the layer on top of the fork: proof-gated override, castle/refutation filters, afterstate value re-rank, full-budget clock plan
- [unclejoe shadow measurements](bots/unclejoe/shadow.md) — what the layer would have done while it did nothing: trigger rates (U2), what the proofs proved (U3), and U4's **no-go** on the value re-rank
- [unclejoe strategy spec](research/strategies/unclejoe.md) — the two claims (provable moments, one-step improvement), triggers, caps, and what the joe-rs contrast has to show
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
- [macaria](bots/macaria.md) — vendored blitz core + a scoped tactical search
- [morpheus](bots/morpheus/index.md) — RL policy/value network + belief-aware simultaneous MCTS
- [morpheus-rs rewrite plan](bots/morpheus-rs/rewrite-plan.md) — Rust sibling of morpheus: milestones, gates, risks
- [morpheus-rs parity corpus](bots/morpheus-rs/parity-corpus.md) — recorded frames and RNG streams the Rust port is checked against
- [morpheus-rs packaging](bots/morpheus-rs/packaging.md) — crate layout, content hash, and the hash-named submission archive
- [morpheus-rs parity harness](bots/morpheus-rs/parity-harness.md) — how a ported surface is proved equal, and how the proof is itself checked
- [morpheus-rs inference](bots/morpheus-rs/inference.md) — the hand-written network engine, the artifact contract, and the shoot-out that chose it
- [morpheus-rs belief filter](bots/morpheus-rs/belief.md) — particles, recovery, the injected RNG, and the two NumPy behaviours the port reproduces
- [morpheus-rs search and tactics](bots/morpheus-rs/search-and-tactics.md) — the arena tree, the shared generator, the BLAS reduction the oracle cannot pin, and the decision surface
- [morpheus-rs telemetry](bots/morpheus-rs/telemetry.md) — the bot's own per-turn trace, why a subprocess needs one, and the thread-count invariant
- [morpheus-rs refactor plan](bots/morpheus-rs/refactor-plan.md) — the flat module list into nine subpackages: the tree, the nine staged moves as they landed, and what a rename costs the content hash
- [morpheus-rs M6 latency](research/measurements/morpheus-rs-m6-latency.md) — the Rust bot on M0's schedule, and the two ways a component table misleads
- [morpheus-rs M6 strength](research/measurements/morpheus-rs-m6-strength.md) — the arena gate at parity knobs, and what a deadline-shaped configuration does to "identical"
- [morpheus-rs M8 submission](research/measurements/morpheus-rs-m8-submission.md) — the audited archives, the intake selfcheck, and why a green smoke test proved nothing
- [joe-net port plan](bots/morpheus-rs/joe-net-plan.md) — joe's frozen network under morpheus's search: the fork, what belief loses, and the gates
- [joe-net N0](research/measurements/joe-net-n0.md) — the forward-only cost, the forward budget by consumer, and what the real controller does at a 21 ms forward

## Arena

- [game record schema](arena/game-record-schema.md)
- [match runner](arena/match-runner.md)
- [ratings](arena/ratings.md) — batch Bradley–Terry + Davidson draws + seat term,
  fitted once per round
- [decision rule](arena/decision-rule.md) — the keep/revert thresholds, in one place
- [bot version registry](arena/bot-version-registry.md) — content hash → source, commit, git ref
- [trajectories](arena/trajectories.md) — per-turn recording, probes, replay
- [tournament](arena/tournament.md)
- [classic tournament](arena/classic-tournament.md)
- [ratings refactor plan](arena/ratings-refactor-plan.md) — the audit that replaced sequential Elo with a batch fit; a historical record of the pooled design
- [per-round ratings plan](arena/per-round-ratings-plan.md) — why the pool itself went away: one independent fit per round, with the measurements behind it

## Research

- [experiment protocol](research/experiment-protocol.md)
- `research/experiments/` — one note per measurable strategy choice
- `research/strategies/` — one strategy spec per bot, written before the code
- [optimize the existing four bots](research/strategies/optimize-existing.md) — root causes and `Parameter revision 1`
- [tournament plan](research/strategies/tournament-plan.md) — seed grid, staging, metrics, and the round 2 schedule
- [diversity constraints](research/strategies/diversity-constraints.md) — hard rules that keep the roster from converging
- [sosipolis strategy](research/strategies/sosipolis.md) — dual-mode MCTS find-and-strike research bot (bot removed 2026-08-18; spec kept as a record)
- [Kubic behavior (merged)](research/strategies/kubic-behavior-spec.md) — reverse-engineered leaderboard bot; defense present-but-rare
- [human 95/100 plan](research/strategies/human-95-plan.md) — remote goal: 95 wins in 100 logged games against humans on classic generals.io
- [top leaderboard loss analysis](research/measurements/leaderboard-top-loss-analysis.md) — how Kubic, bca, and thor lose: the grind, the punch, and the blind stall (per-player profiles linked inside)
- [morpheus-rs leaderboard gaps](research/measurements/morpheus-rs-leaderboard-gaps.md) — winner behaviors mapped onto morpheus-rs, ranked, with the measurements that would settle them
- `research/measurements/` — one result set per tournament round
