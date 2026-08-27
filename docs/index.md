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
- [Joe GCP training](engine/joe-gcp-train.md) — the same run on a Google Cloud G4 spot VM (RTX PRO 6000); same R2 state, no bidding, stopped-not-outbid preemptions
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
- [joe-rs move selection](bots/joe-rs/selection-plan.md) — S1 deterministic Gumbel selection (shipped 2026-08-20, T knob, self-golden harness), the anti-repetition design space, S2/S3 dead, S4 as the post-plan follow-up
- [joe-rs A/B sanity](research/measurements/joe-rs-ab-sanity.md) — matched arms vs the panel: 100 identical game pairs, verdict `no change`
- [joe-rs S1 selection contrast](research/measurements/joe-selection-s1.md) — Gumbel T=1 vs the same network's argmax, round `s1-gumbel-r1`: +3.8 ± 10.7, proven flat; r2 replication pending
- [S3 Gate 1](research/measurements/joe-s3-gate1.md) — the depth-1 value re-rank killed offline: near-tie successors mostly identical, remaining gaps at the fabrication noise floor; no game spent
- [S4 trail rated round](research/measurements/joe-s4-trail-round.md) — round `s4-trail-r1`, 4,276 games: δ=6 and δ=4 proven flat vs the shipped default; full stack vs pure argmax joe unproven at −15.9 ± 11.1 (lean, decomposed to the S1 side); r2 owed
- [joe-rs soft trail penalty (S4)](research/strategies/joe-rs-noundo.md) — anti-circuit tax on re-entering the stack's recent trail (still-owned cells only, so retakes fight free): `JOE_RS_NOUNDO` δ + `JOE_RS_NOUNDO_WINDOW` (defaults 0/8 since 2026-08-21 — off by default); r1 proven flat, r2 owed
- [joe-M checkpoint selection](research/strategies/joe-M-checkpoint-selection-plan.md) — which of the 98 retained EMA sets of `joe-M-vast-20260813-0213` is strongest: **step 50000**, settled by GPU screen on zero arena time; the funnel, the calibration gate, and why stage 3 was not justified
- [joe checkpoint screen calibration](research/measurements/joe-ckpt-screen-calibration.md) — the GPU screen reproduces r3/r4 to 7.7 Elo on the widest contrast (gate passed); ~42 games/s on one H100; per-pair Elo does not chain, fit jointly
- [joe-M stage-1 checkpoint screen](research/measurements/joe-ckpt-s1-screen.md) — 22 checkpoints, 236,544 games: monotone to the end, **step 50000 strongest**, only 47500 within noise; the wide-ladder fit is misspecified (chi2/dof 9.7) so the order holds and the Elo magnitudes do not
- [joe-M stage-2 checkpoint screen](research/measurements/joe-ckpt-s2-screen.md) — the last 6,000 iterations at 500 granularity, 159,744 games: no local peak, 49500 and 50000 tie at 1.2 ± 3.0, **step 50000 is the answer**
- [joe-M7F4 checkpoint screen](research/measurements/joe-ckpt-m7f4-screen.md) — the deployed lineage, 216,064 games: steps 19000/20000 tie for best and the **shipped step 16000 is 12.0 ± 3.0 behind**; the ff×4 graft cost strength before it paid
- [why M7F4 gained so little](research/measurements/joe-m7f4-growth-recruitment.md) — ablation: zeroing the whole ff ×4 graft costs **−1.1 ± 3.8 Elo**; the grafted units sit at 11 % of mature scale after the full budget, and would need ~250k more iterations at this schedule
- [joe argmax limit cycle](research/measurements/joe-argmax-limit-cycle.md) — why joe walks a stack around its own castle: PPO sampled, deployment takes the argmax, and the escape never had to be learned; plus the repetition penalty that shipped for four days and what it never established
- [joe transformer kernel probe](research/measurements/joe-transformer-kernel-probe.md) — on the RTX PRO 6000 at X16 shape: removing the whole depth-16 trunk leaves the **rollout unchanged** while the PPO update collapses 13.6 s → 0.22 s, so the transformer blocks are **97.4 % of the rollout and 98.3 % of the PPO update**; env + observations + augmentation + stacking are **0.6 %**; the earlier "attention rewrites are dead" and "blocks are free" results were invalid (a nested jit cache served the first arm's binary to every arm); on retry **cuDNN flash attention is +5.1 %** and the **q/k/v fusion is -2.0 %**
- [joe training phase profile](research/measurements/joe-train-phase-profile.md) — where one PPO iteration goes on the real loop: **~50/50 rollout and PPO**, both purely the transformer, with env + observations + augmentation at **0.1 %** of the rollout; `use_bf16: true` costs **17x on Turing** and the bare bf16 GEMM proves it is not the matmuls
- [joe X16 training pilots](research/measurements/joe-x16-pilots.md) — the 29.55M depth-16 shape priced for a $100/5-day run: OOMs an 80 GB card at the production shape, 37.6k samples/s at `num_steps 128`, pmap N=2 scales linearly and restores the full recipe on 2×96 GB; three outbids in an hour say measure market churn before committing
- [unclejoe](bots/unclejoe/index.md) — joe-rs fork playing the X16 network (depth 16, 29.55M params, own R2 export) at **pure argmax** (T = 0); the name's second life since 2026-08-25
- [unclejoe export](bots/unclejoe/export.md) — the X16 export chain, artifact provenance (run/step/shas), zip size vs both cap readings, measured latency
- [unclejoe forward steps](research/measurements/unclejoe-forward-steps.md) — the forward pass split 25 ways on eight single-core x86 containers: 83–89 % GEMM, 9–14 % `exp` (`softmax` costs more than the two GEMMs it sits between), nothing else above 1 %
- [unclejoe forward optimisations](research/measurements/unclejoe-forward-ab.md) — four bit-exact changes off the back of that split, **+6.8 % to +9.5 %** of the forward on avx2 and avx512 alike: `exp_poly` was scalar on x86 because Rust's saturating `f32 as i32` blocks vectorization, the layernorm's eight lanes collapsed to one serial chain, and a runtime-width GEMM tail cost a third of `scores`; the GEMM prefetch was rejected on its own numbers
- [unclejoe shadow measurements](bots/unclejoe/shadow.md) — what the layer would have done while it did nothing: trigger rates (U2), what the proofs proved (U3), and U4's **no-go** on the value re-rank (the *previous* unclejoe, a tactics fork removed 2026-08-20)
- [unclejoe strategy spec](research/strategies/unclejoe.md) — the two claims (provable moments, one-step improvement), triggers, caps, and what the joe-rs contrast had to show (bot removed 2026-08-20)
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
