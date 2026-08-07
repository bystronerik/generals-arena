# Belief-proposal ablation vs macaria (100 games per arm)

> Verdict: **no measurable benefit from the learned belief proposal in this
> matchup.** Belief-on won 34/100, belief-off (uniform proposals) won 41/100;
> paired difference −0.07 ± 0.13 (95% CI), McNemar exact p = 0.35. The
> learned enemy model inside the particle filter is not earning its latency
> against macaria under the current deployment. Single-opponent, 100-game
> arms — directional evidence, not a decision-rule verdict.

Date: 2026-08-08 (run started 2026-08-07). Host: Apple M3 Pro, 4 parallel
match workers per arm (identical for both arms).

**Adopted 2026-08-08:** `bots/morpheus` now runs `proposal_policy=None`
(uniform belief proposals), i.e. the belief-off arm. Rationale: macaria was
part of the training corpus, so this matchup should have flattered the
learned proposal — and it still showed no benefit while costing ~13 ms/turn
of the 140 ms budget. Verified: bot tests pass, default suite 592 passed,
competition gate vs `bots/smoke` reached a normal end (win, turn 587).

## Design

- **Arms.** `bots/morpheus` (then-deployed, learned proposal, content hash
  `d3399c6dc5f5`) vs `morpheus_ubelief@c4f9119c7f0a` — identical code except
  `RuntimeController(proposal_policy=None)`, so `propose_enemy_actions` and
  `recover_belief` draw enemy actions uniformly from each particle's legal
  mask. Root, leaf, and enemy-prior inference unchanged; the ablation
  isolates the learned enemy model in the belief filter.
  `bots/morpheus_ubelief/` was a comparison copy and has since been removed
  (same convention as `bots/macaria_base/`); its registry entry stays under
  `data/bot_versions/` as provenance for the belief-off games.
- **Opponent.** `bots/macaria` (research bot, rated ≈ 2292 in the bootstrap
  panel) — deliberately a stronger opponent, per the question "is the belief
  mostly wrong against better players?".
- **Games.** Fixed seeds 0–49, `--seat-policy alternate` → 100 games per arm
  on identical maps and orientations; matched pairs across arms. Rounds
  `belief-ablation-on` / `belief-ablation-off` under `data/games/`; no
  rating refit.
- **Gate.** `morpheus_ubelief` finished a competition match vs `bots/smoke`
  (win, turn 540, normal end) before measurement.
- **Config state.** Both arms ran *after* the `enemy_prior_batch: NaN`
  admission-lockout fix (commit `ce6643e`; the round manifest records
  `morpheus@d3399c6dc5f5`, the fixed tree). Enemy priors are admitted from
  turn 1 in both arms (verified post-fix on the instrumented harness:
  early-turn sims mean 7.4 vs 0.1 under the lockout).

## Results

| Arm | W | L | D | winrate | Wilson 95% |
| --- | ---: | ---: | ---: | ---: | --- |
| belief on (learned proposal) | 34 | 64 | 2 | 0.34 | [0.25, 0.44] |
| belief off (uniform proposal) | 41 | 57 | 2 | 0.41 | [0.32, 0.51] |

Matched pairs (same seed + orientation): 59 concordant, 17 on-better,
24 off-better. McNemar exact two-sided p = 0.35. Paired winrate difference
(on − off): **−0.07 ± 0.13** (95% CI). Median game length 524 turns (on) /
486 (off); only 2/100 pairs had identical turn counts, so the ablation
genuinely changed play.

## Reading

The contrast is statistically indistinguishable from zero, with the point
estimate *favoring the ablation*. Search was live from turn 1 in both arms
(post-NaN-fix), so the result cannot be blamed on a muted search consumer.
One mechanism plausibly offsets any belief quality the learned proposal
adds:

- **Latency refund.** The proposal forward costs ~13 ms per warm move
  (mean, [`morpheus-rust-rewrite-analysis.json`](morpheus-rust-rewrite-analysis.json));
  with `proposal_policy=None` that time returns to the admission budget,
  buying search simulations instead. The stored game records carry no
  per-move telemetry, so the sims-gained side of the trade is unmeasured.

## Caveats

- One opponent, 100 games/arm: far below the decision-rule bar
  ([decision-rule.md](../../arena/decision-rule.md): 5-bot panel, ≥200
  games/arm). This is a directional probe, not a lineage decision.
- macaria is a heuristic bot whose style may sit off the scraped-ladder
  distribution the proposal was trained on
  ([scraped-classes15.md](../../morpheus-implementation/scraped-classes15.md)) —
  the matchup where a learned ladder-population model would help least.
- 8 particles (deployed play config): belief quality is fragile in both arms.
- 4-way match parallelism adds latency contention; identical for both arms
  but not identical to solo play.

## Follow-ups that would sharpen this

1. Record trajectories (`--record`) to compare completed simulations and ESS
   between arms — separates the latency refund from belief quality.
2. A panel round per the decision rule if the uniform-proposal variant is
   ever considered for promotion.
