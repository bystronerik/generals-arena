# Morpheus offline p99 recalibration (uniform belief proposal)

> Asked: re-measure `belief_proposal`. Answer: **2.497 ms** (was a `10.0`
> placeholder; `43.659` at HEAD before that), now shipped.
>
> Found while measuring: **`particle_transitions` is seeded `0.0` but costs
> 66.6 ms** under uniform proposals, because belief collapses on 15–23% of
> turns and `recover_belief` runs inside that stage's timing block. The
> uniform proposal is **not** a net latency saving — it moves ~26 ms of
> proposal forward into ~65 ms of collapse-and-rebuild. Not changed here:
> re-seeding `particle_transitions` materially changes admission behaviour
> and is the operator's call.

Date: 2026-08-08. Host: Apple M3 Pro, `torch.set_num_threads(1)`, float32
artifacts. Method: `training.morpheus.measure_online._calibrate_offline_p99`
(the routine qualification itself uses), side 18, seeds 0 / 100 / 200.

## Harness defect fixed first

`_make_controller` hardcoded `proposal_policy=evaluator.policy_logits` while
`bots/morpheus/agent.py` had been changed to `None`. Calibrating in that
state would have measured a proposal path the bot no longer plays. Both now
read one flag, `DeploymentConfig.use_policy_proposal` (default `False`),
recorded explicitly in `deployment.json` and the operator copy.

## Calibration table

Three seeds, uniform proposal (the deployed mode). "Shipped" is the value in
`deployment.json` before this run.

| Component | shipped | seed 0 | seed 100 | seed 200 | verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| `belief_proposal` | 10.0 (placeholder) | 2.468 | 2.497 | 2.476 | **updated → 2.497** |
| `particle_transitions` | **0.0** | 71.030 | 81.051 | 61.548 | see below — not changed |
| `selection` | 7.308 | 0.994 | 0.967 | 0.979 | over-seeded ~7× |
| `backup` | 2.257 | 0.683 | 0.670 | 0.687 | over-seeded ~3× |
| `belief_tensor` | 2.104 | 1.693 | 1.703 | 1.741 | close |
| `root_inference` | 5.882 | 8.494 | 8.678 | 8.226 | under-seeded ~1.5× |
| `leaf_batch` | 21.188 | 26.671 | 25.387 | 25.452 | under-seeded ~1.25× |
| `enemy_prior_batch` | 19.779 | 24.692 | 22.678 | 22.570 | under-seeded ~1.2× |
| `hashing`, `reply` | 0.0 | 0.0 | 0.0 | 0.0 | genuinely ~free |

Only `belief_proposal` was changed. The rest date from the int8
qualification era; re-seeding them shifts admission in both directions and
belongs in a qualification pass, not a single-field edit.

## What the ablation actually traded

Same calibration, both proposal modes, plus rehearsal belief health:

| | uniform (deployed) | policy proposal |
| --- | ---: | ---: |
| `belief_proposal` p99 | 2.44 ms | 28.06 ms |
| `particle_transitions` p99 | **66.64 ms** | 1.23 ms |
| belief stage total | **~69 ms** | **~29 ms** |
| collapsed rate (loose deadline) | 0.154 | 0.000 |
| collapsed rate (140 ms deadline) | 0.231 | 0.000 |
| deadline faults / 13 warm moves | 1 | 0 |
| sims mean (140 ms deadline) | 10.77 | 8.62 |

Mechanism: `runtime.decide` calls `recover_belief` inside the
`particle_transitions` timing block whenever every particle weight reaches
zero. Uniform enemy actions diverge from real play, the observation
likelihood kills the whole 8-particle set, and the rebuild lands in that
stage. `recovery_rate` stays 0.0 because that counter tracks a different
path (belief not admitted), so the collapse is invisible in the metric an
operator would look at first.

**This corrects an earlier claim.** The commit that disabled the proposal
(`489fa5a`) and my summary of it said skipping the proposal forward
"returns ~13 ms/turn to the search budget". In p99 terms the belief stage
got *more* expensive, not less. The winrate evidence behind the decision is
unaffected — that was measured end-to-end over 200 real games
([belief-ablation-macaria.md](belief-ablation-macaria.md), 41W vs 34W,
paired −0.07 ± 0.13) — and sims mean is higher under uniform here, because
`particle_transitions: 0.0` lets admission spend time it never accounted
for. The latency rationale was wrong; the measured outcome still stands.

## Open items (operator decisions)

1. **`particle_transitions: 0.0` is a wrong seed of the same class as the
   `enemy_prior_batch: NaN` lockout, in the opposite direction.** NaN meant
   "never admit"; 0.0 means "always admit, cost-free". Seeding the measured
   66.6 ms would make admission refuse belief updates on many turns —
   arguably correct accounting, but a large behavioural change that needs
   its own before/after. `deployment.py` currently rejects non-finite and
   negative seeds but accepts 0.0.
2. **Belief collapses on 15–23% of turns under uniform proposals.** If that
   is judged too high, the middle option between the two arms is a cheaper
   proposal (distilled or smaller net) rather than either extreme.
3. Stale seeds for `root_inference`, `leaf_batch`, `enemy_prior_batch`
   (under-seeded) and `selection`, `backup` (over-seeded) — a full
   qualification pass, not one-field edits.

## Verification

Default suite 592 passed. Competition gate
(`matchup.py bots/morpheus/run.sh bots/smoke/run.sh --mode competition
--seed 0`) reached a normal end, win at turn 435.
