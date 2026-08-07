# Open questions

These choices are deliberately deferred. Each question gives a current default
and the evidence that must replace the default.

## Coupled online compute

**Unknown:** Which inference runtime, model width, particle count, simulation
target, and deadline reserve fit together on one competition CPU core.

**Current default:** float32 64-channel network; code defaults of 64
particles, 32 target simulations, and a 125 ms internal deadline, at most 113
forward-equivalents. The shipped best-effort `deployment.json` plays 8
particles, 16 target simulations, search depth 2, a 140 ms deadline with
10 ms reserve, and the uniform belief proposal.

**Evidence:** Measure the complete belief batch, root, enemy priors, leaf
batches, transitions, hashing, backup, and reply at p50 and p99. Compare joint
configurations such as 32/64/128 particles with several simulation targets.
Zero faults is required; no component is tuned in isolation.

**Part 09 status (2026-08-04, Apple M3 Pro):** verdict **no**. See
[`docs/research/measurements/morpheus-online-runtime.md`](../../research/measurements/morpheus-online-runtime.md).
Belief plus root fits only for small particle counts on this host; no trial
completed 8 simulations on every warm normal move. The written
`scripts/configs/morpheus/online-runtime.json` is best-effort play config, not
an accepted deployment.

## Army normalization

**Resolved:** verdict **keep_default**. The bootstrap corpus's maximum owned
stack is 1545, well inside the `log1p(x) / log1p(4096)` scale; quantization
error is small on ordinary and extreme stacks. See
[`morpheus-army-normalization.md`](../../research/measurements/morpheus-army-normalization.md).
Revisit only if a model manifest records a new scale.

## Belief recovery bounds

**Unknown:** Final lag, beam width, history count, and transition cap.

**Current default:** 8 turns, beam 8, 16 histories, and 128 transitions.

**Evidence:** Exact recovery rate and p99 cost after forced proposal mismatch
on recorded trajectories. A first measurement at `n_particles = 8` recovered
38 of 38 forced mismatches at p99 4.7 ms with these bounds
([`morpheus-belief-recovery.json`](../../research/measurements/morpheus-belief-recovery.json));
the sample is small (2 games), so the bounds stay open.

## Search resources

**Unknown:** Widening caps, exploration floor, depth, enemy-table cap, and tree
size.

**Current default:** Self cap 16 (floor 8), enemy cap 12, depth 16 in code
(the shipped `deployment.json` plays depth 2), 8 enemy tables per node, and
4,096 tree nodes.

**Evidence:** Tactical suites for chase, reinforcement, castle, mutual capture,
and deathtouch, plus table hit rate, eviction loss, memory, and arena contrast.
A first pass
([`morpheus-search-resources.json`](../../research/measurements/morpheus-search-resources.json))
passes all five suite cases with table hit rate 0.46 and zero eviction loss at
16 simulations; the resource choices stay open.

## Curriculum promotion

**Unknown:** Whether the Part 10 pilot thresholds stay after the first measured
training runs.

**Current default:** The executable Wilson non-degenerate-WDL rule in
[curriculum.md](curriculum.md#pilot-confidence-rule). Advance only when each
active class passes that rule.

**Evidence:** WDL intervals by class, full-start decisive rate, and held-out
arena strength. Training loss alone is not evidence.

## Opponent mixture

**Unknown:** Final league-to-panel ratio and sampling weights.

**Current default:** 70% checkpoint league and 30% fixed bot panel.

**Evidence:** Held-out exploitability, cycling, opponent coverage, decisive
rate, and pairwise arena contrast.

## Training losses and exploration

**Unknown:** Auxiliary-loss weights, root noise, action temperature, and the
deterministic turn.

**Current default:** No fixed values; rated play disables all exploration.

**Evidence:** Controlled ablations for belief calibration, policy entropy,
action coverage, cycling, and held-out arena strength.

**Provisional pilot freeze (2026-08-05):**
`training/morpheus/configs/pilot-objective.json` selects
`aux-light-explore-dirichlet` for the learning-curve pilot only
(`promotable_main_run: false`). Rated twin disables exploration. Replace after
the pending ablation measures land.

## Training compute

**Unknown:** Hardware, parallelism, self-play throughput, and games required per
checkpoint.

**Current default:** Training may use larger search settings, followed by a
deployment-matched self-play and calibration phase.

**Evidence:** End-to-end games per hour, network forwards per game, checkpoint
learning curves, and strength versus compute. A training design that cannot
produce enough games is rejected before implementation.

**Part 13 status (2026-08-05, Modal):** verdict **no**. Selected layout is
CPU, 1 physical core per game, sequential seat searches, 16 workers per A100.
Measured smoke throughput (`max_turns=8`) fits a 24 A100-hour schedule inside
the 48-hour budget, but no cadence candidate had curriculum WDL, belief
calibration, and pairwise `improvement` evidence. Fallback:
`narrower_non_promotable_research_scope`. See
[`docs/research/measurements/morpheus-modal-qualification.md`](../../research/measurements/morpheus-modal-qualification.md).

**Part 14 status (2026-08-05):** trainer / checkpoint / run-manifest modules
and Modal `train` / `resume` / `inspect_run` / `download` entry points are
implemented. Isolated resume tests pass. Config
`training/morpheus/configs/promotable-run.json` stays
`promotable_main_run: false` under the Part 13 research-scope fallback. Exit
verdict **no** until Part 13 is `yes` and one candidate finishes
deployment-matched calibration. See
[`docs/research/measurements/morpheus-trainer-checkpoints.md`](../../research/measurements/morpheus-trainer-checkpoints.md).

## Opponent belief approximation

**Unknown:** Whether level-zero last-seen belief is sufficient.

**Current default:** No recursive opponent particles. The deployed bot goes
further: the learned belief proposal is switched off entirely
(`use_policy_proposal: false`) after it showed no benefit vs macaria at
~13 ms/turn
([`belief-ablation-macaria.md`](../../research/measurements/belief-ablation-macaria.md));
particles advance on uniform legal enemy actions.

**Evidence:** Enemy-action log loss and real-observation particle survival
([`morpheus-opponent-belief.json`](../../research/measurements/morpheus-opponent-belief.json)
recommends keeping level zero). Compare one bounded extra belief level only if
the default misses critical actions within the same deadline.

## Submission-shaped harness

**Resolved:** `arena.matches.submission` owns the 150 ms, first-move, 2 GB,
fault, and EOF acceptance check. Judged runs stay out of `data/games/` and
ratings. See [Part 08](../../morpheus-implementation/08-submission-harness.md).

**Evidence:** `tests/test_submission_harness.py` rejects each controlled
failure under `tests/fixtures/submission_bots/` for the matching reason and
accepts the good fixture.

## ResBot castle evidence

**Unknown:** Whether distance-7 production candidates are stable build behavior
or a detector effect.

**Current default:** Diagnostic only.

**Evidence:** Action reconstruction that proves spend cell and turn on a
held-out ResBot sample. The result still cannot specify ResBot internals.
