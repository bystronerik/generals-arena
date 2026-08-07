# Part 17: Bounded prior-shaping blend

## Decision

Replace the unbounded heuristic root-prior reshape with a bounded,
configurable blend so the float32 network's policy can actually drive play,
while the proven hard rules stay rules.

Today `apply_pre_contact_prior` computes `score = max(nn, 1e-6) × heuristic`
with heuristic multipliers spanning ~5 orders of magnitude by action class
(enemy take ≈ 2,000–3,000; retreat ≈ 0.005; general capture ≈ 10^6). The
network prior spans ~2–3 orders. In effect the heuristic picks the action
class and the network only ranks within the winning class. That was rational
against the retired randn-calibrated int8 artifact; against the float32
deployment it suppresses whatever the network learned.

Target formula (per legal action, root evaluations only):

```text
h_norm  = h / gmean(h over legal)                 # center BEFORE clipping
p       = max(nn_prior, floor_frac · max(nn_prior))
shaped  = softmax( log p + lambda · clip(log h_norm, ±log_clip) )
```

- `lambda ∈ [0, 1]`: one trust knob. `lambda = 0` is the pure network prior
  over the play mask; `lambda = 1` approximates today's geometric blend.
- `log_clip` bounds any heuristic to a fixed multiplicative nudge
  (default `ln 10` → at most 10×), instead of a ~6,000× override.
- `floor_frac` replaces the absolute `1e-6` floor: an action the network
  zeroed cannot be resurrected by heuristics beyond the clip bound.
- Centering by the geometric mean is load-bearing: raw scores (~120) would
  all saturate the clip and the heuristic term would become a constant.

Class-5 note: the next training run includes class-5 (full-start) items, so
this part does **not** build special pre-contact machinery. The phase split
exists only as two config values that default equal.

## Steps

### A1. Refactor: split heuristics from blending (behavior-identical)

In `bots/morpheus/tactics.py`:

- `heuristic_action_scores(obs, memory, mask, belief) -> Array` — the
  existing scoring body with the `nn` factor removed. Drop the `10^6`
  general-capture term: captures are already guaranteed by
  `mandatory_action_indices` (candidate inclusion) and forced by
  `constrain_nn_action` (final hard rule).
- `blend_prior(nn_prior, scores, mask, *, lam, log_clip, floor_frac)` — the
  formula above.

Land with defaults that reproduce current behavior exactly
(`lam = 1`, `log_clip = inf`, absolute floor `1e-6`) plus a parity test:
shaped priors on the existing fixture boards are unchanged. A1 is a
zero-risk refactor commit.

### A2. Config plumbing and bounded defaults

- `DeploymentConfig` / `RuntimeConfig` / `NetworkEvaluator` gain:
  `shaping_lambda_pre_contact`, `shaping_lambda_post_contact` (defaults
  equal), `shaping_log_clip` (default `ln 10`), `shaping_floor_frac`
  (default `1e-3`). Phase selection uses the existing `enemy_is_visible`.
- Second commit flips defaults to the bounded regime. Every
  `deployment.json` change is a new content-hash rated entity — intended.

### A3. Unchanged

`play_mask`, `constrain_nn_action`, `mandatory_action_indices`, the
never-pass and general-capture hard rules, and all leaf/enemy priors
(shaping remains root-only).

### A4. Candidate-inclusion guard

One test: for any `lambda` (including 0), every index from
`mandatory_action_indices` is present in the root candidate set. Pins
"steer by inclusion, not by score" against future refactors.

### B. Instrument "who's deciding"

- `NetworkEvaluator` keeps `last_unshaped_prior`.
- New probe fields: `nn_top_action`, `nn_top_prior_milli`,
  `chosen_matches_nn_top`, `chosen_in_nn_top3`. Wire the full chain:
  `TurnMetrics` → `_publish_metrics` → `Agent._mirror_metrics` →
  `bots/morpheus/probe.py` → `arena.records.telemetry_schema` (probe keys
  must exist in the schema).
- `training/morpheus/measure_online.py` aggregates agreement rate per phase.

### C. Measurements

- **C0 (prerequisite).** Re-measure component p99s on the float32 runtime;
  the `offline_p99_ms` block in `deployment.json` is int8-era, so admission
  forecasts are stale.
- **C1. Net-value ablation.** `UniformEvaluator` vs `NetworkEvaluator`
  under identical shaping (`lambda = 1`). Implement the variant as a
  closure field in `deployment.json` (`"evaluator": "uniform"`) so each
  variant is an honest content-hash entity — no environment variables in a
  rated bot. Fixed panel, both seats, verdict per
  [decision-rule.md](../arena/decision-rule.md).
  **Gate:** if the contrast is ~0, the network contributes nothing yet;
  stop here and spend on training, not on the sweep.
- **C2. Lambda sweep.** Variants `lambda ∈ {1.0, 0.5, 0.25, 0}` (both phase
  values equal). Each vs the fixed panel; decide from the pairwise contrast
  plus secondaries: agreement rate, completed simulations, faults, decisive
  rate. Games into `data/games/`, then refit and `update-leaderboard`.
  Reports under
  `docs/research/measurements/morpheus-shaping-ablation.{md,json}` and
  `morpheus-shaping-lambda-sweep.{md,json}`.

### D. Docs and tests

- New `docs/bots/morpheus/prior-shaping.md`: blend formula, knob semantics,
  explicit hard-rule list. Fix the
  [design index](../bots/morpheus/index.md) claim that hard rules apply
  only to legality / transition / deadline — the shaping layer exists and
  is now documented config.
- Unit tests: `lambda = 0` returns exactly the renormalized network prior
  over `play_mask`; no action shifted more than `exp(log_clip)`; a
  zero-mass action cannot outrank the network top without clip-bound
  evidence. Existing shaping tests keep passing at the `lambda = 1`
  default. All new tests carry the `morpheus` marker.

## Order and dependencies

| Step | Depends on | Effort |
| --- | --- | --- |
| A1 refactor + parity test | — | ~half day |
| A2 config + bounded defaults, D tests | A1 | ~2–3 h |
| B probe fields + schema | — (parallel with A) | ~2–3 h |
| A4 guard test, D docs | A2 | ~1 h |
| C0 p99 re-measure | — | ~1 h |
| C1 ablation | A2, B, C0 | ~half day (game wall-clock) |
| C2 lambda sweep | C1 contrast > 0 | ~1 day (4 variants × panel) |

Two deliberate orderings: A1 lands as a provable no-op before any behavior
change, and C1 gates C2 so arena time is not spent tuning the blend of a
network that is not yet pulling weight.

## Exit criteria

- Parity test proves A1 changed nothing; bounded defaults land separately.
- Probe reports agreement rate; baseline recorded before any knob moves.
- C1 verdict recorded. If positive, C2 selects a `lambda` by pairwise
  contrast and the chosen value ships in `deployment.json`.
- Competition gate (`--mode competition` vs `bots/smoke/run.sh`) finishes
  normally after every closure commit.

## Related

- [Part 06: Simultaneous search](06-simultaneous-search.md)
- [Part 07: Runtime controller](07-runtime-controller.md)
- [Part 09b: Defer the normal latency gate](09b-defer-normal-deadline.md)
- [Design: network](../bots/morpheus/network.md),
  [runtime](../bots/morpheus/runtime.md)
