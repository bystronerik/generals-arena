# Morpheus Modal compute qualification

> Verdict: **no** — Compute gate closed. No tested cadence passed curriculum confidence, held-out belief calibration, and arena pairwise improvement. Deployment-matched final phase remains required. Main training must not start.

Generated: 2026-08-06T07:57:42.965123+00:00

## Selected layout

- self-play backend: `hybrid`
- physical CPU cores per game: `1`
- seat searches: `sequential`
- games per worker (measured batch): `2`
- concurrent games per container: `1`
- worker containers paired with one A100: `16`

## Throughput

- games/hour (aggregate): `14502.22948589885`
- positions/hour: `232035.6717743816`
- games/worker-hour: `906.3893428686781`
- mean game latency (s): `3.9714766455`

## Checkpoint cadence

- games/checkpoint: `87013`
- checkpoint count: `4`
- useful cadence found: `False`
- selected cadence: `None`

## A100 accounting

| Line | Hours |
| --- | ---: |
| `jax_preflight` | 0.03235120986777778 |
| `throughput_qualification` | 0.002206710106388889 |
| `learning_curve_pilot` | 2.0 |
| `objective_ablations` | 0.0 |
| `gpu_self_play` | 0.0 |
| `main_training` | 20.0 |
| `deployment_calibration` | 2.0 |
| **total** | **24.034557919974166** |
| budget | 48.0 |

- fits budget: `True`
- CPU hours (measured qualification): `0.02588340355222222`

## Checks

| Check | Result |
| --- | --- |
| `exact_layout_stated` | yes |
| `learner_not_starved` | yes |
| `useful_cadence_found` | no |
| `games_per_checkpoint_positive` | yes |
| `checkpoint_count_positive` | yes |
| `a100_fits_budget` | yes |
| `deployment_calibration_fits` | yes |
| `workers_per_a100_named` | yes |

## Fallback

```json
{
  "order": [
    "reduce_training_particles_sims_or_depth",
    "fewer_checkpoints_more_games_per_checkpoint",
    "narrower_non_promotable_research_scope"
  ],
  "selected": "narrower_non_promotable_research_scope",
  "note": "No tested cadence passed curriculum confidence, held-out belief calibration, and arena pairwise improvement. Deployment-matched final phase remains required. Main training must not start.",
  "preserve_deployment_matched_final_phase": true
}
```


## Layout ranking

```json
[
  {
    "name": "hybrid-cpu-control",
    "games_per_cpu_hour": 906.3893428686781,
    "mean_game_latency_s": 3.9714766455
  },
  {
    "name": "cpu1-seq",
    "games_per_cpu_hour": 753.3989988227199,
    "mean_game_latency_s": 4.777982980500001
  },
  {
    "name": "cpu2-par",
    "games_per_cpu_hour": 352.22401060724025,
    "mean_game_latency_s": 5.1100579345
  },
  {
    "name": "cpu2-par-x2",
    "games_per_cpu_hour": 130.34405911988378,
    "mean_game_latency_s": 13.80662328
  }
]
```

## Cadence candidates

```json
[
  {
    "name": "ckpt-4",
    "games_per_checkpoint": 87013,
    "checkpoint_count": 4,
    "curriculum": {
      "ok": false,
      "reason": "missing_class_wdl_or_active_classes",
      "classes": {}
    },
    "belief_calibration": {
      "ok": false,
      "regressed": null,
      "reason": "belief_calibration_not_measured"
    },
    "pairwise": {
      "ok": false,
      "verdict": null,
      "reason": "pairwise_not_improvement_or_missing"
    },
    "useful": false
  },
  {
    "name": "ckpt-8",
    "games_per_checkpoint": 43506,
    "checkpoint_count": 8,
    "curriculum": {
      "ok": false,
      "reason": "missing_class_wdl_or_active_classes",
      "classes": {}
    },
    "belief_calibration": {
      "ok": false,
      "regressed": null,
      "reason": "belief_calibration_not_measured"
    },
    "pairwise": {
      "ok": false,
      "verdict": null,
      "reason": "pairwise_not_improvement_or_missing"
    },
    "useful": false
  },
  {
    "name": "ckpt-16",
    "games_per_checkpoint": 21753,
    "checkpoint_count": 16,
    "curriculum": {
      "ok": false,
      "reason": "missing_class_wdl_or_active_classes",
      "classes": {}
    },
    "belief_calibration": {
      "ok": false,
      "regressed": null,
      "reason": "belief_calibration_not_measured"
    },
    "pairwise": {
      "ok": false,
      "verdict": null,
      "reason": "pairwise_not_improvement_or_missing"
    },
    "useful": false
  }
]
```
