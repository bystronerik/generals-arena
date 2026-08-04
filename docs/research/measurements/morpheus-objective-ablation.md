# Morpheus objective ablation

> Verdict: **yes** — Targets, losses, augmentation, and fail-closed configs match hand fixtures. Main-run weights stay unset until Part 13 records arena evidence.

Generated: 2026-08-04T18:37:47.400605+00:00

## Selection

Selected candidate: `None`

No candidate is selected. Fixture checks passed, but belief calibration, action coverage, cycling, and held-out arena strength require the Part 13 A100 ablation budget. Charged a100_hours are recorded for that gate.

## Checks

| Check | Result |
| --- | --- |
| `hand_targets_and_losses` | yes |
| `symmetry_round_trips` | yes |
| `reward_no_shaping` | yes |
| `omitted_weights_fail_closed` | yes |
| `rated_exploration_disabled` | yes |
| `all_candidates_finite_loss` | yes |

## Accounting

```json
{
  "a100_hours": 0.0,
  "charged_to_part": "13-modal-compute-gate",
  "budget_note": "Part 12 objective ablations charge A100 time to Part 13."
}
```

## Candidates

```json
[
  {
    "name": "aux-off-explore-off",
    "ok": true,
    "loss_total": 0.8086093664169312,
    "loss_terms": {
      "policy": 0.8018488883972168,
      "wdl": 0.006760462652891874,
      "hidden_owner": 0.645661473274231,
      "enemy_army_bins": 0.09628172218799591,
      "enemy_general": 0.09628172218799591,
      "hidden_castle": 0.6694043278694153,
      "land_margin": 0.0,
      "army_margin": 0.0,
      "castle_margin": 0.0,
      "turns_to_termination": 0.0
    },
    "weights": {
      "policy": 1.0,
      "wdl": 1.0,
      "hidden_owner": 0.0,
      "enemy_army_bins": 0.0,
      "enemy_general": 0.0,
      "hidden_castle": 0.0,
      "land_margin": 0.0,
      "army_margin": 0.0,
      "castle_margin": 0.0,
      "turns_to_termination": 0.0
    },
    "exploration": {
      "root_noise_epsilon": 0.0,
      "root_noise_alpha": 0.3,
      "action_temperature": 1.0,
      "deterministic_turn": 0
    },
    "policy_entropy": 0.8018186629417832,
    "sampled_action": 3969,
    "exploration_enabled": false
  },
  {
    "name": "aux-light-explore-dirichlet",
    "ok": true,
    "loss_total": 1.0851060152053833,
    "loss_terms": {
      "policy": 0.8018488883972168,
      "wdl": 0.006760462652891874,
      "hidden_owner": 0.645661473274231,
      "enemy_army_bins": 0.09628172218799591,
      "enemy_general": 0.09628172218799591,
      "hidden_castle": 0.6694043278694153,
      "land_margin": 0.0,
      "army_margin": 0.0,
      "castle_margin": 0.0,
      "turns_to_termination": 0.0
    },
    "weights": {
      "policy": 1.0,
      "wdl": 1.0,
      "hidden_owner": 0.25,
      "enemy_army_bins": 0.25,
      "enemy_general": 0.25,
      "hidden_castle": 0.1,
      "land_margin": 0.05,
      "army_margin": 0.05,
      "castle_margin": 0.05,
      "turns_to_termination": 0.05
    },
    "exploration": {
      "root_noise_epsilon": 0.25,
      "root_noise_alpha": 0.3,
      "action_temperature": 1.0,
      "deterministic_turn": 30
    },
    "policy_entropy": 2.9792157141015605,
    "sampled_action": 20,
    "exploration_enabled": true
  },
  {
    "name": "aux-full-explore-temp",
    "ok": true,
    "loss_total": 1.3950730562210083,
    "loss_terms": {
      "policy": 0.8018488883972168,
      "wdl": 0.006760462652891874,
      "hidden_owner": 0.645661473274231,
      "enemy_army_bins": 0.09628172218799591,
      "enemy_general": 0.09628172218799591,
      "hidden_castle": 0.6694043278694153,
      "land_margin": 0.0,
      "army_margin": 0.0,
      "castle_margin": 0.0,
      "turns_to_termination": 0.0
    },
    "weights": {
      "policy": 1.0,
      "wdl": 1.0,
      "hidden_owner": 0.5,
      "enemy_army_bins": 0.5,
      "enemy_general": 0.5,
      "hidden_castle": 0.25,
      "land_margin": 0.1,
      "army_margin": 0.1,
      "castle_margin": 0.1,
      "turns_to_termination": 0.1
    },
    "exploration": {
      "root_noise_epsilon": 0.25,
      "root_noise_alpha": 0.3,
      "action_temperature": 1.5,
      "deterministic_turn": 50
    },
    "policy_entropy": 2.9792157141015605,
    "sampled_action": 521,
    "exploration_enabled": true
  }
]
```

## Metrics pending A100

- `belief_calibration`
- `policy_entropy_vs_coverage`
- `action_coverage`
- `cycling`
- `held_out_arena_strength`
