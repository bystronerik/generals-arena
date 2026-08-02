# Kubic holdout verification

Holdout wins n=37 (every 10th sorted win id). Rules derived on fit only; this page only scores holdout.

Script: `scripts/verify_kubic_holdout.py`

## Per-rule agreement

| Rule | n | hits | rate | status |
| --- | ---: | ---: | ---: | --- |
| `H1_pass_ticks_0_1` | 37 | 37 | 1.000 | **confirmed** |
| `H2_tiles50_band` | 37 | 37 | 1.000 | **confirmed** |
| `H3_contiguous_expansion` | 37 | 37 | 1.000 | **confirmed** |
| `H4_army_ratio_contact_ge1` | 37 | 37 | 1.000 | **confirmed** |
| `H5_has_general_sight` | 37 | 37 | 1.000 | **confirmed** |
| `H6_toward_after_sight` | 37 | 31 | 0.838 | **confirmed** |
| `H7_gather_wave` | 37 | 37 | 1.000 | **confirmed** |
| `H8_low_pass_rate` | 37 | 37 | 1.000 | **confirmed** |
| `H9_full_over_half` | 37 | 36 | 0.973 | **confirmed** |
| `H10_react_le_3` | 37 | 29 | 0.784 | **weak** |
| `H11_tip_at_sight_ge10` | 37 | 37 | 1.000 | **confirmed** |
| `H12_sight_to_kill_le120` | 37 | 36 | 0.973 | **confirmed** |
| `H13_enemy_not_adjacent_home` | 37 | 36 | 0.973 | **confirmed** |
| `H14_legacy_event_castle_tick10` | 22 | 21 | 0.955 | **refuted_policy_artifact** |
| `H14b_no_real_castle_by_50` | 37 | 37 | 1.000 | **confirmed** |
| `H15_pulse_tick3` | 37 | 37 | 1.000 | **confirmed** |
| `H16_flood_near_27` | 37 | 23 | 0.622 | **weak** |

## Failures / counterexamples

- `H6_toward_after_sight` (confirmed): Post-sight toward_fraction >= 0.7 — fails 6: `['20533', '20543', '20579', '21503', '21933', '21973']`
- `H9_full_over_half` (confirmed): Among full|half, full >= 90% — fails 1: `['21083']`
- `H10_react_le_3` (weak): React to first visible enemy within 3 ticks — fails 8: `['20563', '21083', '21219', '21537', '21697', '21845', '22023', '22163']`
- `H12_sight_to_kill_le120` (confirmed): Sight→kill <= 120 ticks — fails 1: `['22360']`
- `H13_enemy_not_adjacent_home` (confirmed): Enemy never reaches Manhattan<=1 of home — fails 1: `['21933']`
- `H14_legacy_event_castle_tick10` (refuted_policy_artifact): LEGACY/REFUTED policy: EventLog first castle stamp==10 if stamp<=20 — fails 1: `['21083']`
- `H16_flood_near_27` (weak): Flood start within 27±1 — fails 14: `['20553', '21219', '21247', '21697', '21713', '21729', '21773', '21817', '21905', '21933', '22057', '22910', '23766', '25159']`

## Status summary

- confirmed: 14
- weak: 2
- refuted: 0
- no_sample: 0

Refuted rules stay in the behavioral spec marked **refuted**, with these counterexamples. They are not dropped.

