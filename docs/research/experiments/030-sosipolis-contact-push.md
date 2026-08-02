# 030 — sosipolis: contact push-expand (sticky frontier)

Bot: [`bots/sosipolis/`](../../../bots/sosipolis/).
Prior: [`029-sosipolis-contact-hunt-ownership.md`](029-sosipolis-contact-hunt-ownership.md).
Baseline panel: [`../measurements/sosipolis-contact-p1-diag.md`](../measurements/sosipolis-contact-p1-diag.md).

## Hypothesis

Kubic keeps expanding after contact (~63% neutral gains) with one tip toward a
stable believed cell. Softening sticky macro churn (drop split/mid_edge),
preferring frontier waypoints, and biasing wave scores to push-expand raises
sight rate vs contact-p1 on seeds 0–4 alternate vs `macaria`.

## Revision (applied, then reverted)

| Change | Old | New |
| --- | --- | --- |
| `CONTACT_COMMIT_MIN_TURNS` | 12 | 24 |
| `CONTACT_SWITCH_RATIO` | 1.35 | 1.75 |
| `CONTACT_SWITCH_MARGIN` | 0.08 | 0.20 |
| `CONTACT_MAX_MACROS` | 6 | 3 |
| `CONTACT_LAND_BONUS` | 55 | 90 |
| `CONTACT_ENEMY_BONUS` | 45 | 40 |
| `CONTACT_PUSH_NEUTRAL_BONUS` | — | 45 |
| `CONTACT_ALLOW_SPLIT_MACROS` | (always on) | False |
| `CONTACT_ALLOW_MID_EDGE` | (always on) | False |
| `CONTACT_FRONTIER_PREF` | — | 0.12 |
| `STRIKE_TIP_HOLD` | 8 | 16 |
| tip retip in contact | always `select_mass_tip` | keep tip unless new ≥1.25× mass |

## Gate

Seed 0 vs `smoke`: sosipolis win, first castle turn 135 (≥116), no faults.

## Measurement

Round: [`../measurements/sosipolis-contact-push1.md`](../measurements/sosipolis-contact-push1.md).

| Round | Sight rate | Contact median | Contact→sight | W–L |
| --- | ---: | ---: | ---: | --- |
| contact-p1-diag (baseline) | **4/10 (40%)** | 82.5 | **373** | 3–7 |
| **contact-push1** | **0/10 (0%)** | 81.5 | — | **0–9–1** |
| Kubic vs erik/macaria | 100% wins | 51 | 110 | 11–1 |

## Decision

**Revert.** Hypothesis falsified. Push-expand + stickier commitment cut sight
rate from 40% to 0% on the same panel. Restore pre-030 contact params and tip
retip logic (kept `enemy_land_visible` probe from the shared clock work).

Next work: improve belief waypoint quality / route into fog without freezing
macro switches or over-weighting neutral land. Do not raise `CONTACT_LAND_BONUS`
or tip-hold further without a smaller A/B.
