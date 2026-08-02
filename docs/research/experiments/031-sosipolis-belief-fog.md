# 031 — sosipolis: belief fog-behind waypoint

Bot: [`bots/sosipolis/`](../../../bots/sosipolis/).
Prior: [`030-sosipolis-contact-push.md`](030-sosipolis-contact-push.md) (reverted).
Baseline: [`../measurements/sosipolis-contact-p1-diag.md`](../measurements/sosipolis-contact-p1-diag.md).

## Hypothesis

A smaller belief/waypoint change — stronger behind-front / away-from-home belief
weights, and cluster waypoints as fog-deep belief peaks instead of medoids —
raises sight rate vs contact-p1 on seeds 0–4 alternate vs `macaria`, without
changing land bonuses, tip-hold, or sticky commit thresholds.

## Revision (applied, then reverted)

| Change | Old | New |
| --- | --- | --- |
| `CONTACT_BELIEF_HOME` | hardcoded 0.04 | 0.08 |
| `CONTACT_BELIEF_FOOT` | hardcoded 0.08 | 0.18 |
| cluster waypoint | belief medoid | fog-behind peak (`CONTACT_FOG_WAYPOINT=True`) |

## Gate

Seed 0 vs `smoke`: sosipolis win, first castle turn 180 (≥116), no faults.

## Measurement

Round: [`../measurements/sosipolis-belief-fog1.md`](../measurements/sosipolis-belief-fog1.md).

| Round | Sight rate | Contact median | W–L |
| --- | ---: | ---: | --- |
| contact-p1-diag | **4/10 (40%)** | 82.5 | 3–7 |
| belief-fog1 | **0/10 (0%)** | 81.5 | 0–10 |
| contact-push1 (030) | 0/10 | 81.5 | 0–9–1 |

## Decision

**Revert.** Deeper fog-behind belief/waypoints also collapsed sight rate to 0%
on this panel. Contact-p1 remains the best measured contact owner.

Next work: do **not** retune belief depth next. Inspect contact-p1 trajectories
where sight happened (4 games) vs where it did not, and change only a defect
those traces name (for example tip path blocked, commitment invalidation, or
assault stack gate). Keep `CONTACT_PATH_MODE=shallow`.
