# 032 — sosipolis: disable split contact macros

Bot: [`bots/sosipolis/`](../../../bots/sosipolis/).
Evidence: [`../measurements/sosipolis-contact-p1-sighted-vs-blind.md`](../measurements/sosipolis-contact-p1-sighted-vs-blind.md).
Baseline: [`../measurements/sosipolis-contact-p1-diag.md`](../measurements/sosipolis-contact-p1-diag.md).

## Hypothesis

Blind contact-p1 games spend ~59% of contact turns on `split` macros; sighted
games spend ~16% and prefer `cluster`. Disabling split only raises sight rate
above contact-p1’s 40% on seeds 0–4 alternate vs `macaria`.

## Revision (applied, then reverted)

| Change | Old | New |
| --- | --- | --- |
| `CONTACT_ALLOW_SPLIT_MACROS` | (always on) | False |

## Gate

Seed 0 vs `smoke`: **draw at 1200** (first castle 169). Gate win not met.

## Measurement

Round: [`../measurements/sosipolis-no-split1.md`](../measurements/sosipolis-no-split1.md).

| Round | Sight rate | Contact→sight | W–L |
| --- | ---: | ---: | --- |
| contact-p1-diag | **4/10 (40%)** | 373 | 3–7 |
| no-split1 | **3/10 (30%)** | 259 (n=3) | 2–8 |

## Decision

**Revert.** Split removal alone did not raise sight rate. Trace contrast still
stands (blinds over-use split), but cutting split without fixing **false
waypoints** (tip reaches goal without sight) is not enough.

Next defect from the same contrast: when `tip_dist_goal==0` and the general is
still unsighted, invalidate the commitment and forbid re-picking that cell
(false-waypoint park). Keep split enabled for that arm.
