# 039 — sosipolis: zero CONTACT_FOOT_FAR_BONUS

Bot: [`bots/sosipolis/`](../../../bots/sosipolis/) — **reverted**.
Baseline: [`../measurements/sosipolis-contact-p1-diag.md`](../measurements/sosipolis-contact-p1-diag.md) (sight 40%).
Prior fail: [`038-sosipolis-belief-prior.md`](038-sosipolis-belief-prior.md) (prior decay 0.35 → 20%).

## Defect

`w *= 1 + 0.08 * d_foot` boosted candidates far from the enemy footprint and
relatively penalized near-army fog. The general sits inside enemy land, so the
tip must approach that army; that term fights the hunt.

## Hypothesis

Set `CONTACT_FOOT_FAR_BONUS=0` (no footprint-distance reweight). Keep mid_pen,
home distance, army-delta, and all ContactMCTS macros / sticky / shallow
scorer. Sight rate ≥ contact-p1’s 40%.

## Revision (tried, then reverted)

| Param | Was | Now |
| --- | ---: | ---: |
| `CONTACT_FOOT_FAR_BONUS` | (hardcoded 0.08) | **0.0** |

## Gate

Seed 0 vs `smoke`: sosipolis win, 3 castles. Passed.

## Measurement

Round: `sosipolis-foot-zero1`.

| Round | Sight rate | Median C→S | W–L |
| --- | ---: | ---: | --- |
| contact-p1-diag | **4/10 (40%)** | 373 | 3–7 |
| foot-zero1 | **2/10 (20%)** | 472.5 | 1–9 |

## Verdict

**Revert.** Zeroing the far-footprint boost alone cut sight 40% → 20%. On this
10-game panel the hardcoded `0.08` term still correlates with better sight
than removing it, even though the geometry of that term is wrong for
“cross the army.” Contact-p1 remains the keep baseline among 034–039.
