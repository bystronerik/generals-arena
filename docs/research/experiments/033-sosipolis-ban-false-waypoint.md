# 033 — sosipolis: ban false contact waypoints

Bot: [`bots/sosipolis/`](../../../bots/sosipolis/).
Evidence: [`../measurements/sosipolis-contact-p1-sighted-vs-blind.md`](../measurements/sosipolis-contact-p1-sighted-vs-blind.md).
Baseline: [`../measurements/sosipolis-contact-p1-diag.md`](../measurements/sosipolis-contact-p1-diag.md).

## Hypothesis

When the tip reaches or owns a probe cell without sighting the general, banning
that cell and switching commitment raises sight rate above contact-p1’s 40%.

## Revision (applied, then reverted)

| Change | Old | New |
| --- | --- | --- |
| `CONTACT_BAN_FALSE_WAYPOINT` | — | True |
| tip at / own waypoint, no sight | soft invalidate | invalidate + ban cell |

## Gate

Seed 0 vs `smoke`: sosipolis win, first castle turn 131 (≥116), no faults.

## Measurement

Round: [`../measurements/sosipolis-ban-wp1.md`](../measurements/sosipolis-ban-wp1.md).

| Round | Sight rate | Contact→sight | W–L |
| --- | ---: | ---: | --- |
| contact-p1-diag | **4/10 (40%)** | 373 | 3–7 |
| ban-wp1 | **1/10 (10%)** | 222 (n=1) | 1–9 |

## Decision

**Revert.** Aggressive false-waypoint bans reduced sight rate. Tip@goal without
sight remains real in the traces, but banning those cells on arrival hurts more
than it helps on this panel — likely thrashing through bans before a useful
probe sticks.

contact-p1 remains the best measured contact owner among 030–033.
