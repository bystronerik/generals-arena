# 024 — sosipolis: tip path-gather strike delivery

Bot: [`bots/sosipolis/`](../../../bots/sosipolis/).
Prior: [`023-sosipolis-belief-hunt.md`](023-sosipolis-belief-hunt.md).

## Hypothesis

Path-gather onto one tip, a finish gate before tip march, and Contact assault
stack gating raise kill-ready adjacent turns and tip closing rate vs the seed-4
failure mode (tip at dist 2 with army 28, then drift to 12).

## Design

- `StrikeMCTS`: select tip on path to known general; while tip army `< gen +
  FINISH_MARGIN + STRIKE_PATH_BUFFER * dist`, only feed tip; when ready, march
  tip; land roots only when ready; intercept only imminent.
- `ContactMCTS`: if largest stack `< CONTACT_ASSAULT_STACK`, gather/feed tip
  instead of multi-stack hunt march.

## Gate

Seed 0 vs `smoke`: sosipolis win at turn 337, no faults.

## Macaria

| Round | Games | W-L-D | Mean turns | Notes |
| --- | ---: | --- | ---: | --- |
| sosipolis-hunt1 | 10 | 0-10-0 | 364 | pre-tip |
| sosipolis-tip1 | 10 | **1-9-0** | 323 | win seed 4 A @ 228 |

Recorded A-seat tip closing (largest stack → enemy gen):

| Seed | Sight | Strike turns | Tip close rate | Max tip | Kill-ready |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 439 | 13 | 6/9 | 20 | 0 |
| 3 | 400 | 55 | **42/48** | 36 | 0 (still won) |

Vs hunt1 seed 4 failure: tip close rate was **7%**. Tip feed is working; finish mass
on the tip is still short of Kubic (~55–65) on some games.

## Decision

**Keep.** First alternate-grid win vs Macaria after tip delivery. Next: raise tip
mass (lower finish bar slightly or stronger feed) so kill-ready adjacent > 0
more often.

Report: [`../measurements/sosipolis-tip1.md`](../measurements/sosipolis-tip1.md).
