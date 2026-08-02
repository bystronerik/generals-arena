# 027 — sosipolis: contact hunt unlock (no tip-feed lock)

Bot: [`bots/sosipolis/`](../../../bots/sosipolis/).
Prior: [`026-sosipolis-kubic-align.md`](026-sosipolis-kubic-align.md).
Kubic hunt targets: contact ≤82, sight ≤181.5, contact→sight ≤95.5, sight rate ~100% in wins.

## Hypothesis

Disabling hard `_tip_feed_wave` in contact and lowering `CONTACT_ASSAULT_STACK`
to 15 (soft ContactMCTS staging only) raises recorded sight rate above 50% and
cuts contact→sight below 100 turns on seeds 0–4 alternate vs `macaria`.

## Parameter / logic revision

| Change | Old | New |
| --- | --- | --- |
| `brain._tip_feed_wave` contact branch | exclusive feed if tip &lt; stack | always `None` (strike-only override) |
| `ContactMCTS` under assault | early exclusive `feed_tip_action` | hunt roots always available; soft feed competes |
| `CONTACT_ASSAULT_STACK` | 30 | 15 |
| under-ready `hunt_scale` | 0.15 | 0.55 |

Content hash: tip4 `77b19da7ac39` → treat `1db7d24fcb46`.

## Gate

Seed 0 vs `smoke`: sosipolis win at turn 360, 1 castle (turn 98), no faults.

## Recorded sight panel

Round: `sosipolis-sight2` — seeds 0–4, `--seat-policy alternate`, `--record`
(10 games). Compare to `sosipolis-sight1` (tip4 params).

| Round | Sight rate | Median contact | Median sight (if any) | Median contact→sight |
| --- | ---: | ---: | ---: | ---: |
| sight1 (tip4) | 1/10 (10%) | 130.5 | 592 | 467 |
| sight2 (this) | **0/10 (0%)** | 132.0 | — | — |
| Kubic wins | ~100% | 82 | 181.5 | 95.5 |

Long contact without sight still occurs (e.g. seed 0 B: 502 contact turns).
Several games end before turn 185 with short contact windows.

## Winrate grid

Round: [`sosipolis-tip5`](../measurements/sosipolis-tip5.md) — seeds 0–19
alternate (40 games), no record.

| Round | Games | sosipolis W-L-D | Mean turns |
| --- | ---: | --- | ---: |
| tip4 | 40 | 5-35-0 (12.5%) | 245 |
| tip5 | 40 | **7-33-0 (17.5%)** | 231 |

Formal Elo contrast vs tip4 is **unproven** (≪ 200 games/arm; separate rounds).

## Decision

**Revise — sight hypothesis falsified.** Removing the contact tip-feed lock did
not raise sight rate (10% → 0% on the matched recorded panel). Contact median
stayed ~132 (Kubic 82). Raw tip5 win rate rose slightly (12.5% → 17.5%) but
that is not a keep signal for this hunt experiment.

Keep the tip4 castle / `STRIKE_MIN_TIP` programme. Next hunt work should target
**search expand tempo to contact ≤82** and belief-hunt quality (candidate pool /
`HUNT_CONTACT_RADIUS`), not another tip-feed knob. Revert or retain the
contact unlock pending a follow-up ask — default recommendation: **revert**
`CONTACT_ASSAULT_STACK` / contact tip-feed / ContactMCTS exclusive-feed removal
to tip4 (`77b19da7ac39` behaviour) before the next hunt design, since the
measured metric moved the wrong way.
