# 028 — sosipolis: contact→sight tempo under clocked shell

Bot: [`bots/sosipolis/`](../../../bots/sosipolis/).
Prior: Stage B probes
[`sosipolis-kubic-clocked-probes`](../measurements/sosipolis-kubic-clocked-probes.md);
sight panels in [`027`](027-sosipolis-contact-hunt.md).

Kubic hunt targets: contact ≤82, sight ≤181.5, contact→sight ≤95.5,
sight rate ~100% in wins.

## Hypothesis

Under the clocked shell, stronger Search/Contact hunt march (no raw chain
bypass, higher hunt priors, earlier flood, tip latched to hunt) cuts median
contact to ≤82 and raises recorded sight rate above 50% with
contact→sight ≤100 on seeds 0–4 alternate vs `macaria`.

## Revision (under clock masks)

| Change | Intent |
| --- | --- |
| `prefer_chain_roots` falls back to scored list | stop sideways raw chain inject |
| `OPEN_FLOOD_START` 27→24; opening force-pass only t≤2 | earlier expand |
| Higher `SEARCH_*` / `CONTACT_*` land/fog/hunt bonuses | push frontier / hunt |
| `HUNT_INTERVAL` 8→4; `HUNT_CONTACT_RADIUS` 6→10; `HUNT_STEP_BONUS` 220→320 | retarget + march |
| Contact/Search wave: tip→hunt first; no soft feed on contact wave | hunt before gather dilutes |
| Brain: latch hunt-facing tip via `select_mass_tip` in search/contact | shared muster |

## Gate

Seed 0 vs `smoke`: sosipolis win at turn 300, first castle turn 128 (≥116),
no faults.

## Recorded sight panel

Round: [`sosipolis-sight3`](../measurements/sosipolis-sight3.md) — seeds 0–4,
`--seat-policy alternate`, `--record` (10 games).

| Round | Sight rate | Median contact | Median sight | Median contact→sight | W-L |
| --- | ---: | ---: | ---: | ---: | ---: |
| sight1 | 1/10 (10%) | 130.5 | 592 | 467 | — |
| sight2 | 0/10 (0%) | 132.0 | — | — | — |
| **sight3** | **1/10 (10%)** | **80.0** | 503 | 437 | **1-9** |
| Kubic wins | ~100% | 82 | 181.5 | 95.5 | — |

## Decision

**Partial keep — contact tempo; revise sight conversion.**

Contact median **80** meets ≤82 (prior ~131). Sight rate stays **10%**;
contact→sight stays **437** on the one sight. Hypothesis on sight rate and
contact→sight is **falsified**.

Next work: belief-hunt quality and strike close after contact (stable hunt
cell, candidate pool / fog approach), not more expand bonuses. Keep the
contact-tempo changes. Do not treat 1–9 as a decision-rule Elo verdict
(n≪200; separate from tip5 / clocked-macaria-10 panels).
