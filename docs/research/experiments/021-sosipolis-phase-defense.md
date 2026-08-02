# 021 — sosipolis: phase-adjusted defense and conversion

Bot: [`bots/sosipolis/`](../../../bots/sosipolis/).
Spec: [`../strategies/sosipolis.md`](../strategies/sosipolis.md).
Prior: [`020-sosipolis-contact-castle.md`](020-sosipolis-contact-castle.md).

## Hypothesis

Folding perimeter defense, soft home bank, and pre-sight / post-sight staging
into Search / Contact / Strike scoring (no fourth MCTS) raises survival vs
`macaria` above sosipolis-r2b (~417 mean turns) and produces non-zero wins on
seeds 0–19 (both seats), while `move_ms` stays ≤ 100.

## Kubic win evidence (observational)

| Signal | Value |
| --- | --- |
| Contact / sight / sight→end | ~82 / 182 / 24 |
| Post-sight toward | ~0.92 |
| Gather waves | ~2 before sight, ~1 after on long games |
| Enemy adjacent to home | ~4% of wins |
| Enemy within 2 of home | ~16% |
| Gen army contact→sight | ~7 → 12 |

## Design

No `phase=defend` and no DefenseMCTS. Hard override = kill shot or imminent
loss only. Perimeter, home bank, staging, and land-tempo live inside each
phase MCTS via [`components/threat.py`](../../../bots/sosipolis/components/threat.py).

## Gate

Seed 0 vs `smoke`: draw at 1200, 1 castle, no faults.

## Macaria grids

| Round | Hash | Games | sosipolis W-L-D | Mean turns | Notes |
| --- | --- | ---: | --- | ---: | --- |
| r2b (baseline) | A+B castle/contact | 20 | 0-20-0 | 416.6 | pre-defense |
| sosipolis-r3 | `43ffe33d4f6f` | 40 | 0-38-2 | 308.7 | defense defaults too heavy |
| sosipolis-r3a | `2af842afb395` | 20 | **1-19-0** | 382.8 | lower contact/strike weights |

### Weight revision (kept)

| Knob | r3 | r3a |
| --- | ---: | ---: |
| `DEFENSE_WEIGHT_CONTACT` | 25 | **15** |
| `DEFENSE_WEIGHT_STRIKE` | 20 | **16** |
| `CONTACT_STAGE_STACK` | 45 | **35** |

## Decision

**Keep** phase-folded defense architecture + r3a weights. First non-zero win vs
`macaria` on this panel. Mean turns still below r2b; win hypothesis only
partially supported. Next: stronger pre-sight staging / conversion without
re-raising defense weights.

Reports: [`../measurements/sosipolis-r3.md`](../measurements/sosipolis-r3.md),
[`../measurements/sosipolis-r3a.md`](../measurements/sosipolis-r3a.md).
