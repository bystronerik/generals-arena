# 029 — sosipolis: ContactMCTS owns post-contact hunt

Bot: [`bots/sosipolis/`](../../../bots/sosipolis/).
Prior: [`028-sosipolis-contact-sight-tempo.md`](028-sosipolis-contact-sight-tempo.md).
Baseline: [`../measurements/sosipolis-sight3.md`](../measurements/sosipolis-sight3.md).

## Hypothesis

Moving post-contact target ownership into ContactMCTS with belief-driven probe
macros, sticky commitment, and a shallow path scorer raises sight rate and cuts
contact→sight toward the Kubic gate (≤95.5 median) vs sight3 on seeds 0–4
alternate vs macaria.

## Design (Phase 1)

- `MapMemory`: per-cell `last_seen_turn`, `last_enemy_army`, `enemy_army_delta`;
  epoch counters for invalidate-on-change caches.
- `ContactMCTS.prepare_contact()`: belief over candidates, ≤6 probe macros
  (`cluster` / `frontier` / `mid_edge` / `split`), adaptive sticky commitment.
- Contact probe waypoint becomes `state.objective` (clock masks unchanged).
- Shallow tip/feeder path scorer replaces flat UCT during contact.
- One tip + gather feeders (Kubic); no multi-stack hunt march.
- `hunt_target()` remains pre-contact Search only.

## Gate

Competition seed 0 vs macaria: match finishes (macaria win turn 756), no faults.

## Results

Report: [`../measurements/sosipolis-contact-p1-diag.md`](../measurements/sosipolis-contact-p1-diag.md).

| Round | Sight rate | Median contact | Median c→sight | W-L |
| --- | ---: | ---: | ---: | ---: |
| sight3 | 1/10 | 80.0 | 437 | 1-9 |
| **contact-p1-diag** | **4/10** | **82.5** | **373** | **3-7** |

## Decision

**Keep Phase 1 belief + sticky probes.** Sight rate moved 10% → 40%; contact→sight
improved but remains far above ≤95.5. Commitment is stickier than sight3 hunt
churn (median distinct waypoints 7.5 vs 12.5).

## Phase 2 comparison

Round: [`../measurements/sosipolis-contact-p2-diag.md`](../measurements/sosipolis-contact-p2-diag.md).

| Round | Path layer | Sight | Contact | c→sight | W-L |
| --- | --- | ---: | ---: | ---: | ---: |
| sight3 | prior hunt + UCT | 1/10 | 80.0 | 437 | 1-9 |
| contact-p1-diag | shallow scorer | **4/10** | 82.5 | **373** | **3-7** |
| contact-p2-diag | macro MCTS | 1/10 | 84.0 | 421 | 1-9 |

**Keep `CONTACT_PATH_MODE=shallow`.** Macro MCTS did not improve sight conversion
on this panel.

