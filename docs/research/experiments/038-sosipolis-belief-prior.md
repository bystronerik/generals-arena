# 038 — sosipolis: ContactMCTS belief prior near enemy_seen

Bot: [`bots/sosipolis/`](../../../bots/sosipolis/) — **reverted**.
Baseline: [`../measurements/sosipolis-contact-p1-diag.md`](../measurements/sosipolis-contact-p1-diag.md) (sight 40%).
Keeps ContactMCTS (macros + sticky commit + shallow scorer). Belief-only change.

## Defect

Contact belief boosted **high** BFS distance from the enemy footprint. That
pulls macros into empty fog opposite the army path — the opposite of the
intended hunt and of macaria’s `hunt_prior_decay`.

## Hypothesis

Replace that boost with a macaria-style penalty on distance from sticky
`enemy_seen`. Sight rate stays ≥ contact-p1’s 40% on seeds 0–4 alternate vs
`macaria`. If so, arm 2 turns on `CONTACT_NEAR_VISIBLE_BONUS`.

## Revision (arm 1, tried then reverted)

| Change | Intent |
| --- | --- |
| `MapMemory.enemy_seen` | remember every enemy cell ever seen |
| Footprint BFS from `enemy_seen` | sticky army path |
| Drop `w *= 1 + 0.08 * d_foot` | stop empty opposite fog |
| `w /= 1 + CONTACT_BELIEF_PRIOR_DECAY * d_foot` (`0.35`) | prefer near army path |
| `CONTACT_NEAR_VISIBLE_BONUS=0` | arm 2 not run |

Macros, commit sticky, tip invalidate, and shallow MCTS unchanged.

## Gate

Seed 0 vs `smoke`: sosipolis win, 4 castles. Passed.

## Measurement

Round: `sosipolis-belief-prior1`.

| Round | Sight rate | Median C→S | W–L |
| --- | ---: | ---: | --- |
| contact-p1-diag | **4/10 (40%)** | 373 | 3–7 |
| belief-prior1 | **2/10 (20%)** | 187 | 1–9 |

## Verdict

**Revert.** Arm 1 failed the ≥40% keep gate. Arm 2 (near-visible fog bonus)
was not run. The high-`d_foot` boost is still wrong in intent; a milder decay
or combining prior with reveal-mass / mid_pen retune may be needed before the
near-visible arm. ContactMCTS structure stays the right place to fix this.
