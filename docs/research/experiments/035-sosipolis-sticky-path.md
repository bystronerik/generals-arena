# 035 — sosipolis: sticky waypoint on enemy path

Bot: [`bots/sosipolis/`](../../../bots/sosipolis/) — **reverted**.
Baseline: [`../measurements/sosipolis-contact-p1-diag.md`](../measurements/sosipolis-contact-p1-diag.md) (sight 40%).
Prior fail: [`034-sosipolis-path-hunt.md`](034-sosipolis-path-hunt.md) (belief reweight → 20%).

GUI: after contact, sosipolis explores opposite the enemy approach and chases
new flanks. 034 changed belief only; this change stuck the **probe waypoint
and tip** to the first-contact path.

## Hypothesis

Seeding the first commit past `first_contact` along home→contact, holding that
waypoint longer, ignoring opposite-side force-switches, remembering every
ever-seen enemy cell for footprint align, and holding the tip longer raises
sight rate above contact-p1’s 40% on seeds 0–4 alternate vs `macaria`.

## Revision (tried, then reverted)

| Change | Intent |
| --- | --- |
| `MapMemory.ever_enemy` | remember every enemy cell ever seen |
| `path` probe macro | first waypoint past first contact |
| Prefer on-axis macros | do not open on empty opposite fog |
| `CONTACT_COMMIT_MIN_TURNS=40` | hold path longer |
| `CONTACT_SWITCH_RATIO=1.8` | harder to leave path |
| `CONTACT_FORCE_OPP_SWITCH=False` | no flank force-switch |
| Off-axis switch needs 2.5× score | stop distraction by new expansions |
| `CONTACT_TIP_HOLD=40` | tip stays on committed waypoint |

Belief formula unchanged (034 lesson).

## Gate

Seed 0 vs `smoke`: sosipolis win, 4 castles. Passed before panel.

## Measurement

Round: `sosipolis-sticky-path1` (seeds 0–4 alternate vs `macaria`, `--record`).

| Round | Sight rate | Median C→S | W–L |
| --- | ---: | ---: | --- |
| contact-p1-diag | **4/10 (40%)** | 373 | 3–7 |
| path-hunt1 (034) | 2/10 (20%) | 295.5 | 2–8 |
| sticky-path1 (035) | **3/10 (30%)** | 201 | 2–8 |

Sighted: seed3 seat B (C→S 257, loss); seed4 both seats (wins, C→S 163 / 201).

## Verdict

**Revert.** Sight rate 30% < contact-p1 40%. Sticky path commit + tip hold
did not beat the baseline. Axis lock may starve useful cluster probes that
contact-p1 still converts.

Keep this note; bot bytes restored to pre-035.
