# 036 — sosipolis: block empty-fog waypoint teleports

Bot: [`bots/sosipolis/`](../../../bots/sosipolis/) — **reverted**.
Baseline: [`../measurements/sosipolis-contact-p1-diag.md`](../measurements/sosipolis-contact-p1-diag.md) (sight 40%).
Prior fails: [`034`](034-sosipolis-path-hunt.md) belief, [`035`](035-sosipolis-sticky-path.md) axis lock.

## Defect (GUI + contact-p1 traces)

After contact the tip often follows the enemy army, then the commit jumps
8–25 cells into empty fog (example: `(18,3)→(18,17)` when tip was at the old
waypoint). Cause: tip-arrival / invalidate then picks `macros[0]` with no
locality or footprint filter; opposite-side evidence also force-switches to
new flanks.

Far jumps in contact-p1 (d≥8): common across blinds; macros mix
`cluster` / `split` / `mid_edge`; several are tip-at-old then teleport.

## Hypothesis

Preferring macros near `ever_enemy` footprint, capping switch jumps at
`CONTACT_JUMP_MAX`, extending past a reached empty waypoint instead of a free
pick, and disabling flank force-switch raises sight rate above contact-p1’s 40%.

## Revision (tried, then reverted)

| Change | Intent |
| --- | --- |
| `MapMemory.ever_enemy` | remember every enemy cell ever seen |
| Footprint BFS from `ever_enemy` | army-path locality |
| `_pick_macro` near foot / near old wp | no empty-fog teleport |
| Tip-arrival → `extend` macro | continue past reached cell |
| Hard jump cap on voluntary switch | d ≤ 8 from sticky wp |
| `CONTACT_FORCE_OPP_SWITCH=False` | new flanks do not yank hunt |

## Gate

Seed 0 vs `smoke`: sosipolis win, 3 castles. Passed.

## Measurement

Round: `sosipolis-no-teleport1`.

| Round | Sight rate | Median C→S | W–L |
| --- | ---: | ---: | --- |
| contact-p1-diag | **4/10 (40%)** | 373 | 3–7 |
| no-teleport1 | **1/10 (10%)** | 109 | 1–9 |

## Verdict

**Revert.** Footprint locality + jump cap cut teleports but also blocked useful
fog probes past the visible army (toward the general). Sight collapsed.

The GUI defect is real (trace-confirmed teleports after tip arrival). A keepable
fix must allow fog **beyond** the army path without allowing map-wide jumps
to empty corners — e.g. extend-only on tip arrival, without a hard foot filter
on all macros, or a one-sided “past first_contact” cone with no jump-back.
