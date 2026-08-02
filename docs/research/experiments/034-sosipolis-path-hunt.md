# 034 — sosipolis: hunt along remembered enemy path

Bot: [`bots/sosipolis/`](../../../bots/sosipolis/) — **reverted**.
Baseline: [`../measurements/sosipolis-contact-p1-diag.md`](../measurements/sosipolis-contact-p1-diag.md) (sight 40%).
GUI observation: after contact, sosipolis often explores opposite the enemy
approach and chases new flanks instead of the path toward their general.

## Hypothesis

Anchoring contact belief on `first_contact` + every ever-seen enemy cell, and
preferring candidates past that path (home → contact → enemy interior) while
not chasing off-axis fresh expansions, raises sight rate above contact-p1’s 40%
on seeds 0–4 alternate vs `macaria`.

## Revision (tried, then reverted)

| Change | Intent |
| --- | --- |
| `MapMemory.ever_enemy` | remember every enemy cell ever seen |
| Belief: home→first_contact axis | prefer past contact toward enemy general |
| Belief: near `ever_enemy` path | stop rewarding empty opposite fog |
| Weak on-axis fringe only | stop distraction by new side expansions |
| `CONTACT_FORCE_OPP_SWITCH=False` | stop sticky commit jumping to new flanks |

## Gate

Seed 0 vs `smoke`: sosipolis win, 4 castles. Passed before panel.

## Measurement

Round: `sosipolis-path-hunt1` (seeds 0–4 alternate vs `macaria`, `--record`).

| Round | Sight rate | Median C→S | W–L |
| --- | ---: | ---: | --- |
| contact-p1-diag | **4/10 (40%)** | 373 | 3–7 |
| path-hunt1 | **2/10 (20%)** | 295.5 | 2–8 |

Sighted games: seed1 seat A (C→S 574, win); seed4 seat B (C→S 17, win).

## Verdict

**Revert.** Sight rate fell 40% → 20%. Axis alignment and weaker fringe chase
did not raise conversion. The GUI “wrong direction” claim may need a different
lever than belief reweight (for example tip selection, waypoint sticky, or
conveyor push direction after contact).

Keep this note; bot bytes restored to pre-034.
