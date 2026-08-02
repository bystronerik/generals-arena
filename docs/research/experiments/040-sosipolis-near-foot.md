# 040 — sosipolis: ContactMCTS belief toward enemy_seen

Bot: [`bots/sosipolis/`](../../../bots/sosipolis/) — **live for GUI check** (panel sight 10%; do not keep until GUI + gate say so).
Baseline: [`../measurements/sosipolis-contact-p1-diag.md`](../measurements/sosipolis-contact-p1-diag.md) (sight 40%).
Prior fails: [`038`](038-sosipolis-belief-prior.md) (decay 0.35 → 20%),
[`039`](039-sosipolis-foot-zero.md) (zero far-bonus → 20%).

## Defect

Belief boosted high BFS distance from the enemy footprint (“behind” by a bad
proxy). That promotes empty fog away from the army. The general is inside
enemy land, so the tip must go **against** the seen army path.

## Hypothesis

Rewrite the footprint term to `1 + CONTACT_BELIEF_NEAR_FOOT / (1 + d_foot)`
over sticky `enemy_seen`, keep mid_pen / home / army-delta, leave ContactMCTS
macros and shallow scorer unchanged. Sight rate ≥ contact-p1’s 40%.

## Revision (in tree for GUI)

| Change | Intent |
| --- | --- |
| `MapMemory.enemy_seen` | sticky army path |
| Drop `0.08 * d_foot` far boost | stop empty opposite fog |
| `CONTACT_BELIEF_NEAR_FOOT=2.0` | promote near seen enemy |
| mid_pen kept | still avoid home-side of front |

## Gate

Seed 0 vs `smoke`: sosipolis win, 4 castles. Passed.

## Measurement

Round: `sosipolis-near-foot1`.

| Round | Sight rate | Median C→S | W–L |
| --- | ---: | ---: | --- |
| contact-p1-diag | **4/10 (40%)** | 373 | 3–7 |
| near-foot1 | **1/10 (10%)** | 645 | 1–9 |

## Status

Panel failed the ≥40% keep gate. Changes are **kept in the working tree** for
GUI verification before revert/keep. Say keep or revert after the GUI check.

```bash
python competition-module/competition/matchup.py \
  bots/sosipolis/run.sh bots/macaria/run.sh \
  --mode competition --seed 1 --gui --fps 8
```
