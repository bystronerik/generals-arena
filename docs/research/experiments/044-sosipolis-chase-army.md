# 044 — sosipolis: tip marches to live army, not abstract path fog

Bot: [`bots/sosipolis/`](../../../bots/sosipolis/) — builds on 040–043.
**Do not auto-revert**; wait for GUI.

## Defect (GUI)

Tip still ignored visible enemy. Cause: wave roots did `step_toward(tip,
committed_path_waypoint)` with prior ~640+, while adjacent enemy captures
scored ~45. Axis `path` macros won commits; movement never chased the army.

## Revision

| Change | Intent |
| --- | --- |
| `_live_army_target` (fog behind visible enemy / strongest stack) | macaria-style chase cell |
| `search()` overrides hunt with live target while enemy visible | tip walks at the army |
| `chase` macro preferred over `path` | commit follows army |
| `CONTACT_CHASE_STEP_BONUS=750` | chase step beats fog march |
| `CONTACT_ENEMY_BONUS=420` | adjacent fight beats fog |

Path / first-contact axis remains the backup when no enemy is visible.

## Gate

Seed 0 vs `smoke`: sosipolis win.

## GUI

```bash
python competition-module/competition/matchup.py \
  bots/sosipolis/run.sh bots/macaria/run.sh \
  --mode competition --seed 1 --gui --fps 8
```
