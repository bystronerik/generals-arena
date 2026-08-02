# 043 — sosipolis: chase live army + faster opening

Bot: [`bots/sosipolis/`](../../../bots/sosipolis/) — builds on 040–042.
**Do not auto-revert**; wait for GUI.

## Defects (GUI)

1. After 042 axis lock, the tip stopped following the enemy army — recover
   snapped away whenever the tip left a narrow first-contact cone.
2. Opening land is too slow: by turn ~45 macaria has ~2× land. Later contact
   makes the first-contact direction less trustworthy.

## Revision

| Change | Intent |
| --- | --- |
| Recover only if lateral ≥ 8 **and** tip not near visible enemy | keep fighting the army |
| Footprint = primary_path ∪ **visible** enemy | chase live stacks |
| `CONTACT_CHASE_VISIBLE` belief boost | prefer fog/candidates by the army |
| Softer axis bonuses; confidence from `first_contact_turn` | early contact → stronger axis |
| `OPEN_FLOOD_START` 24→12 | expand out of home half sooner |
| `SEARCH_LAND/FOG/ENEMY_BONUS` up | more aggressive pre-contact land |

## Gate

Seed 0 vs `smoke`: sosipolis win.

## GUI

```bash
python competition-module/competition/matchup.py \
  bots/sosipolis/run.sh bots/macaria/run.sh \
  --mode competition --seed 1 --gui --fps 8
```

Check: tip still engages the army; land at ~45 closer to macaria; late bottom
flank still does not fully abandon an early upper-right signal.
