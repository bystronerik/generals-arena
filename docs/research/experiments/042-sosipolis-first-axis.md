# 042 — sosipolis: first-contact axis backup (ignore late flanks)

Bot: [`bots/sosipolis/`](../../../bots/sosipolis/) — builds on 040 near-foot + 041 hold-line.
**Do not auto-revert**; wait for GUI.

## Defect (GUI)

Original enemy signal came from the upper-right. Around turn 58 macaria
expanded into the bottom half; a second contact from bottom-left made
sosipolis treat the bottom as the hunt. The tip stayed in the bottom half and
never returned to the upper-right — where continuing the original direction
would have found the general quickly.

### Code cause

`enemy_seen` / near-foot belief and army-delta treat **all** later enemy cells
equally. A late bottom flank pulls belief and macros off the first-contact
axis. Extend/replace then followed the tip into the flank.

## Revision

| Change | Intent |
| --- | --- |
| `primary_path` (early + on-axis only) | footprint prior ignores late flanks |
| Belief axis bonus / off-axis penalty | mass stays on home→first_contact |
| Flank army-delta × `CONTACT_FLANK_DELTA_SCALE` | weak pull from bottom contact |
| `path` macro + first commit prefer axis | open on original signal |
| Extend snaps to first-contact axis | not from off-axis tip |
| `CONTACT_AXIS_RECOVER` | tip/commit off-axis → snap to `path` |
| Off-axis voluntary switch blocked | no bottom-corner commit |

## Gate

Seed 0 vs `smoke`: sosipolis win.

## GUI

```bash
python competition-module/competition/matchup.py \
  bots/sosipolis/run.sh bots/macaria/run.sh \
  --mode competition --seed 1 --gui --fps 8
```

Watch turn ~58: second bottom contact must not abandon the upper-right hunt.
