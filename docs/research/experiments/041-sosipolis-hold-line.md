# 041 — sosipolis: hold hunt line after spent wave

Bot: [`bots/sosipolis/`](../../../bots/sosipolis/) — builds on **040 near-foot** (still in tree).
Baseline: contact-p1 40%; 040 panel 10% (GUI: direction toward army OK).

## Defect (GUI)

After contact the tip walked correctly into the enemy army toward the source.
When that army was spent and the next wave gathered, the commit jumped to a
different direction — often a fully owned corner — and kept sending army there.

### Code cause (near-foot1 traces)

1. Tip arrives / candidate cluster pruned by vision → `_commitment_invalid`
2. Invalidate free-picks `macros[0]` (`cluster` / `split` / `mid_edge`)
3. Jumps of 8–20 cells (e.g. tip-at-old → corner); gather-phase switches too
4. `_opposite_side_evidence` can force early flank switches
5. Dead waypoints (ever_seen, no reveal mass) still scored as mid_edge/split

## Hypothesis

Extend past the tip on the same home→contact line, hold the commit through
gather, reject dead owned-corner waypoints, disable flank force-switch, and
cap voluntary jump distance. Wrong-corner marches stop; sight does not fall
below the 040 GUI-accepted near-foot behaviour.

## Revision

| Change | Intent |
| --- | --- |
| Tip-arrival / invalidate → `extend` or local useful macro | same direction |
| Soften candidate-set empty invalidate | spent fight does not teleport |
| `CONTACT_HOLD_GATHER=True` | no voluntary switch while rebuilding |
| `CONTACT_FORCE_OPP_SWITCH=False` | flanks do not yank path |
| `_waypoint_useful` filter | no march into scouted empty corners |
| `CONTACT_REPLACE_JUMP_MAX=8` | cap voluntary teleports |

040 near-foot belief kept. Speed vs macaria is a separate lever (assault /
tip mass); not changed here.

## Gate

Seed 0 vs `smoke`: sosipolis win, castles built.

## GUI check

```bash
python competition-module/competition/matchup.py \
  bots/sosipolis/run.sh bots/macaria/run.sh \
  --mode competition --seed 1 --gui --fps 8
```

Do not auto-revert; wait for GUI verdict. Optional panel after keep:

```bash
python -m arena.tournaments.competition \
  bots/sosipolis/run.sh bots/macaria/run.sh \
  --round sosipolis-hold-line1 \
  --seeds 0-4 --seat-policy alternate --no-ratings --record --jobs 4
```
