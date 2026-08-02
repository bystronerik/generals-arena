# 045 — sosipolis: enemy general latch + kill shot

Bot: [`bots/sosipolis/`](../../../bots/sosipolis/)
**Do not auto-revert**; wait for GUI.

## Defect (GUI turn ~572)

Sosipolis had 12 army adjacent to an enemy general with 8 and did not capture.
Root cause: `MapMemory.update` reused name `t` for the primary-path axis
projection. That overwrote the cell type before the `t == T_GENERAL` latch.
Off-axis (or same-turn first-sight) generals never entered
`enemy_general`, so `_kill_shot` returned None and ContactMCTS kept chasing
land.

## Revision

| Change | Intent |
| --- | --- |
| Rename projection to `axis_t` | type latch always sees `T_GENERAL` |
| `_kill_shot` also scans visible `owner==2` generals | belt-and-suspenders kill |
| Deathtouch `need=1` (was 2) | RULES §07: one unit wins; leave-1 → army ≥ 2 |

Fixtures + tests: `bots/sosipolis/tests/boards.py`,
`bots/sosipolis/tests/test_logic.py` (outside the content hash).

## Gate

```bash
pytest bots/sosipolis/tests -q
python competition-module/competition/matchup.py \
  bots/sosipolis/run.sh bots/smoke/run.sh \
  --mode competition --seed 0
```

Fixtures: `bots/sosipolis/tests/boards.py` (`kill_12v8_t572`,
`off_axis_general_sight`).

## GUI

```bash
python competition-module/competition/matchup.py \
  bots/sosipolis/run.sh bots/macaria/run.sh \
  --mode competition --seed 1 --gui --fps 8
```
