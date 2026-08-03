# 046 — sosipolis: chase the source ray, and let the probe re-decide

Bot: [`bots/sosipolis/`](../../../bots/sosipolis/) — follows 044.
**Do not auto-revert**; wait for GUI.

## Defect (GUI seed 1 vs macaria)

After contact the tip did not follow the enemy army back to its source; it
explored fog in an unrelated direction. Traced on seed 1 (own general `(5,3)`,
enemy general `(4,15)`, first contact `(11,8)` at turn 65). Two independent
causes:

1. **The chase target had no direction term.** `_live_army_target` ranked fog
   behind the front by `(distance from our general, -distance from the front,
   cell)`. Behind a front, that first term ties across the whole arc — at turn
   67 six cells tied at `d_home=17, d_front=6`: `(18,7) (17,8) (16,9) (15,10)
   (14,11) (13,12)`. The coordinate tie-break then took the last-sorting cell,
   always the southernmost. Over 1135 contact turns the chase target never once
   landed in the half of the board holding the enemy general.
2. **The probe commitment could not be re-examined.** `CONTACT_PREP_BUDGET_MS`
   was 3 ms; the belief refresh in `_refresh_cache` costs ~4.5 ms median (14 ms
   worst) on a 21×20 board, so `prepare_contact` expired before generating
   macros on **764/1135** contact turns. One commitment held for **449 turns**
   on `(17,1)` — the corner behind us — while the chase pointed at `(9,17)`.
   Since the committed waypoint drives `state.objective`, the tip re-aim and
   the gather rally, while `search()` marches at the chase, the two halves of
   the bot pulled opposite ways: median separation 7 cells, 496/1135 turns more
   than 8 apart.

The 3 ms budget also made the bot **non-deterministic across runs** — whether
the deadline beat the refresh depended on machine load. Seed 1 gave a draw at
1200 under an instrumented harness and a loss at 403 clean.

## Revision

| Change | Intent |
| --- | --- |
| `_enemy_source_anchor`: deepest visible enemy tile by walking distance from home | read where their chain runs back to |
| `_source_ray_rank`: rank chase fog by depth along home→anchor minus lateral drift | direction decides, not board geometry |
| `CONTACT_CHASE_LATERAL = 1.0` | cells of depth traded per cell off the ray |
| `_home_depth` cached on `terrain_epoch` | the extra BFS is paid once per terrain change |
| `CONTACT_PREP_BUDGET_MS` 3 → 20 | the refresh fits; the commitment is re-examined every turn |

Not addressed here — still open after this change:

- Macro scores are min–max normalised to `[0,1]`, so a challenger is always
  exactly `1.0` while `CONTACT_SWITCH_RATIO` puts the bar at `1.35·last + 0.08`
  = 1.43 for a commitment that was top-ranked when made. 161/723 commit
  evaluations still die on that gate.
- `CONTACT_HOLD_GATHER` blocks 264/723 evaluations.
- The hunt axis is still anchored on `first_contact` (the first enemy *tile
  seen*), not on the inferred source.

## Measurement (seed 1 vs macaria, same harness before/after)

| | before | after |
| --- | --- | --- |
| result | draw at 1200 (no sighting) | **win, general captured turn 835** |
| strike turns / first sight | 0 / never | 46 / turn 789 |
| chase target in the general's half | 0/1135 | 180/724 |
| longest single-waypoint hold | 449 turns | 150 turns |
| chase vs commitment separation | median 7, mean 13.1 | median 0, mean 1.9 |
| `prepare_contact` expired early | 764/1135 (67%) | 0/724 |
| seed 1 repeats | draw 1200 / loss 403 / loss 403 | 835, 835, 835 |

Eight-seed grid vs `macaria` (one game per seed, clean harness — noisy, not a
verdict): 1 win / 7 loss, mean 306 turns → **3 win / 5 loss, mean 403 turns**.
Move latency is unchanged (median 9 ms, max 81 ms against a 100 ms cap; the
80 ms turns are opening SearchMCTS, not contact).

## Gate

```bash
pytest bots/sosipolis/tests tests -q
python competition-module/competition/matchup.py \
  bots/sosipolis/run.sh bots/macaria/run.sh \
  --mode competition --seed 1
```

Tests: `bots/sosipolis/tests/test_contact_chase.py` — the two cases that pinned
the old distance-from-home rule now pin the source ray, including the
equidistant-arc tie.

## GUI

```bash
python competition-module/competition/matchup.py \
  bots/sosipolis/run.sh bots/macaria/run.sh \
  --mode competition --seed 1 --gui --fps 8
```
