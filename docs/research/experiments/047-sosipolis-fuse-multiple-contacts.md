# 047 — sosipolis: fuse every contact, not just the first

Bot: [`bots/sosipolis/`](../../../bots/sosipolis/) — follows 046.
**Do not auto-revert**; wait for GUI.

## Defect (GUI seed 2 vs macaria)

Seed 2 is the shape that exposes it: our general `(16,0)` bottom-left, theirs
`(16,18)` bottom-right, so the short road is straight along the bottom. Our
search drifts to the middle, meets them at `(7,10)` on turn 80, and from then
on hunts up through the middle — the long way round. Contacts then arrive in
the bottom band (rows 12–17, turns 153–185) and on the right edge (cols 17–19,
turns 320/487), and none of them moves the estimate.

Cause: every directional test in `ContactMCTS` ran through
`mem.first_contact` — one cell, latched for the whole game.
`_on_hunt_axis`, `_far_off_axis`, `_axis_path_waypoint`,
`_extend_past_waypoint`, the belief's axis term and the macro evidence anchor
all measured against it, and `MapMemory.primary_path` only grew along that
same line. A second contact region could raise a candidate's belief a little
but could never change the bearing, and the off-axis vetoes actively demoted
the bearing the later contacts supported.

Measured on seed 2: the true general sat at belief rank **118–145 of ~120–176
candidates** — the bottom third — for the whole game, and never improved.

A second term made it worse in exactly this geometry: candidates closer to us
than the first contact were divided by `1 + 0.35·conf`. With contacts in
several places the general is often nearer to us than the cell we first met
them on — on seed 2 the true general is 18 from home against first contact's
19, so it took that penalty every turn.

## Revision

New component [`components/contact_evidence.py`](../../../bots/sosipolis/components/contact_evidence.py):
two fields over *all* remembered enemy land, as window counts on a
summed-area table (O(H·W) to refresh, O(1) per lookup).

| Change | Intent |
| --- | --- |
| `density(cell)` — enemy land seen around a cell | their base region is thick, a raiding snake is one tile wide; two contact regions near the same candidate add up |
| `openness(cell)` — never-seen area around a cell | their general is in the part of their region we have not scouted |
| `ContactEvidence.anchor` — deepest × thickest contact cell | one anchor standing for every contact, replacing `first_contact` |
| `ContactMCTS._hunt_anchor` feeds all six axis call sites | later contacts can move the bearing |
| Dropped the "closer to us than first contact" penalty | it demoted exactly the cells multi-contact evidence points at |
| `CONTACT_EVIDENCE_RADIUS=4`, `CONTACT_DENSITY_WEIGHT=3.0`, `CONTACT_OPEN_WEIGHT=1.5` | untuned first values; only the aim-error measurement below justifies them |

`first_contact` remains the fallback while there is nothing else to fuse, so
the single-contact behaviour from 046 is unchanged.

## Measurement

**Aim error** — distance from the committed probe waypoint to the true enemy
general, pooled over every post-contact turn of 8 seeds vs `macaria`, same
harness on every arm. This is per-turn over thousands of samples, unlike the
win/loss grid.

| arm | pooled median aim error | turns aiming within 6 cells |
| --- | --- | --- |
| v0 (before 046) | 20 cells | 9% |
| v1 (046 source ray) | 20 cells | 17% |
| v2 (this change) | **11 cells** | **22%** |

Per-seed median aim error, v1 → v2: s4 20→12, s5 8→4, s6 8→8, s7 20→**7**,
s3 25→26, s2 4→6, s1 24→29. Seeds 1 and 7 put the enemy general closer than
`MIN_GENERAL_DISTANCE`=17, so it is never in the candidate set and the aim
metric is not meaningful there.

**Seed 2, the reported scene** — belief rank of the true general goes from the
bottom third to the **top 5–10%** (rank 5–19 of 60–116) from turn 220 on, and
the committed waypoint closes from 15 cells to **3 cells** by turn 197 and
holds.

## Still losing seed 2

Aiming is fixed; converting is not. Along the same trace the tip army falls
30 → 26 → 20 → 14 → 8 → 4 and sits at 2–5 for the rest of the game, so the
probe arrives next to the general with nothing to kill it. That is the
gather/wave economy, not the location estimate, and it is untouched here.

Also still open from 046: macro scores are min–max normalised, so
`CONTACT_SWITCH_RATIO` puts the switch bar above the maximum possible
challenger score; `CONTACT_HOLD_GATHER` blocks roughly a third of commit
re-evaluations.

The win/loss grid stayed inside noise (v0 0/8, v1 1/7, v2 1/7 in the traced
batch; v2 scored 3/8 in an untraced batch of the same seeds). One game per
seed on a deadline-driven bot does not separate these — do not read it as a
verdict.

## Gate

```bash
pytest bots/sosipolis/tests tests -q
python competition-module/competition/matchup.py \
  bots/sosipolis/run.sh bots/macaria/run.sh \
  --mode competition --seed 1
```

Tests: [`bots/sosipolis/tests/test_contact_evidence.py`](../../../bots/sosipolis/tests/test_contact_evidence.py)
— window counts, density thick-vs-snake, openness, refresh epochs, and the
two that pin the defect: the anchor moving to a later thicker region, and a
bearing that the first-contact axis rejects being on-axis once fused.

## GUI

```bash
python competition-module/competition/matchup.py \
  bots/sosipolis/run.sh bots/macaria/run.sh \
  --mode competition --seed 2 --gui --fps 8
```
