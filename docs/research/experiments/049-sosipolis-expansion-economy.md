# 049 — sosipolis: expansion is a transit problem

Bot: [`bots/sosipolis/`](../../../bots/sosipolis/) — follows 048.
**Do not auto-revert**; wait for GUI.

## Defect

046–048 all worked on the hunt. None of them moved the win rate against
`macaria` (20% → 20% → 20% over 20 seeds). Measuring the games instead of the
hunt says why:

| | sosipolis | macaria | Kubic (spec §3/§4) |
| --- | --- | --- | --- |
| land@50 | **16** | 23 | 24 (band 20–25) |
| peak land share | **0.18** | 0.41 | ~0.24 (p90 0.31) |

We were losing the economy from turn 50 and never recovering. The hunt could
not matter.

**Why 16.** RULES §04: the general makes one army every *other* turn, so ~24
army exist by t=50 and each neutral cell costs exactly one — land@50 is capped
near 25 and Kubic converts ~96% of it. Each produced unit therefore gets about
one spare turn of travel before it costs a capture.

`decide_opening` handed the opening to SearchMCTS, whose objective is
`_fog_frontier` — which explicitly prefers frontier *away from home*
(`s += 0.02 * dist_from_home`). That draws a one-cell-wide tendril, so the
only capturable cells end up five to seven steps from the only stack that can
take them. Per-move trace of seed 0, turns 3–50:

| | before | after |
| --- | --- | --- |
| captures | 17 | **22** |
| moves across our own land | 30 | 25 |

Same disease after the opening: 191 of 280 post-50 moves walked over ground we
already owned, hauling the corner general's production fourteen steps to a
distant tip.

## Revision

New [`components/expand.py`](../../../bots/sosipolis/components/expand.py):
continue the chain if it can still take a cell (a snake pays transit once),
else take the capturable cell nearest the general through our own land, else
walk the biggest stack toward the nearest neutral.

| Change | Intent |
| --- | --- |
| `decide_opening` runs the expansion script, not SearchMCTS | Kubic §3 is a script; the opening is transit, not search |
| Pre-contact wave takes land when it can | Kubic captures ~0.5 neutrals/tick before contact — one per unit made |
| `far_haul_capture` on gather ticks, `GATHER_LOCAL_HAUL = 8` | a stack 8+ steps from the muster is worth more as land than as the fraction of itself that arrives |
| `exclude` on the expansion helpers | keeps the assault tip out of the economy |

## Measurement

10 recorded games vs `macaria`, per stage:

| | wins | land@50 | our peak land | macaria peak |
| --- | --- | --- | --- | --- |
| before | 2/10 | 16 | 0.18 | 0.41 |
| + opening script | 0/10 | 22 | 0.15 | 0.42 |
| + pre-contact wave | 3/10 | 22 | 0.19 | 0.36 |
| + far-haul capture | 3/10 | 22 | **0.23** | 0.37 |

20 seeds, clean harness: **4/20 (20%) → 5/20 (25%)**.

Land is now at Kubic's numbers. The win rate is not, and 5-vs-4 over 20 seeds
is inside noise — the economy was necessary, not sufficient.

## Rejected, with measurements

- **Expanding during contact** (take neutrals while the tip is under weight).
  Our land reached 0.22 and we lost *more*: macaria's peak went 0.36 → 0.51
  once the pressure came off, 0/10 with the strict `tip_is_ready` gate and
  2/10 with a loose one. Contesting their ground is what holds them down.
- **Home muster** — haul nearby army into the general when an enemy stack that
  beats it closes in. Motivated by the spec's own §7 exploit surface and by a
  lost game where 113 army sat within three steps of our general, one per
  tile, while it held 43 and an 80-army stack walked in. Measured 3/20 against
  5/20: it pulls army home and the offence dies. Not kept.
- **`MIN_GENERAL_DISTANCE = 17` is not a bug.** Two of five no-sight games had
  their general closer than 17, which looked like the candidate set being
  structurally blind. Over 60 seeds the engine's minimum BFS separation is
  exactly 17. The constant is right.

## What actually gates the win rate

The funnel over 10 games:

| stage | rate |
| --- | --- |
| contact | 10/10 (median t≈97; Kubic 82) |
| **sight the enemy general** | **5/10** |
| win given sight | 3/5 |

Kubic never kills without sight (§6) and sights in essentially every win.
Sight means owning a cell adjacent to their general, so it is army projection
into their half — and macaria holds ~2× our army there. That is the next
target, not the hunt geometry, which 046–047 already put within 11 cells.

## Gate

```bash
pytest bots/sosipolis/tests tests -q
python competition-module/competition/matchup.py \
  bots/sosipolis/run.sh bots/macaria/run.sh \
  --mode competition --seed 5
```

Tests: [`bots/sosipolis/tests/test_expand.py`](../../../bots/sosipolis/tests/test_expand.py)
— neutral castles are not economy, transit-nearest capture beats a far one,
a live chain outranks both, tip exclusion, and the haul threshold.

## GUI

```bash
python competition-module/competition/matchup.py \
  bots/sosipolis/run.sh bots/macaria/run.sh \
  --mode competition --seed 0 --gui --fps 8
```
