# 019 — sosipolis: dual-MCTS find-and-strike

Bot: [`bots/sosipolis/`](../../../bots/sosipolis/).
Spec: [`../strategies/sosipolis.md`](../strategies/sosipolis.md).

## Hypothesis

Against `smoke` and `fog_scout` on seeds 0–9 (both seats), Sosipolis reaches a
lower median enemy-general sight turn than `fog_scout` and a higher win rate
than `smoke`, while keeping per-move wall time ≤ 100 ms in probe samples.

## Kubic calibration (observational)

Scraped leaderboard replays under `competition-replays/Kubic/` (never written
to `data/games/` or ratings). Command:

```bash
python scripts/replay.py batch Kubic --json
python scripts/replay.py batch Kubic --outcome lose
```

| Cohort | n | median ticks | median contact | median sight | % saw gen | median toward | median peak stack | median sight→end |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| wins | 331 | 202 | 82 | 182 | 100% | 83.9% | 55 | 24 |
| losses | 4 | 196.5 | 54 | 211.5 | 50% | 81.7% | 53 | 68 |
| all played | 336 | 201.5 | 82 | 182 | 99.1% | 83.9% | 55.5 | 24 |

Loss flaw aggregate: 2/4 never saw the enemy general; median closest approach
2. Draw (1): no contact, 1200 ticks.

**Parameter seeds taken from these medians** (see `bots/sosipolis/params.py`):

- `TARGET_CONTACT_TURN` = 82
- `TARGET_SIGHT_TURN` = 182
- `STRIKE_TOWARD_BIAS` = 0.84
- `GATHER_WAVE_HINT` = 4

Loss lesson: failing to sight the general loses. SearchMCTS therefore prioritizes
section-prior frontiers over thin corridors and dead pockets, while still
taking free land each turn.

## Seed grid

- Opponents: `smoke`, `fog_scout`, `macaria`.
- Seeds: 0–2 (sosipolis as player 0).
- Mode: `--mode competition` only.

## Metrics

| Matchup | Games | sosipolis W-L-D | Notes |
| --- | --- | --- | --- |
| sosipolis vs smoke | 3 | 1-0-2 | seed 0 win at turn 385; seeds 1–2 draw at 1200 |
| sosipolis vs fog_scout | 3 | 0-3-0 | losses at turns 618, 400, 903 |
| sosipolis vs macaria | 3 | 0-3-0 | losses at turns 226, 293, 365 |

Verification gate (seed 0 vs `smoke`): finished cleanly, no faults, 0 castles.
After the land-tempo scoring fix, seed 0 vs `smoke` ended with sosipolis
capturing on turn 385.

## Decision

**Keep as research scaffold.** Gate passed. Hypothesis vs `smoke` is only
partially supported (1 win, 2 draws on seeds 0–2). Hypothesis vs `fog_scout`
(earlier sight / higher win rate) is **not** supported on this grid — revise
SearchMCTS / section priors in a later parameter pass. Do not promote to
roster.
