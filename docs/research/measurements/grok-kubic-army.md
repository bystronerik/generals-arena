# Kubic army management & routing (fit wins)

Analyst #3. Corpus: **341** fit wins (of 341 listed); holdout excluded; 11 losses skimmed.

Phase buckets: `expansion` / `contest` / `post_sight` (see JSON `meta.thresholds.phase_definition`).

## Top rules

### R1_tip_not_general_bank [MEASURED]

Tip stack holds more army than the general bank across the game (median mean_tip_frac ~2x mean_gen_frac). Most army still sits on ordinary land; castles hold a small extra bank.

Thresholds / numbers:

```
{
  "mean_tip_frac_median": 0.25146804547320817,
  "mean_gen_frac_median": 0.1326788155057153,
  "mean_castle_frac_median": 0.054259208968956495
}
```

### R2_gather_before_sight [MEASURED]

Every fit win has >=1 gather_wave (max-stack grows while moving >= 5 ticks). Most waves start before first_general_sight; waves begin near home and end closer to the enemy general.

Thresholds / numbers:

```
{
  "gather_min_ticks": 5,
  "unit": "ticks of consecutive growing moves",
  "pre_sight_wave_share": 0.7247990105132962,
  "waves_per_game_median": 4.0,
  "zero_wave_wins": 0,
  "wave_start_dist_home_median": 2.0,
  "wave_end_dist_enemy_gen_median": 13.0
}
```

### R3_sustained_move_start_size [MEASURED]

Sustained tip marches (>= 5 consecutive max-stack moves) typically start with tip army in the mid-teens (p25~9, p75~25).

Thresholds / numbers:

```
{
  "sustained_move_min_ticks": 5,
  "unit": "consecutive max-stack move ticks",
  "streak_start_army_median": 15.0,
  "streak_start_army_p25": 9.0,
  "streak_start_army_p75": 25.0
}
```

### R4_full_over_half [INFERRED]

Among classifiable sends, leave-1 (full) dominates (~98.5%); half is ~1.5%. Half is slightly more common in expansion than contest/post_sight. Ambiguity (unknown+multi) is ~32% of all ticks — treat half/full rates as conditional on successful classification.

Thresholds / numbers:

```
{
  "note": "Rates conditioned on inferred full|half; ambiguity (unknown+multi) is high.",
  "full_of_actionable": 0.9848355767592435,
  "half_of_actionable": 0.015164423240756517,
  "ambiguity_rate": 0.31737545873598394
}
```

### R5_home_reserve_modest [MEASURED]

General peak bank median ~27 army. At contact median gen~8; at sight median gen~14 — home is not the strike reservoir.

Thresholds / numbers:

```
{
  "units": "army on general cell",
  "max_gen_army": {
    "n": 341,
    "min": 12.0,
    "p10": 18.0,
    "p25": 23.0,
    "median": 27.0,
    "p75": 33.0,
    "p90": 49.0,
    "max": 121.0,
    "mean": 30.65982404692082
  },
  "gen_at_contact": {
    "n": 341,
    "min": 1.0,
    "p10": 4.0,
    "p25": 5.0,
    "median": 8.0,
    "p75": 11.0,
    "p90": 14.0,
    "max": 57.0,
    "mean": 8.736070381231672
  },
  "gen_at_sight": {
    "n": 341,
    "min": 2.0,
    "p10": 9.0,
    "p25": 11.0,
    "median": 14.0,
    "p75": 19.0,
    "p90": 26.0,
    "max": 62.0,
    "mean": 16.21700879765396
  }
}
```

### R6_tip_mass_at_sight_and_kill [MEASURED]

Tip at first_general_sight median ~23 army; peak tip in the last 20 ticks median ~53 (strike consolidation). Tip at the final tick is lower (median ~24) after the kill spend.

Thresholds / numbers:

```
{
  "near_kill_window_ticks": 20,
  "tip_at_sight": {
    "n": 341,
    "min": 5.0,
    "p10": 12.0,
    "p25": 15.0,
    "median": 23.0,
    "p75": 33.0,
    "p90": 51.0,
    "max": 175.0,
    "mean": 28.44574780058651
  },
  "tip_near_kill": {
    "n": 341,
    "min": 17.0,
    "p10": 33.0,
    "p25": 43.0,
    "median": 53.0,
    "p75": 75.0,
    "p90": 100.0,
    "max": 249.0,
    "mean": 62.17008797653959
  },
  "tip_at_kill": {
    "n": 341,
    "min": 7.0,
    "p10": 12.0,
    "p25": 15.0,
    "median": 24.0,
    "p75": 33.0,
    "p90": 49.0,
    "max": 111.0,
    "mean": 27.070381231671554
  }
}
```

## Key distributions

### Where army sits (fraction of total army)

| Metric | n | median | p25 | p75 | mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| mean_tip_frac | 341 | 0.251 | 0.231 | 0.277 | 0.255 |
| mean_gen_frac | 341 | 0.133 | 0.118 | 0.150 | 0.135 |
| mean_castle_frac | 341 | 0.054 | 0.016 | 0.075 | 0.050 |
| tip_frac_at_sight | 341 | 0.164 | 0.129 | 0.217 | 0.176 |
| gen_frac_at_sight | 341 | 0.105 | 0.079 | 0.134 | 0.109 |
| tip_frac_near_kill | 341 | 0.313 | 0.263 | 0.355 | 0.313 |

### Home reserve (army units on general)

| Metric | n | median | p25 | p75 | mean | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| max_gen_army | 341 | 27.0 | 23.0 | 33.0 | 30.7 | 121.0 |
| gen_at_contact | 341 | 8.0 | 5.0 | 11.0 | 8.7 | 57.0 |
| gen_at_sight | 341 | 14.0 | 11.0 | 19.0 | 16.2 | 62.0 |
| tip_at_contact | 341 | 15.0 | 10.0 | 20.0 | 18.2 | 174.0 |

### Tip mass at sight / near kill (army units)

| Metric | n | median | p25 | p75 | mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| tip_at_sight | 341 | 23.0 | 15.0 | 33.0 | 28.4 |
| tip_near_kill | 341 | 53.0 | 43.0 | 75.0 | 62.2 |
| tip_at_kill | 341 | 24.0 | 15.0 | 33.0 | 27.1 |

### Gather waves

- Waves per game: median **4.0** (mean 4.74), zero-wave wins: **0** (0.0%).
- Pre-sight wave count share: **72.5%** (1172 pre / 445 post).
- Wave length: median **8** ticks (p25=6, p75=12).
- Wave army: from median **24** → to median **40**.
- Start dist to home (Manhattan): median **2.0**; end dist to enemy general: median **13.0**.

### Sustained tip marches (>= 5 moves)

- n=2933; start army median **15** (p25=9, p75=25, mean=19.4).
- Streak length median **10** ticks.

### Half vs full sends

- Of classifiable sends (full|half): **98.5%** full, **1.5%** half.
- Ambiguity rate (unknown+multi over all ticks): **31.7%**.

| Phase | n_ticks | full/(full+half) | half/(full+half) | ambiguity |
| --- | ---: | ---: | ---: | ---: |
| expansion | 28751 | 96.6% | 3.4% | 27.4% |
| contest | 36550 | 99.7% | 0.3% | 33.7% |
| post_sight | 13448 | 99.3% | 0.7% | 35.6% |

Overall kind rates:

```
{
  "build_suspect": {
    "count": 24,
    "rate": 0.00030476577480348957
  },
  "full": {
    "count": 52020,
    "rate": 0.6605798168865636
  },
  "half": {
    "count": 801,
    "rate": 0.010171557734066465
  },
  "multi": {
    "count": 23199,
    "rate": 0.2945942170694231
  },
  "pass": {
    "count": 911,
    "rate": 0.011568400868582459
  },
  "unknown": {
    "count": 1794,
    "rate": 0.022781241666560845
  }
}
```

## Counterexamples

- High gen (>=40) with tip < gen at sight: **0** games (sample ids: none).
- Zero gather-wave wins: **0** (sample: none).
- Half-heavy (>=25% of actionable): **0** games.

## Loss skim (army mismanagement signals)

| match | opp | ticks | max_gen | tip@sight | tip@near_kill | waves | signals |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 20583 | Non-Linear Slob | 193 | 33 | None | 47 | 3 | never_saw_enemy_general |
| 20595 | Nicholas | 428 | 56 | 41 | 55 | 9 | — |
| 20603 | __ | 202 | 29 | None | 59 | 3 | never_saw_enemy_general |
| 20923 | Jonas Dujava | 133 | 20 | 7 | 35 | 2 | tiny_tip_at_sight |
| 24184 | bist | 620 | 146 | None | 218 | 0 | zero_gather_waves, never_saw_enemy_general, half_move_heavy |
| 24185 | bist | 536 | 49 | None | 43 | 0 | zero_gather_waves, never_saw_enemy_general |
| 24186 | bist | 171 | 27 | None | 60 | 0 | zero_gather_waves, never_saw_enemy_general |
| 24187 | bist | 242 | 47 | None | 66 | 0 | zero_gather_waves, never_saw_enemy_general, half_move_heavy |
| 24188 | bist | 367 | 49 | None | 62 | 0 | zero_gather_waves, never_saw_enemy_general, half_move_heavy |
| 24189 | bist | 345 | 48 | None | 80 | 0 | zero_gather_waves, never_saw_enemy_general, half_move_heavy |
| 24988 | erik.bystron | 89 | 15 | None | 25 | 0 | zero_gather_waves, never_saw_enemy_general |

## Unknowns

- **U1_multi_source_ticks** [UNKNOWN]: Move inference marks many ticks as multi/unknown (simultaneous sources or combat). Half/full rates on those ticks are unreliable.
- **U2_castle_ownership_timing** [UNKNOWN]: Castle army fraction uses all confirmed castle cells; enemy-captured or late-built castles blur the owned-castle bank estimate.
- **U3_gather_wave_definition** [UNKNOWN]: gather_wave requires the largest stack to grow while moving. Secondary gathering into a non-max tip is invisible.
- **U4_routing_policy** [UNKNOWN]: This report measures mass location and send mode, not path choice (toward/away is path analysis, not army-routing thresholds).

## Paths

- Script: `scripts/grok-analyze/analyze_kubic_army.py`
- JSON: `docs/research/measurements/grok-kubic-army.json`
- Markdown: `docs/research/measurements/grok-kubic-army.md`
- Corpus split: `docs/research/measurements/grok-kubic-corpus-split.json`
- Helpers: `scripts/grok-analyze/kubic_corpus.py`, `scripts/grok-analyze/kubic_moves.py`
- Replay APIs: `arena/instrument/replay/` (metrics.max_stack/general_army, path.StackStep, events.gather_wave)
