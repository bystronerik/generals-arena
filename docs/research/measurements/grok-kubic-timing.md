# Kubic timing and tempo (fit wins)

Analyst #6. Fit wins only for rules (n=341). Loss skim n=11. Script: `scripts/analyze_kubic_timing.py`.

## Top rules

- **[MEASURED]** `T1_near_continuous_action`: Fit wins act almost every tick after the opening: global pass_rate=0.0116 (911/78749); per-game median=0.0110, p90=0.0203.
  - Threshold: pass_rate < 0.03 per game after accounting for ticks 0-1
- **[MEASURED]** `T1b_opening_forced_passes`: Every fit win passes tick 0 and tick 1 (rates 1.00/1.00). Tick 3 is also often a pass (0.45 of games). After tick 5, passes are rare.
  - Threshold: no move on ticks 0-1; optional pass on tick 3 while army grows
- **[MEASURED]** `T2_full_dominates_half`: Full (leave-1) moves dominate half-moves: global full=0.661, half=0.010. Per-game half median=0.0083.
  - Threshold: use full moves by default; half_rate < 0.05
- **[MEASURED]** `T3_action_density_by_bucket`: Pass rate by turn bucket (pooled): 1-50 pass=0.0315 active=0.9685 mean_tiles_gained/tick=0.46240469208211143; 51-100 pass=0.0003 active=0.9997 mean_tiles_gained/tick=0.6225925954789897; 101-200 pass=0.0007 active=0.9993 mean_tiles_gained/tick=0.6782195125389393; 201+ pass=0.0213 active=0.9787 mean_tiles_gained/tick=3.0677459420784747. Mid-game (51-200) is essentially zero-pass.
- **[MEASURED]** `T4_immediate_reaction`: Reaction to first visible enemy tile is immediate: median latency=0.0 ticks, p75=1.0, p90=3.0, max=18.0 (n=341). Kinds: {'toward_visible_enemy': 160, 'inferred_capture_move': 165, 'first_capture_event': 16}. First visible enemy median tick=82.0.
  - Threshold: react within 3 ticks of first enemy vision (p90); never >20 on fit wins
- **[MEASURED]** `T5_no_wait_for_bulk_growth`: After excluding ticks 0-1: odd pass=0.0047, even pass=0.0012. Before bulk (t%50==49) pass=0.0029 is LOWER than other odd ticks (0.0048). No wait-for-bulk-growth pattern.
  - Threshold: do NOT idle before tick%50==0; keep acting through growth boundaries
- **[MEASURED]** `T6_milestone_timing`: Median first_contact=82.0 (38.33% of game length); first_general_sight=180.0 (88.65%); game_length=202.0. Sight→kill median=24.0 ticks (p25=2.0, p75=59.0). Contact→sight median gap=93.0 ticks.
  - Threshold: contact ~tick 70-90; sight often late (~0.89 of game); convert sight to kill in ~24 ticks median
- **[MEASURED]** `T7_first_castle_tick_10`: First castle_built production stamp: median=10.0; exactly tick 10 in 168/256 castle games; ≤12 in 181. Overall castle events n=385 in 256/341 games. mod50 mass at residue 10: 184.
  - Threshold: if building a castle, first production often appears at tick 10
- **[INFERRED]** `T9_loss_tempo_collapse`: Loss skim n=11: median pass_rate=0.370 vs fit 0.011; never_sight=9/11. 6/11 losses have pass_rate≥0.30 (cluster match_ids 24184-24189) — tempo collapse / possible disconnect, not the fit-win policy.
  - Threshold: fit-like tempo has pass_rate≪0.05; ≥0.30 flags collapse
  - Counterexamples: `[{'match_id': '20583', 'ticks': 192, 'pass_rate': 0.0156, 'active_rate': 0.9844, 'first_contact': 74, 'first_sight': None, 'reaction_latency': 0, 'bucket_pass': {'1-50': 0.02, '51-100': 0.0, '101-200': 0.011, '201+': 1.0}}, {'match_id': '20595', 'ticks': 427, 'pass_rate': 0.0094, 'active_rate': 0.9906, 'first_contact': 31, 'first_sight': 350, 'reaction_latency': 0, 'bucket_pass': {'1-50': 0.02, '51-100': 0.0, '101-200': 0.0, '201+': 0.0132}}, {'match_id': '20603', 'ticks': 201, 'pass_rate': 0.0149, 'active_rate': 0.9851, 'first_contact': 87, 'first_sight': None, 'reaction_latency': 0, 'bucket_pass': {'1-50': 0.02, '51-100': 0.0, '101-200': 0.01, '201+': 1.0}}, {'match_id': '20923', 'ticks': 132, 'pass_rate': 0.0227, 'active_rate': 0.9773, 'first_contact': 34, 'first_sight': 73, 'reaction_latency': 1, 'bucket_pass': {'1-50': 0.02, '51-100': 0.0, '101-200': 0.0323, '201+': 1.0}}, {'match_id': '24988', 'ticks': 88, 'pass_rate': 0.0341, 'active_rate': 0.9659, 'first_contact': 82, 'first_sight': None, 'reaction_latency': 1, 'bucket_pass': {'1-50': 0.02, '51-100': 0.027, '201+': 1.0}}]`

## Move rates (global pool)

| kind | count | rate |
| --- | ---: | ---: |
| build_suspect | 24 | 0.0003 |
| full | 52045 | 0.6609 |
| half | 801 | 0.0102 |
| multi | 23163 | 0.2941 |
| pass | 911 | 0.0116 |
| unknown | 1805 | 0.0229 |

Per-game distributions:

- `pass_rate`: n=341 min=0.00343 p25=0.00858 med=0.011 p75=0.0155 p90=0.0203 max=0.146 mean=0.0132
- `active_rate`: n=341 min=0.854 p25=0.984 med=0.989 p75=0.991 p90=0.993 max=0.997 mean=0.987
- `full_rate`: n=341 min=0.447 p25=0.504 med=0.526 p75=0.877 p90=0.966 max=0.984 mean=0.671
- `half_rate`: n=341 min=0 p25=0.00495 med=0.00833 p75=0.016 p90=0.0266 max=0.0876 mean=0.0114
- `build_suspect_rate`: n=341 min=0 p25=0 med=0 p75=0 p90=0 max=0.0132 mean=0.000234
- `multi_rate`: n=341 min=0 p25=0.0833 med=0.38 p75=0.434 p90=0.456 max=0.5 mean=0.28

## Action density by turn bucket

| bucket | games | transitions | pass | active | full | half | mean tiles_gained/tick |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1-50 | 341 | 17050 | 0.0315 | 0.9685 | 0.6992 | 0.0388 | 0.4624 |
| 51-100 | 341 | 16957 | 0.0003 | 0.9997 | 0.6404 | 0.0034 | 0.6226 |
| 101-200 | 336 | 28352 | 0.0007 | 0.9993 | 0.6402 | 0.0019 | 0.6782 |
| 201+ | 341 | 16390 | 0.0213 | 0.9787 | 0.6780 | 0.0016 | 3.0677 |

## Reaction latency

First visible enemy tile → first capture or toward-move. n_reacted=341, never_saw=0, saw_no_react=0.

- latency_ticks: n=341 min=0 p25=0 med=0 p75=1 p90=3 max=18 mean=0.985
- first_visible_enemy_tick: n=341 min=30 p25=68 med=82 p75=90 p90=119 max=272 mean=82.7
- reaction_kind_counts: `{'toward_visible_enemy': 160, 'inferred_capture_move': 165, 'first_capture_event': 16}`
- slow (>30 ticks): n=0 rate=0.000

## Turn-modulo periodicity

Ticks 0-1 are excluded from modulo tables: every fit win passes both (general army starts at 1; leave-1 is impossible).

- Opening passes: tick0=1.00, tick1=1.00, tick3=0.45
- mod2 (tick≥2): odd pass=0.0047 (n=38954), even pass=0.0012 (n=39113)
- before bulk (t%50==49) pass=0.0029 (n=1367); other odds pass=0.0048
- mod50 highest pass residues: `[{'residue': 3, 'pass_rate': 0.09112426035502959, 'n': 1690}, {'residue': 5, 'pass_rate': 0.003569303985722784, 'n': 1681}, {'residue': 4, 'pass_rate': 0.0029691211401425177, 'n': 1684}, {'residue': 2, 'pass_rate': 0.0029533372711163615, 'n': 1693}, {'residue': 49, 'pass_rate': 0.002926115581565472, 'n': 1367}, {'residue': 6, 'pass_rate': 0.002386634844868735, 'n': 1676}, {'residue': 34, 'pass_rate': 0.0019828155981493722, 'n': 1513}, {'residue': 7, 'pass_rate': 0.0017953321364452424, 'n': 1671}]`

## Milestone timing

- `game_length`: n=341 min=78 p25=179 med=202 p75=278 p90=348 max=583 mean=231
- `first_contact_tick`: n=341 min=30 p25=69 med=82 p75=91 p90=124 max=273 mean=84.3
- `first_sight_tick`: n=341 min=73 p25=136 med=180 p75=235 p90=296 max=581 mean=191
- `kill_tick`: n=341 min=78 p25=179 med=202 p75=278 p90=348 max=583 mean=231
- `contact_frac_of_game`: n=341 min=0.0722 p25=0.266 med=0.383 p75=0.489 p90=0.664 max=0.996 mean=0.407
- `sight_frac_of_game`: n=341 min=0.272 p25=0.72 med=0.887 p75=0.993 p90=0.995 max=0.998 mean=0.836
- `sight_minus_contact`: n=341 min=-57 p25=51 med=93 p75=148 p90=228 max=500 mean=107
- `kill_minus_sight`: n=341 min=1 p25=2 med=24 p75=59 p90=101 max=305 mean=39.7
- `expansion_phase_len`: n=341 min=30 p25=69 med=82 p75=91 p90=124 max=273 mean=84.3
- `contest_phase_len`: n=341 min=2 p25=91 med=115 p75=192 p90=281 max=503 mean=146
- never_contact=0, never_sight=0
- Note: kill_tick ≈ game_length on wins; prefer kill_minus_sight.

## Castle build tick modulo

castle_built n=385 games_with=256; all-events tick dist: n=385 min=8 p25=10 med=118 p75=160 p90=174 max=320 mean=84.5
- first_castle_tick: n=256 min=8 p25=10 med=10 p75=118 p90=132 max=236 mean=44; eq10=168, le12=181
- mod2: `{0: 385}` — All castle_built stamps land on even ticks because detection keys off production (+1 on even ticks). This is a detector artifact, not evidence that the spend action is even-only.
- mod50 top: `[(10, 184), (22, 32), (20, 30), (12, 23), (16, 20), (26, 16), (24, 15), (18, 14), (14, 13), (28, 12)]`
- build_suspect n=24; tick dist: n=24 min=122 p25=150 med=193 p75=241 p90=290 max=349 mean=200; mod2=`{0: 19, 1: 5}`

## Loss skim (tempo collapse)

n=11. pass_rate n=11 min=0.00937 p25=0.0192 med=0.37 p75=0.424 p90=0.498 max=0.706 mean=0.265; length n=11 min=88 p25=181 med=241 p75=396 p90=535 max=619 mean=301; never_sight=9.

| match | ticks | pass | contact | sight | react |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20583 | 192 | 0.0156 | 74 | None | 0 |
| 20595 | 427 | 0.0094 | 31 | 350 | 0 |
| 20603 | 201 | 0.0149 | 87 | None | 0 |
| 20923 | 132 | 0.0227 | 34 | 73 | 1 |
| 24184 | 619 | 0.37 | 206 | None | 8 |
| 24185 | 535 | 0.3944 | 209 | None | 4 |
| 24186 | 170 | 0.7059 | 123 | None | 13 |
| 24187 | 241 | 0.4979 | 145 | None | 5 |
| 24188 | 366 | 0.4262 | 232 | None | 5 |
| 24189 | 344 | 0.4215 | 157 | None | 5 |
| 24988 | 88 | 0.0341 | 82 | None | 1 |

## Claims (all tags)

### [MEASURED] `T1_near_continuous_action`

Fit wins act almost every tick after the opening: global pass_rate=0.0116 (911/78749); per-game median=0.0110, p90=0.0203.

### [MEASURED] `T1b_opening_forced_passes`

Every fit win passes tick 0 and tick 1 (rates 1.00/1.00). Tick 3 is also often a pass (0.45 of games). After tick 5, passes are rare.

### [MEASURED] `T2_full_dominates_half`

Full (leave-1) moves dominate half-moves: global full=0.661, half=0.010. Per-game half median=0.0083.

### [MEASURED] `T3_action_density_by_bucket`

Pass rate by turn bucket (pooled): 1-50 pass=0.0315 active=0.9685 mean_tiles_gained/tick=0.46240469208211143; 51-100 pass=0.0003 active=0.9997 mean_tiles_gained/tick=0.6225925954789897; 101-200 pass=0.0007 active=0.9993 mean_tiles_gained/tick=0.6782195125389393; 201+ pass=0.0213 active=0.9787 mean_tiles_gained/tick=3.0677459420784747. Mid-game (51-200) is essentially zero-pass.

### [MEASURED] `T4_immediate_reaction`

Reaction to first visible enemy tile is immediate: median latency=0.0 ticks, p75=1.0, p90=3.0, max=18.0 (n=341). Kinds: {'toward_visible_enemy': 160, 'inferred_capture_move': 165, 'first_capture_event': 16}. First visible enemy median tick=82.0.

### [MEASURED] `T5_no_wait_for_bulk_growth`

After excluding ticks 0-1: odd pass=0.0047, even pass=0.0012. Before bulk (t%50==49) pass=0.0029 is LOWER than other odd ticks (0.0048). No wait-for-bulk-growth pattern.

### [MEASURED] `T6_milestone_timing`

Median first_contact=82.0 (38.33% of game length); first_general_sight=180.0 (88.65%); game_length=202.0. Sight→kill median=24.0 ticks (p25=2.0, p75=59.0). Contact→sight median gap=93.0 ticks.

Note: kill_tick ≈ game_length on wins (game ends on general capture), so kill_frac≈1 is tautological; use kill_minus_sight.

### [MEASURED] `T7_first_castle_tick_10`

First castle_built production stamp: median=10.0; exactly tick 10 in 168/256 castle games; ≤12 in 181. Overall castle events n=385 in 256/341 games. mod50 mass at residue 10: 184.

Note: All castle_built stamps land on even ticks because detection keys off production (+1 on even ticks). This is a detector artifact, not evidence that the spend action is even-only.

### [UNKNOWN] `T8_multi_ambiguity`

Global multi rate is 0.294. Cannot determine true multi-source simultaneous moves versus inference collisions when several sources drop army.

### [INFERRED] `T9_loss_tempo_collapse`

Loss skim n=11: median pass_rate=0.370 vs fit 0.011; never_sight=9/11. 6/11 losses have pass_rate≥0.30 (cluster match_ids 24184-24189) — tempo collapse / possible disconnect, not the fit-win policy.

## Cannot determine

- Whether tick-3 passes are intentional waits for army=2 or pathfinding startup.
- True simultaneous multi-cell orders vs single-move inference collisions (kind=multi).
- Exact castle spend tick (castle_built stamps first production; always even).
- Whether reaction targets any visible enemy tile or a specific threat threshold.
- Internal turn-budget / search-time policy beyond observable action density.
- Cause of high-pass loss cluster 24184-24189 (policy change vs disconnect).
- Holdout verification (this script derives on fit only).

## Paths

- Script: `scripts/analyze_kubic_timing.py`
- JSON: `docs/research/measurements/grok-kubic-timing.json`
- Corpus: `scripts/kubic_corpus.py`, split `grok-kubic-corpus-split.json`
- Moves: `scripts/kubic_moves.py`

