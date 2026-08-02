# Kubic opening (first 50 turns) — fit wins

**Corpus:** fit wins only, n=341. Holdout excluded. Losses skimmed for failure modes only (not used to derive rules). Split: `docs/research/measurements/grok-kubic-corpus-split.json`.

**Raw JSON:** `docs/research/measurements/grok-kubic-opening.json`. Script: `scripts/analyze_kubic_opening.py`.

Every claim is tagged MEASURED, INFERRED, or UNKNOWN. Folder labels are not outcomes; seats use `Replay.outcome`.

## Key distributions

- MEASURED (n=341): tiles at tick 50 — n=341 median=24.0 [p10=23.0, p90=25.0] min=20.0 max=25.0.
- MEASURED (n=341): first tile-gain tick — n=341 median=3.0 [p10=3.0, p90=3.0] min=3.0 max=25.0; counts={'3': 336, '7': 2, '25': 1, '11': 1, '9': 1}.
- MEASURED: gen army before first gain — {'2': 336, '4': 2, '13': 1, '6': 1, '5': 1}.
- MEASURED: tiles trajectory medians — t5=2.0, t10=4.0, t15=4.0, t20=4.0, t25=4.0, t30=7.0, t35=12.0, t40=15.0, t45=20.0, t50=24.0.
- MEASURED: p(tiles_gained>0) at ticks 3/6/9/20/27/30/37 = 0.985/0.935/0.739/0.003/0.103/0.982/0.085.
- MEASURED: phase-C flood start tick — n=341 median=27.0 [p10=26.0, p90=28.0] min=20.0 max=31.0; rate within 27±1 = 0.944.
- MEASURED: at flood start — stack n=341 median=10.0 [p10=9.0, p90=11.0] min=5.0 max=13.0; army n=341 median=14.0 [p10=14.0, p90=15.0] min=11.0 max=16.0; tiles n=341 median=4.0 [p10=3.0, p90=5.0] min=1.0 max=6.0.
- MEASURED: first-move toward map center rate=0.845; toward true enemy general rate=0.809 (fog: bot cannot see enemy general).
- MEASURED: min-dist-to-center neighbour rule hit rate=0.909; min-dist-to-enemy-gen rule=0.859.
- MEASURED: early stack median dist to general=1.0; frac ticks with stack on general=0.44.
- MEASURED: first-50 captures neutral=7874 enemy=10 (enemy_rate=0.0013); owned-enemy-half frac at 50 median=0.0.
- MEASURED: moves from general on even tick transitions rate=0.773 (even=3427, odd=1009).
- MEASURED: real castle builds in first 50 — {'0': 341} (0 in all 341 games). First real build tick — n=281 median=138.0 [p10=120.0, p90=202.0] min=116.0 max=321.0; dist_to_general — n=281 median=9.0 [p10=7.0, p90=20.0] min=2.0 max=31.0; army_before — n=281 median=37.0 [p10=33.0, p90=62.0] min=30.0 max=174.0; drop — n=281 median=35.0 [p10=32.0, p90=60.0] min=30.0 max=173.0.
- MEASURED: EventLog `castle_built` with tick<=50 and no spend evidence — 190 false positives. Do not use early event ticks as build times.

## Candidate decision rules

### R1_first_expand_tick3 [MEASURED]

After even-tick production raises general army to >=2, on the transition into tick 3 issue a full leave-1 move from the general onto an open orthogonal neighbour.

Thresholds: `{'first_gain_tick': 3, 'gen_army_before': 2, 'units': 'ticks / army'}`

Support: `{'p_gain_tick3': 0.9853372434017595, 'gen_army_2_rate': 0.9853372434017595, 'n': 341}`

Counterexamples (n=5): 20560, 21972, 22909, 24990, 25166

### R2_pulse_3_6_9 [MEASURED]

Repeat a capture pulse at ticks 6 and 9 (every +3), then stop net tile growth until ~tick 27.

Thresholds: `{'pulse_ticks': [3, 6, 9], 'pause_until': 27, 'units': 'ticks'}`

Support: `{'p_gain_3': 0.9853372434017595, 'p_gain_6': 0.9354838709677419, 'p_gain_9': 0.7390029325513197, 'p_gain_20': 0.002932551319648094, 'n': 341}`

### R3_first_step_toward_center [INFERRED]

Among open orthogonal neighbours of the general, pick a cell that minimizes Manhattan distance to map center ((rows-1)/2,(cols-1)/2). Tie-break: unknown (dirs W/E/N/S all appear).

Thresholds: `{'center_rule_hit_rate': 0.9090909090909091, 'egen_rule_hit_rate': 0.8592375366568915, 'units': 'rate over fit wins'}`

Support: `{'center_hits': 310, 'egen_hits': 293, 'neither': 22, 'n': 341}`

Reasoning: Center rule 90.9% > true-enemy-gen rule 85.9%. Center is fog-legal. Enemy-gen alignment is likely a side effect of opposite spawns.

Counterexamples (n=22): 20551, 20552, 20569, 20602, 20898, 20922, 20993, 21061, 21062, 21084, 21248, 21536, 21548, 21549, 21775, 21859, 21894, 21931, 22020, 22220, 23768, 24809

### R4_no_castle_before_50 [MEASURED]

Do not build castles in the first 50 ticks. Prefer neutral land. First real build (spend detector) occurs at median tick 138, typically dist_to_general in 6..10 with army_before near price.

Thresholds: `{'castle_builds_in_first50': 0, 'first_build_tick_median': 138.0, 'units': 'ticks / army on cell'}`

Support: `{'games_with_0_real_builds_in_50': 341, 'n': 341, 'event_false_early': 190}`

### R5_flood_at_tick_27 [MEASURED]

At tick ~27 (median 27; 94.4% within 27±1), begin nearly every-tick neutral expansion until tick 50. Precondition observed: total army median 14, max_stack median 10, tiles still ~4.

Thresholds: `{'flood_start_tick': 27, 'tolerance': 1, 'army_at_start_median': 14.0, 'stack_at_start_median': 10.0, 'units': 'ticks / army'}`

Support: `{'start_near_27_pm1_rate': 0.9442815249266863, 'n': 341}`

### R6_gen_moves_on_even_ticks [MEASURED]

Moves that leave the general prefer even tick transitions (production ticks). Non-general moves prefer odd transitions.

Thresholds: `{'gen_move_even_rate': 0.7725428313796213, 'units': 'fraction of inferred moves in ticks 0..49'}`

Support: `{'from_gen': {0: 3427, 1: 1009}, 'not_from_gen': {1: 6801, 0: 4732}}`

## Counterexamples (match_ids)

- No gain at tick 3 (n=5): 20560, 21972, 22909, 24990, 25166
- Phase C missing or after tick 35 (n=0): 
- First move neither min-center nor min-egen (n=22): 20551, 20552, 20569, 20602, 20898, 20922, 20993, 21061, 21062, 21084, 21248, 21536, 21548, 21549, 21775, 21859, 21894, 21931, 22020, 22220, 23768, 24809
- First move away from true enemy general (n=65): 20527, 20540, 20541, 20551, 20552, 20560, 20564, 20567, 20569, 20602, 20898, 20921, 20922, 20993, 20996, 21058, 21061, 21062, 21084, 21086, 21215, 21221, 21248, 21479, 21502, 21507, 21534, 21536, 21547, 21548, 21549, 21699, 21763, 21765, 21774, 21775, 21777, 21780, 21781, 21807 ...

## Loss skim (excluded from derivation)

Losses 24184-24189 break the 3/6/9 pulse and end first-50 with tiles50 in 4..7 — possible version skew or extreme maps; not used to derive opening rules.

MEASURED: 4/11 losses still show gains at ticks 3,6,9.

- `20583` ticks=192 tiles50=25 first_gain=3 contact=74 3/6/9=1/1/1
- `20595` ticks=427 tiles50=21 first_gain=3 contact=31 3/6/9=1/1/1
- `20603` ticks=201 tiles50=24 first_gain=3 contact=87 3/6/9=1/1/1
- `20923` ticks=132 tiles50=22 first_gain=3 contact=34 3/6/9=1/1/1
- `24184` ticks=619 tiles50=7 first_gain=6 contact=206 3/6/9=0/1/0
- `24185` ticks=535 tiles50=4 first_gain=14 contact=209 3/6/9=0/0/0
- `24186` ticks=170 tiles50=7 first_gain=7 contact=123 3/6/9=0/0/0
- `24187` ticks=241 tiles50=4 first_gain=36 contact=145 3/6/9=0/0/0
- `24188` ticks=366 tiles50=6 first_gain=12 contact=232 3/6/9=0/0/0
- `24189` ticks=344 tiles50=6 first_gain=8 contact=157 3/6/9=0/0/0
- `24988` ticks=88 tiles50=24 first_gain=3 contact=82 3/6/9=1/1/0

## Could not determine

- UNKNOWN: Exact tie-break among equal min-center neighbours (W/E/N/S all common).
- UNKNOWN: Whether flood start is clock(tick==27) vs army/stack threshold (both co-occur).
- UNKNOWN: Exact mid-phase (10-25) micro-policy: many moves away from general without tile gains — gather vs patrol vs pathing noise not resolved.
- UNKNOWN: Half-move policy (512 half moves in ticks 10-25, src_before median 3).
- UNKNOWN: True action log (builds/moves inferred from frames only).
- UNKNOWN: Whether first-step rule uses map center, enemy-half heuristic, or openness score — center fits best among tested fog-legal rules but 22 neither cases remain.
- UNKNOWN: Bot version identity across the scrape window (loss cluster 24184-24189 opens differently).

## Method notes

- Competition rules (`RULES.md`): no pre-placed castles; build cost base 35 + crowding surcharge; production every other turn; bulk +1 every 50.
- Move inference: `scripts/kubic_moves.py` (frame diffs). Ambiguous ticks exist.
- Real castle detector: army drop >=30 with no neighbour receive; EventLog `castle_built` alone is unsafe before ~tick 100.
- Spawn classes: corner = within 1 of both edges; edge = min edge dist <=2; else centerish.

