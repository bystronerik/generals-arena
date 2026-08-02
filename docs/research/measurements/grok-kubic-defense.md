# Kubic defense / reaction (analyst #5)

Corpus: fit wins for rules; all seat-resolved losses for failure modes. Observational leaderboard replays only.

## Corpus

- Player: `Kubic`
- Fit wins analysed: **341**
- Losses analysed: **11**
- Split: `wins sorted by match_id; index i with i%10==9 -> holdout; else fit. Losses/draws are held out of fit for rule derivation; skimmed separately for failure modes.`
- Definitions: recall = `stack_home_dist` drops by ≥1 within 30 ticks; already_home if dist ≤2; enemy_home_dist = `them.nearest_tile_dist`.

## Top rules

### `D1_proximity_asymmetry` — **MEASURED**

Enemy rarely reaches Manhattan <=3 of Kubic's general in fit wins; losses almost always do.

### `D2_recall_distance` — **MEASURED**

Fit wins that admit enemy_home_dist<=3 recall away stacks at 83% (25/30), median latency 5 ticks; at <=5 the rate is 91% (62/68). Losses at <=3 recall only 40% (4/10). Exact trigger D remains under-sampled in wins (only 55/341 reach <=3).

### `D3_incursion_redirect` — **MEASURED**

On sampled tile-loss episodes in fit wins, away max-stacks redirect home in 92% of cases; median latency 5.0 ticks; median stack_home_dist at threat 4.0.

### `D4_abandon_under_pressure` — **MEASURED**

After first enemy_home_dist<=3 in fit wins: median pre/post tile gains 7.0/9.0; median frac post moves toward enemy general 0.0; toward home 0.2222222222222222.

### `D5_home_reserve` — **MEASURED**

Fit wins: peak general_army median 27, at closest enemy approach median 8 (closest usually far: min_enemy_dist median 7). Losses: peak median 47, at closest (always dist=1) median 15 — bank is present but stack_home_dist at closest is median 12 vs fit 5 (stack not home). Failure is late recall / stack position, not an empty general cell.

### `D6_loss_failure_modes` — **MEASURED**

Primary defense failures appear in losses: enemy reaches <=1 in 100% of losses vs 4% of fit wins. See all_losses counterexamples.

### `D7_vision_gated_recall` — **INFERRED**

Whether recall requires vision of the threatening tile is only partially measured (visible_threat on incursions / prox events). Causal vision→recall link remains uncertain without action logs.

### `D8_exact_recall_threshold` — **UNKNOWN**

Exact integer recall radius (policy threshold) is UNKNOWN: fit wins rarely admit enemy near home, so the trigger D is under-sampled; losses confound failed defense with late proximity.

## Proximity rates (enemy to Kubic general)

| D | fit wins rate | losses rate | tag |
| --- | --- | --- | --- |
| ≤1 | 0.038 (13/341) | 1.000 (11/11) | MEASURED |
| ≤2 | 0.091 (31/341) | 1.000 (11/11) | MEASURED |
| ≤3 | 0.161 (55/341) | 1.000 (11/11) | MEASURED |

Min enemy_home_dist (pre-capture): fit median=7.0, p10=3.0; losses median=1.0, p10=1.0.

## Recall by proximity threshold (fit)

| D | n reached | recall frac (away) | median latency | median stack_home | median gen_army |
| --- | --- | --- | --- | --- | --- |
| ≤1 | 13 | 0.6666666666666666 (6/9) | 5.0 | 8.0 | 12.0 |
| ≤2 | 31 | 0.8095238095238095 (17/21) | 4.0 | 8.0 | 9.0 |
| ≤3 | 55 | 0.8333333333333334 (25/30) | 5.0 | 3.0 | 8.0 |
| ≤4 | 80 | 0.8372093023255814 (36/43) | 5.5 | 4.0 | 7.0 |
| ≤5 | 112 | 0.9117647058823529 (62/68) | 5.0 | 5.0 | 7.0 |
| ≤6 | 133 | 0.9080459770114943 (79/87) | 4.0 | 5.0 | 6.0 |
| ≤8 | 201 | 0.9562043795620438 (131/137) | 5.0 | 5.0 | 5.0 |

## Recall by proximity threshold (losses)

| D | n reached | recall frac (away) | median latency | median stack_home | median gen_army |
| --- | --- | --- | --- | --- | --- |
| ≤1 | 11 | 0.2 (2/10) | 4.0 | 12.0 | 15.0 |
| ≤2 | 11 | 0.4 (4/10) | 2.0 | 10.0 | 9.0 |
| ≤3 | 11 | 0.4 (4/10) | 6.5 | 8.0 | 8.0 |
| ≤4 | 11 | 0.4444444444444444 (4/9) | 4.5 | 8.0 | 8.0 |
| ≤5 | 11 | 0.5555555555555556 (5/9) | 13.0 | 7.0 | 7.0 |
| ≤6 | 11 | 0.5555555555555556 (5/9) | 3.0 | 5.0 | 5.0 |
| ≤8 | 11 | 0.5555555555555556 (5/9) | 5.0 | 5.0 | 9.0 |

## Incursion redirect

### fit wins

- Samples: 1427; visible_threat rate=0.3644008409250175; already_home=0.416958654519972; recall among away=0.921875 (767/832).
- Latency ticks: median=5.0, p25=1.0, p75=10.0.
- Stack home dist at threat: median=4.0, p90=15.0.

### losses

- Samples: 58; visible_threat rate=0.7068965517241379; already_home=0.25862068965517243; recall among away=0.37209302325581395 (16/43).
- Latency ticks: median=10.5, p25=3.75, p75=19.25.
- Stack home dist at threat: median=4.0, p90=12.399999999999999.

## Abandoned under pressure (fit, first reach ≤3 or ≤5)

### First enemy_home_dist ≤3 (fit)

- Tile gains window±20: pre median=7.0, post median=9.0.
- Castles in window: pre median=0.0, post median=0.0.
- Frac post moves toward enemy general: median=0.0; toward home: median=0.2222222222222222.

### First enemy_home_dist ≤5 (fit)

- Tile gains window±20: pre median=7.5, post median=11.0.
- Castles in window: pre median=0.0, post median=0.0.
- Frac post moves toward enemy general: median=0.0; toward home: median=0.125.

## Home army reserve

- Fit peak general_army: median=27.0, p10=18.0.
- Fit general_army at closest enemy: median=8.0.
- Loss peak general_army: median=47.0.
- Loss general_army at closest enemy: median=15.0.
- Stack home dist at closest enemy: fit median=5.0; losses median=12.0 (MEASURED failure mode: army exists on general, max stack is elsewhere).

## Tile-loss streaks

- Fit games with streak: 0.4046920821114369 (138/341).
- Loss games with streak: 0.2727272727272727 (3/11).

## Losses (failure-mode skim)

| match | opponent | ticks | min_d | ≤3/≤2/≤1 | prox3 recall | latency | stack_home | gen@threat | gen@closest |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20583 | Non-Linear Slob | 192 | 1 | 1/1/1 | True (home=False) | 4 | 8 | 7 | 13 |
| 20595 | Nicholas | 427 | 1 | 1/1/1 | True (home=False) | 8 | 3 | 2 | 29 |
| 20603 | __ | 201 | 1 | 1/1/1 | False (home=False) | None | 18 | 14 | 16 |
| 20923 | Jonas Dujava | 132 | 1 | 1/1/1 | False (home=False) | None | 14 | 8 | 9 |
| 24184 | bist | 619 | 1 | 1/1/1 | False (home=True) | None | 1 | 5 | 33 |
| 24185 | bist | 535 | 1 | 1/1/1 | True (home=False) | 5 | 4 | 15 | 43 |
| 24186 | bist | 170 | 1 | 1/1/1 | False (home=False) | None | 8 | 20 | 3 |
| 24187 | bist | 241 | 1 | 1/1/1 | False (home=False) | None | 3 | 12 | 43 |
| 24188 | bist | 366 | 1 | 1/1/1 | True (home=False) | 27 | 3 | 2 | 15 |
| 24189 | bist | 344 | 1 | 1/1/1 | False (home=False) | None | 15 | 25 | 7 |
| 24988 | erik.bystron | 88 | 1 | 1/1/1 | False (home=False) | None | 9 | 8 | 9 |

## Counterexamples

### Fit wins with enemy ≤2 of home

- `20527` min_dist=1 tick=186 gen=14 stack_home=2
- `20592` min_dist=1 tick=464 gen=21 stack_home=15
- `20594` min_dist=1 tick=79 gen=9 stack_home=0
- `20602` min_dist=1 tick=138 gen=12 stack_home=5
- `20606` min_dist=1 tick=142 gen=16 stack_home=0
- `21502` min_dist=1 tick=198 gen=36 stack_home=0
- `21907` min_dist=1 tick=169 gen=9 stack_home=8
- `21909` min_dist=1 tick=175 gen=8 stack_home=12
- `21976` min_dist=1 tick=140 gen=2 stack_home=21
- `22361` min_dist=1 tick=196 gen=8 stack_home=12
- `22362` min_dist=1 tick=291 gen=3 stack_home=13
- `22364` min_dist=1 tick=248 gen=30 stack_home=8
- `24813` min_dist=1 tick=174 gen=12 stack_home=14
- `20581` min_dist=2 tick=79 gen=9 stack_home=15
- `20585` min_dist=2 tick=181 gen=8 stack_home=15

### Fit wins with no recall at prox ≤3 (stack away)

- `20584` D≤3 tick=339 stack_home=18
- `21909` D≤3 tick=173 stack_home=10
- `21990` D≤2 tick=222 stack_home=19
- `21993` D≤3 tick=351 stack_home=26
- `22362` D≤1 tick=291 stack_home=13
- `24813` D≤3 tick=172 stack_home=12
- `25166` D≤3 tick=178 stack_home=21

## Unknowns

- Exact policy recall radius (integer D) — **UNKNOWN** (under-sampled in wins).
- Whether recall is vision-gated vs omniscient proximity — **INFERRED** only.
- Half-moves / leave-1 vs full pulls on defense — not decoded here (see `kubic_moves.py` if needed).
- Intentional abandon of castle builds vs coincidence of phase — **INFERRED** from pre/post counts only.

## Paths

- Script: `scripts/analyze_kubic_defense.py`
- JSON: `docs/research/measurements/grok-kubic-defense.json`
- This report: `docs/research/measurements/grok-kubic-defense.md`
- Corpus split: `docs/research/measurements/grok-kubic-corpus-split.json`

