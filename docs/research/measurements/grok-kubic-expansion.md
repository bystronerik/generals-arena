# Kubic expansion / land grab (fit wins)

Corpus: player `Kubic`, set=`fit`, n=341 wins analyzed (meta fit=341, holdout=37). Played outcomes (seat-resolved): {'win': 378, 'lose': 11, 'draw': 1}; forfeits=6. Losses skimmed n=11 and **excluded** from rule thresholds.

Tag legend: **MEASURED** = direct corpus statistic; **INFERRED** = candidate decision rule from the measured pattern; **UNKNOWN** = cannot determine from replays.

## Top candidate rules

1. **INFERRED** Expand neutrals until orthogonal enemy contact. **MEASURED** contact tick median=82.0 (p25=69.0, p75=91.0). Phase `expansion` ends at contact−1 by definition.
2. **INFERRED** Land operating curve (tiles): t25≈4.0, t50≈24.0, t100≈52.0, contact≈43.0 (p25=33.0, p75=52.0). **MEASURED**.
3. **INFERRED** Keep a single contiguous blob. **MEASURED** components at expansion end median=1.0 (max=1 on fit); largest-component share=1.0; frontier width median=25.0.
4. **INFERRED** Multi-option neutral capture bias: toward enemy general (best-option rate=0.767) and/or local fill (best-option rate=0.775). **MEASURED** direction counts: closes_egen=3540, opens_egen=1497, opens_ogen=4569, closes_ogen=468 (opens from own general dominates).
5. **INFERRED** During expansion, convey on own tiles more than capture: neutral/(neutral+own) inferred-move share median≈0.2222222222222222. Largest-stack path is almost always blind to enemy targets (blind_share median≈1.0). **MEASURED**.
6. **INFERRED** After contact, prefer continued neutral grab over a hard pivot. **MEASURED** 50-tick post-contact: neutral-share median=0.6296296296296297, enemy-share median=0.37037037037037035; games with neutral share≥0.6 rate=0.5659824046920822; enemy share≥0.6 rate=0.2316715542521994.

## Distributions (fit wins)

| Metric | Distribution |
| --- | --- |
| first_contact tick | `n=341 min=30 p10=40 p25=69 med=82 p75=91 p90=124 max=273 mean=84.3` |
| expansion length (ticks) | `n=341 min=29 p10=39 p25=68 med=81 p75=90 p90=123 max=272 mean=83.3` |
| expansion path toward_fraction (visible targets) | `n=115 min=0 p10=0.333 p25=0.913 med=1 p75=1 p90=1 max=1 mean=0.85` |
| expansion path blind_share | `n=341 min=0.474 p10=0.958 p25=0.975 med=1 p75=1 p90=1 max=1 mean=0.977` |
| expansion directed moves (toward+away) | `n=341 min=0 p10=0 p25=0 med=0 p75=1 p90=1 max=31 mean=0.853` |
| neutral/(neutral+own) inferred moves in expansion | `n=341 min=0.0526 p10=0.119 p25=0.159 med=0.222 p75=0.295 p90=0.353 max=0.463 mean=0.232` |
| largest-stack move landings onto former-neutral | `n=341 min=0.143 p10=0.333 p25=0.391 med=0.474 p75=0.537 p90=0.588 max=0.769 mean=0.463` |
| per-game mean stray Manhattan (neutral captures; always 1 if orth) | `n=341 min=1 p10=1 p25=1 med=1 p75=1 p90=1 max=1 mean=1` |
| per-game max stray Manhattan | `n=341 min=1 p10=1 p25=1 med=1 p75=1 p90=1 max=1 mean=1` |
| all stray samples pooled | `n=6268 min=1 p10=1 p25=1 med=1 p75=1 p90=1 max=1 mean=1` |
| per-game mean capture dist to own general | `n=341 min=1 p10=4.33 p25=5.43 med=6.4 p75=7.63 p90=9.51 max=17.7 mean=6.78` |
| per-game mean capture dist to enemy general | `n=341 min=5.25 p10=10.8 p25=13.4 med=16 p75=19.6 p90=23.1 max=32.6 mean=16.6` |
| post-contact 50t neutral share of gains | `n=341 min=0 p10=0.205 p25=0.444 med=0.63 p75=0.769 p90=0.893 max=1 mean=0.59` |
| post-contact 50t enemy share of gains | `n=341 min=0 p10=0.107 p25=0.231 med=0.37 p75=0.556 p90=0.795 max=1 mean=0.41` |
| post-contact 50t inferred neutral captures | `n=341 min=0 p10=1 p25=2 med=4 p75=7 p90=10 max=19 mean=4.84` |
| post-contact 50t inferred enemy captures | `n=341 min=0 p10=0 p25=1 med=3 p75=5 p90=8 max=19 mean=3.51` |

### Land growth

| Checkpoint | Distribution |
| --- | --- |
| tiles @ t=25 | `n=341 min=2 p10=3 p25=4 med=4 p75=5 p90=5 max=9 mean=4.37` |
| tiles @ t=50 | `n=341 min=20 p10=23 p25=23 med=24 p75=25 p90=25 max=25 mean=24` |
| tiles @ t=100 | `n=336 min=28 p10=42 p25=47 med=52 p75=57 p90=59.5 max=65 mean=51.3` |
| tiles @ first_contact | `n=341 min=7 p10=15 p25=33 med=43 p75=52 p90=60 max=127 mean=42.9` |
| tiles @ first_general_sight | `n=341 min=27 p10=44 p25=55 med=65 p75=76 p90=87 max=128 mean=66.2` |
| tiles @ expansion end | `n=341 min=6 p10=14 p25=32 med=42 p75=51 p90=60 max=126 mean=42` |
| army @ t=25 | `n=341 min=13 p10=13 p25=13 med=13 p75=13 p90=13 max=13 mean=13` |
| army @ t=50 | `n=341 min=41 p10=49 p25=49 med=50 p75=51 p90=51 max=51 mean=49.9` |
| army @ t=100 | `n=336 min=63 p10=88.5 p25=102 med=117 p75=128 p90=132 max=139 mean=113` |
| army @ first_contact | `n=341 min=16 p10=21 p25=60 med=66 p75=70 p90=139 max=494 mean=77.6` |
| army @ first_general_sight | `n=341 min=46 p10=66 p25=105 med=140 p75=195 p90=285 max=573 mean=165` |
| tile lead @ contact (us−them) | `n=341 min=-16 p10=-3 p25=1 med=6 p75=14 p90=31 max=120 mean=12.1` |
| army lead @ contact (us−them) | `n=341 min=-22 p10=0 p25=1 med=3 p75=11 p90=47 max=344 mean=16.4` |

### Contiguity / frontier

**t25**
- components: `n=341 min=1 p10=1 p25=1 med=1 p75=1 p90=1 max=1 mean=1`
- largest_component_share: `n=341 min=1 p10=1 p25=1 med=1 p75=1 p90=1 max=1 mean=1`
- frontier_width: `n=341 min=2 p10=3 p25=4 med=4 p75=5 p90=5 max=8 mean=4.14`

**t50**
- components: `n=341 min=1 p10=1 p25=1 med=1 p75=1 p90=1 max=2 mean=1.03`
- largest_component_share: `n=341 min=0.773 p10=1 p25=1 med=1 p75=1 p90=1 max=1 mean=0.998`
- frontier_width: `n=341 min=7 p10=14 p25=17 med=19 p75=21 p90=22 max=25 mean=18.6`

**t100**
- components: `n=336 min=1 p10=1 p25=1 med=1 p75=2 p90=2 max=5 mean=1.4`
- largest_component_share: `n=336 min=0.511 p10=0.912 p25=0.964 med=1 p75=1 p90=1 max=1 mean=0.971`
- frontier_width: `n=336 min=9 p10=21.5 p25=26 med=32 p75=37 p90=42 max=48 mean=31.7`

**exp_end**
- components: `n=341 min=1 p10=1 p25=1 med=1 p75=1 p90=1 max=1 mean=1`
- largest_component_share: `n=341 min=1 p10=1 p25=1 med=1 p75=1 p90=1 max=1 mean=1`
- frontier_width: `n=341 min=6 p10=12 p25=19 med=25 p75=31 p90=43 max=98 mean=27.2`

**contact**
- components: `n=341 min=1 p10=1 p25=1 med=1 p75=1 p90=1 max=1 mean=1`
- largest_component_share: `n=341 min=1 p10=1 p25=1 med=1 p75=1 p90=1 max=1 mean=1`
- frontier_width: `n=341 min=7 p10=13 p25=19 med=26 p75=32 p90=44 max=99 mean=28`

## Target selection (expansion)

- **MEASURED** Multi-option neutral captures: n=5037.
- **MEASURED** Chose a destination that is best (tied OK) toward enemy general: 3864 (rate=0.7671232876712328).
- **MEASURED** Chose a destination that is best (tied OK) for local fill (max own orthogonal neighbours): 3905 (rate=0.7752630534048044).
- **MEASURED** Direction counts vs enemy general among multi-option captures: `{'closes_egen': 3540, 'opens_ogen': 4569, 'opens_egen': 1497, 'closes_ogen': 468}`.
- **MEASURED** Note: `MEASURED among inferred orthogonal neutral captures with >=2 neutral neighbour options. Orth capture is always Manhattan-1 from src, so 'nearest neutral' is not discriminative.`
- **UNKNOWN** Exact scoring weights among toward-enemy-gen vs fill vs other heuristics (replays do not expose the policy).
- **MEASURED** `toward_fraction` on the expansion path uses *visible* enemy targets only; most expansion moves are blind (see blind_share).

## When expansion stops

- **MEASURED** Phase `expansion` ends at `first_contact - 1` (or game end if no contact).
- **INFERRED** Contact itself is the operational stop for pure neutral expansion; tile/army levels at that tick are the measured operating point (see land table).
- **UNKNOWN** Whether the bot has an internal tile/army threshold that ends expansion before contact (phase definition cannot reveal a pre-contact stop).

## Post-contact: expand elsewhere vs pivot

- **MEASURED** Games with tile gains in first 50 ticks after contact: n=341.
- **MEASURED** Enemy-share ≥ 0.6: n=79 (rate=0.2316715542521994).
- **MEASURED** Neutral-share ≥ 0.6: n=193 (rate=0.5659824046920822).
- **INFERRED** After contact Kubic often mixes continued neutral grab with contest; see distributions for the share mix (not a hard exclusive pivot).

## Counterexamples (match_ids)

- High components (≥3) at expansion end: `[]` (empty list = **MEASURED** no fit win had ≥3 components)
- Lowest tiles at contact: `[{'match_id': '20586', 'tiles': 7, 'contact': 30}, {'match_id': '20587', 'tiles': 7, 'contact': 30}, {'match_id': '20604', 'tiles': 8, 'contact': 31}, {'match_id': '20585', 'tiles': 9, 'contact': 32}, {'match_id': '21506', 'tiles': 9, 'contact': 32}, {'match_id': '21762', 'tiles': 9, 'contact': 32}, {'match_id': '24991', 'tiles': 9, 'contact': 32}, {'match_id': '20581', 'tiles': 10, 'contact': 33}]`
- Highest tiles at contact: `[{'match_id': '21548', 'tiles': 127, 'contact': 257}, {'match_id': '22218', 'tiles': 113, 'contact': 273}, {'match_id': '22219', 'tiles': 108, 'contact': 247}, {'match_id': '20532', 'tiles': 103, 'contact': 217}, {'match_id': '20535', 'tiles': 100, 'contact': 186}, {'match_id': '20570', 'tiles': 100, 'contact': 262}, {'match_id': '20997', 'tiles': 98, 'contact': 192}, {'match_id': '21505', 'tiles': 94, 'contact': 179}]`
- Earliest contact: `[{'match_id': '20586', 'contact': 30, 'tiles': 7}, {'match_id': '20587', 'contact': 30, 'tiles': 7}, {'match_id': '20604', 'contact': 31, 'tiles': 8}, {'match_id': '20585', 'contact': 32, 'tiles': 9}, {'match_id': '21506', 'contact': 32, 'tiles': 9}, {'match_id': '21762', 'contact': 32, 'tiles': 9}, {'match_id': '23079', 'contact': 32, 'tiles': 11}, {'match_id': '24991', 'contact': 32, 'tiles': 9}]`
- Latest contact: `[{'match_id': '22218', 'contact': 273, 'tiles': 113}, {'match_id': '20570', 'contact': 262, 'tiles': 100}, {'match_id': '21548', 'contact': 257, 'tiles': 127}, {'match_id': '22219', 'contact': 247, 'tiles': 108}, {'match_id': '20532', 'contact': 217, 'tiles': 103}, {'match_id': '21249', 'contact': 199, 'tiles': 85}, {'match_id': '23769', 'contact': 195, 'tiles': 92}, {'match_id': '20997', 'contact': 192, 'tiles': 98}]`
- Highest post-contact neutral share: `[{'match_id': '20534', 'neutral_share': 1.0, 'enemy_share': 0.0}, {'match_id': '20554', 'neutral_share': 1.0, 'enemy_share': 0.0}, {'match_id': '21059', 'neutral_share': 1.0, 'enemy_share': 0.0}, {'match_id': '21214', 'neutral_share': 1.0, 'enemy_share': 0.0}, {'match_id': '21244', 'neutral_share': 1.0, 'enemy_share': 0.0}, {'match_id': '21248', 'neutral_share': 1.0, 'enemy_share': 0.0}, {'match_id': '21731', 'neutral_share': 1.0, 'enemy_share': 0.0}, {'match_id': '21993', 'neutral_share': 1.0, 'enemy_share': 0.0}]`
- Highest post-contact enemy share: `[{'match_id': '20532', 'neutral_share': 0.0, 'enemy_share': 1.0}, {'match_id': '20567', 'neutral_share': 0.0, 'enemy_share': 1.0}, {'match_id': '20570', 'neutral_share': 0.0, 'enemy_share': 1.0}, {'match_id': '20996', 'neutral_share': 0.0, 'enemy_share': 1.0}, {'match_id': '21223', 'neutral_share': 0.0, 'enemy_share': 1.0}, {'match_id': '21224', 'neutral_share': 0.0, 'enemy_share': 1.0}, {'match_id': '21225', 'neutral_share': 0.0, 'enemy_share': 1.0}, {'match_id': '22218', 'neutral_share': 0.0, 'enemy_share': 1.0}]`

## Cannot-determine list

- **UNKNOWN** Exact expansion objective function (nearest-neutral vs fog-BFS vs score weights).
- **UNKNOWN** Whether expansion uses half-moves vs full-moves as a deliberate land rule (see army/timing analysts).
- **UNKNOWN** Pre-contact voluntary stop thresholds (tile/army/time) independent of contact.
- **UNKNOWN** Fog-aware exploration target cells (replays lack action intents).
- **UNKNOWN** Priority between local fill and toward-enemy-general when both conflict (rates overlap; need paired conflict subset — not fully isolated here).

## Loss skim (excluded from thresholds)

**MEASURED** n=11 losses skimmed; `excluded_from_rule_derivation=True`.

| match_id | opponent | ticks | contact | tiles@c | tiles@50 | comps@exp_end | stall | tile_lead@c |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20583 | Non-Linear Slob | 192 | 74 | 36 | 25 | 1 | 0 | 0 |
| 20595 | Nicholas | 427 | 31 | 8 | 21 | 1 | 1 | -2 |
| 20603 | __ | 201 | 87 | 49 | 24 | 1 | 0 | 5 |
| 20923 | Jonas Dujava | 132 | 34 | 11 | 22 | 1 | 0 | -3 |
| 24184 | bist | 619 | 206 | 29 | 7 | 1 | 3 | -102 |
| 24185 | bist | 535 | 209 | 13 | 4 | 1 | 3 | -111 |
| 24186 | bist | 170 | 123 | 16 | 7 | 1 | 1 | -55 |
| 24187 | bist | 241 | 145 | 11 | 4 | 1 | 2 | -73 |
| 24188 | bist | 366 | 232 | 17 | 6 | 1 | 1 | -133 |
| 24189 | bist | 344 | 157 | 13 | 6 | 1 | 2 | -85 |
| 24988 | erik.bystron | 88 | 82 | 45 | 24 | 1 | 0 | 13 |

**INFERRED** (skim only; excluded from thresholds): several losses vs `bist` show expansion stall — tiles@t50 in {4..7} vs fit median 24, large negative tile lead at contact, and `expansion_stall` events. Early-contact losses (e.g. 20595, 20923) meet the enemy with low land. These are failure modes, not rule sources.

## Files

- Script: `scripts/grok-analyze/analyze_kubic_expansion.py`
- JSON: `docs/research/measurements/grok-kubic-expansion.json`
- This report: `docs/research/measurements/grok-kubic-expansion.md`
- Split: `docs/research/measurements/grok-kubic-corpus-split.json`

