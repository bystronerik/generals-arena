# Morpheus bootstrap measurement corpus

> Verdict: **yes** — Panel names exact registered hashes; all games are competition mode; every trajectory verifies in one engine era; decisive and forced-mismatch cases are present for Parts 03, 05, and 10.

Generated: 2026-08-03T18:50:54.050137+00:00

## Checks

| Check | Result |
| --- | --- |
| `panel_hashes_named` | yes |
| `panel_roles_complete` | yes |
| `competition_mode_only` | yes |
| `single_engine_era` | yes |
| `all_trajectories_verified` | yes |
| `source_labels_present` | yes |
| `enough_trajectories` | yes |
| `enough_decisive` | yes |
| `enough_forced_mismatch` | yes |
| `critical_events_present` | yes |

## Panel

Name: `morpheus-bootstrap`
Selection date: `2026-08-04`
Rating era: `9e3b9d13cca51caa1bb07db48bb85c9e90ce0462`
Source label: `fixed_panel`

| Bot | Hash | Role | Rating | Decisive |
| --- | --- | --- | ---: | ---: |
| `cm_expander` | `3097ee53a533` | anchor | 1500.0 | 5214 |
| `castle_builder` | `34c33cba59d8` | heuristic | 1408.8 | 95 |
| `fog_scout` | `62b1acf0c289` | heuristic | 1861.5 | 1534 |
| `aegis` | `b160fb7cbf75` | heuristic | 2050.1 | 2834 |
| `boom` | `9e4a37370ddf` | heuristic | 2166.6 | 3178 |
| `macaria` | `5760b30723e4` | research | 2291.6 | 723 |
| `sosipolis` | `e8d618ef7dc6` | research | 2034.5 | 245 |

## Coverage

Trajectories: 42
Modes: `{'competition': 42}`
Engine versions: `{'9e3b9d13cca51caa1bb07db48bb85c9e90ce0462': 42}`
Source labels: `{'fixed_panel': 42}`

### Events

| Event | Count |
| --- | ---: |
| `contact` | 42 |
| `sight` | 41 |
| `castle` | 20 |
| `deathtouch` | 8 |
| `decisive` | 37 |
| `forced_mismatch_eligible` | 42 |

### Board sizes

```json
{
  "18x18": 2,
  "18x19": 4,
  "18x20": 4,
  "18x21": 2,
  "19x18": 4,
  "19x20": 2,
  "19x21": 2,
  "20x20": 6,
  "20x21": 2,
  "21x18": 4,
  "21x19": 8,
  "21x20": 2
}
```

### Turn bands

```json
{
  "1-200": 2,
  "1001-1200": 6,
  "201-400": 12,
  "401-600": 14,
  "601-800": 6,
  "801-1000": 2
}
```

### Outcomes

```json
{
  "a": 17,
  "b": 20,
  "draw": 5
}
```

## Missing classes

- `board_size`: ['19x19', '20x18', '20x19', '21x21']

## Verify

ok=42 fail=0 era_mismatch=0

Trajectories dir: `/Users/erikbystron/Work/learning/generals-arena/data/trajectories/morpheus-bootstrap`

Machine-readable: [`morpheus-bootstrap-corpus.json`](morpheus-bootstrap-corpus.json)
