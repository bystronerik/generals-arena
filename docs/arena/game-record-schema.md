# Game record schema (Phase 2)

Minimum fields for one stored match under
`data/games/<round>/<game_id>.json` (batch rounds) or
`data/games/<game_id>.json` (legacy / single-match store).

## Required fields (schema v1)

| Field | Type | Notes |
| --- | --- | --- |
| `game_id` | string | unique id |
| `seed` | int | matchup seed |
| `mode` | string | always `"competition"` for arena games |
| `bot_a` | string | bot id / path label |
| `bot_b` | string | bot id / path label |
| `bot_a_commit_or_tag` | string | version pin for bot A |
| `bot_b_commit_or_tag` | string | version pin for bot B (optional but recommended) |
| `winner` | string | `"a"` \| `"b"` \| `"draw"` |
| `turns` | int | turns played |
| `terminated` | bool | general capture / deathtouch win path |
| `truncated` | bool | hit turn cap |
| `started_at` | string | ISO-8601 |
| `finished_at` | string | ISO-8601 |

## Optional fields (schema v2)

New records from `arena/run_match.py` set `schema_version` to `2`. Older records
default to `1` when the key is absent. Read every optional key with `.get()`;
missing means not measured, never zero.

| Field | Type | Source |
| --- | --- | --- |
| `schema_version` | int | `2` for new records; `1` when absent |
| `duration_seconds` | float or null | wall clock between `started_at` and `finished_at` |
| `castles_built_a` / `_b` | int or null | `[matchup] castles built: N (...) vs M (...)` |
| `final_land_a` / `_b` | int or null | bot stderr `[telemetry]` line (last frame) |
| `final_army_a` / `_b` | int or null | same |
| `metrics` | object | free-form counters; default `{}` |

### Bot stderr telemetry

Each bot may write one line to stderr when stdin reaches EOF:

```
[telemetry] player=<0|1> turn=<int> my_land=<int> my_army=<int> opp_land=<int> opp_army=<int> [enemy_general_sighted=<0|1>] [first_sighting_turn=<int>]
```

`run_match.py` takes the last line per player, maps player 0 to bot A, and
fills the opposite land/army from `opp_*` when only one bot reports.

Sighting fields are copied into `metrics`:

| Metric key | Meaning |
| --- | --- |
| `enemy_general_sighted_a` / `_b` | bool — bot ever saw the enemy general |
| `first_sighting_turn_a` / `_b` | int — turn of first sighting |

See [`docs/research/strategies/optimize-existing.md`](../research/strategies/optimize-existing.md) §5.2.

## Rules

- Store every game **before** updating ratings.
- Arena code lives in `arena/store.py` (Phase 2).
- Do not invent extra required fields without updating this page.
- Arena matches set `mode` to `"competition"` only.

## Related

- [match-runner.md](match-runner.md)
- [ratings.md](ratings.md)
- [tournament.md](tournament.md)
