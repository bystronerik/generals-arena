# Game record schema

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
| `bot_a_commit_or_tag` | string | repo HEAD when the game ran — **not** a per-bot version (see `bot_a_content_hash`) |
| `bot_b_commit_or_tag` | string | same value as bot A for arena-run games |
| `winner` | string | `"a"` \| `"b"` \| `"draw"` |
| `turns` | int | turns played |
| `terminated` | bool | general capture / deathtouch win path |
| `truncated` | bool | hit turn cap |
| `started_at` | string | ISO-8601 |
| `finished_at` | string | ISO-8601 |

## Optional fields (schema v2+)

New records set `schema_version` to `3`. Older records default to `1` when the
key is absent. Read every optional key with `.get()`; missing means not
measured, never zero.

| Field | Type | Source |
| --- | --- | --- |
| `schema_version` | int | `3` for new records; `1` when absent |
| `duration_seconds` | float or null | wall clock between `started_at` and `finished_at` |
| `bot_a_content_hash` / `_b` | string or null | schema v3; `arena/fingerprint.py` |
| `castles_built_a` / `_b` | int or null | `[matchup] castles built: N (...) vs M (...)` |
| `final_land_a` / `_b` | int or null | bot stderr `[telemetry]` line (last frame) |
| `final_army_a` / `_b` | int or null | same |
| `metrics` | object | free-form counters; default `{}` |

### Bot content hash (schema v3)

`bot_a_content_hash` / `bot_b_content_hash` pin the *bot*, where
`bot_a_commit_or_tag` only pins the repo. A 12-hex-character SHA-256 over the
bot's source closure, computed by `arena/fingerprint.py`:

- every file in `bots/<id>/`, excluding `__pycache__` and `*.pyc`;
- every module under `bots/` it imports, transitively — this crosses bot
  directories, so `proteus` covers `aegis`, `blitz`, `boom` and `metro`, and
  every bot covers `_common/wire.py`;
- shell `source` targets, so `cm_*` bots cover `_common/cm_run.sh`.

Imports that do not resolve under `bots/` (stdlib, `jax`, `generals`) are
excluded — third-party versions are the lockfile's job. `"unknown"` when the
bot directory cannot be read. Absent (`null`) on schema v1 and v2 records.

Two games share a bot's content hash only if that bot was byte-identical,
which is what rating identity should key on: the repo commit changes when any
unrelated file is committed, and does not change when a bot is edited without
committing.

Inspect a roster:

```bash
python -m arena.fingerprint --files proteus cm_random
```

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
- Arena code lives in `arena/store.py`.
- Do not invent extra required fields without updating this page.
- Arena matches set `mode` to `"competition"` only.

## Related

- [match-runner.md](match-runner.md)
- [ratings.md](ratings.md)
- [tournament.md](tournament.md)
