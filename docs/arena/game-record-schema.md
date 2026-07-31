# Game record schema

Minimum fields for one stored match under
`data/games/<round>/<game_id>.json` (batch rounds) or
`data/games/<game_id>.json` (single-match store).

**Schema v4 is the only readable version.** `GameRecord.from_dict` rejects
anything older, loudly. Every pre-v4 record was deleted rather than migrated:
none of them carried a content hash, so none of them had a rating identity.

## Required fields (schema v4)

| Field | Type | Notes |
| --- | --- | --- |
| `game_id` | string | unique id |
| `seed` | int | map seed |
| `mode` | string | always `"competition"` for arena games |
| `round` | string | round name — **stored, not inferred from the path** |
| `bot_a` | string | bot id / path label |
| `bot_b` | string | bot id / path label |
| `bot_a_content_hash` | string | 12-hex source-closure hash — the rating identity |
| `bot_b_content_hash` | string | same, for bot B |
| `engine_version` | string | `competition-module` submodule SHA |
| `winner` | string | `"a"` \| `"b"` \| `"draw"` |
| `turns` | int | turns played |
| `terminated` | bool | general capture / deathtouch win path |
| `truncated` | bool | hit turn cap |
| `started_at` | string | ISO-8601 |
| `finished_at` | string | ISO-8601 |

`bot_a_commit_or_tag` / `bot_b_commit_or_tag` were **removed** at v4. A repo
HEAD pin is not a bot version: it moves when any unrelated file is committed
and stays put when a bot is edited without committing. The per-hash commit now
lives in the version registry
([bot-version-registry.md](bot-version-registry.md)), where it is recorded
alongside a dirty-tree flag and a git ref that actually resolves.

## Optional fields

Read every optional key with `.get()`; missing means not measured, never zero.

| Field | Type | Source |
| --- | --- | --- |
| `schema_version` | int | `4`; absent or lower is a hard error |
| `duration_seconds` | float or null | wall clock between `started_at` and `finished_at` |
| `castles_built_a` / `_b` | int or null | `[matchup] castles built: N (...) vs M (...)` |
| `final_land_a` / `_b` | int or null | bot stderr `[telemetry]` line (last frame) |
| `final_army_a` / `_b` | int or null | same |
| `metrics` | object | free-form counters; default `{}` |

## Bot content hash

`bot_a_content_hash` / `bot_b_content_hash` pin the *bot*. A 12-hex-character
SHA-256 over the bot's source closure, computed by
`arena/records/fingerprint.py`:

- every file in `bots/<id>/`, excluding `__pycache__` and `*.pyc`;
- every module under `bots/` it imports, transitively — this crosses bot
  directories, so `proteus` covers `aegis`, `blitz`, `boom` and `metro`, and
  every bot covers `_common/wire.py`;
- shell `source` targets, so `cm_*` bots cover `_common/cm_run.sh`.

Imports that do not resolve under `bots/` (stdlib, `jax`, `generals`) are
excluded — third-party versions are the lockfile's job.

Two games share a bot's content hash only if that bot was byte-identical. The
hash is required and must never be `"unknown"`: `fingerprint.bot_content_hash`
raises instead of returning a sentinel, and both `record_from_match_result`
and `GameRecord.from_dict` reject it. A sentinel hash would pool every
unreadable closure into one rated entity.

Inspect a roster:

```bash
python -m arena.records.fingerprint --files proteus cm_random
```

Map a hash back to source:

```bash
python -m arena.records.registry --files <hash>
```

## Engine version

The SHA of the checked-out `competition-module`, read from the submodule's own
HEAD — the checkout that actually played the game, not the gitlink in the
superproject's tree. A rules or engine change moves win probabilities, so
records from either side of a submodule bump are **never** pooled: the rating
policy counts them under `excluded.engine_mismatch` instead. A bump invalidates
the leaderboard and needs a fresh regeneration round.

## Round

Stored on the record so eligibility never has to parse a path. Batch rounds
pass their own name; one-off matches through `arena/matches/run_match.py`
default to `adhoc` and can override it with `--round`.

## Bot stderr telemetry

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

- Store every game **before** refitting ratings.
- Arena code lives in `arena/records/store.py`.
- Do not invent extra required fields without updating this page.
- Arena matches set `mode` to `"competition"` only.

## Related

- [match-runner.md](match-runner.md)
- [ratings.md](ratings.md)
- [bot-version-registry.md](bot-version-registry.md)
- [tournament.md](tournament.md)
