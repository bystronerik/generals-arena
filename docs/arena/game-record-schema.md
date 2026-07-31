# Game record schema

Minimum fields for one stored match under
`data/games/<round>/<game_id>.json` (batch rounds) or
`data/games/<game_id>.json` (single-match store).

**Schema v5 is the only readable version.** `GameRecord.from_dict` rejects
anything older, loudly. There is never a two-branch reader: the stored v4 pool
was projected onto v5 once, in place, by
[`scripts/migrate_games_v5.py`](../../scripts/migrate_games_v5.py) — exactly as
v4 refused to read v3.

The split rule at v5: **required fields are the rating identity and the
outcome; everything observational lives in `metrics`.**

## Required fields (schema v5)

| Field | Type | Notes |
| --- | --- | --- |
| `game_id` | string | unique id (embeds a UTC minute stamp) |
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
| `truncated` | bool | hit the turn cap |

`truncated` is outcome, not observation: a 1200-cap stall and a simultaneous
general capture (RULES.md §02) are both `winner == "draw"`, and only this field
tells them apart.

`bot_a_commit_or_tag` / `bot_b_commit_or_tag` were **removed** at v4. A repo
HEAD pin is not a bot version: it moves when any unrelated file is committed
and stays put when a bot is edited without committing. The per-hash commit now
lives in the version registry
([bot-version-registry.md](bot-version-registry.md)), where it is recorded
alongside a dirty-tree flag and a git ref that actually resolves.

## Optional fields

| Field | Type | Source |
| --- | --- | --- |
| `schema_version` | int | `5`; absent or lower is a hard error |
| `metrics` | object | every observation; default `{}` |

## Removed at v5

| Field | Why |
| --- | --- |
| `started_at`, `finished_at`, `duration_seconds` | no reader anywhere in `arena/` or `scripts/`; `game_id`'s embedded stamp keeps minute provenance |
| `terminated` | exactly `winner != "draw"`, so v5 derives it (`GameRecord.terminated` is a property, not a stored field) |
| `castles_built_a` / `_b`, `final_land_*`, `final_army_*` | observations, moved into `metrics` under the same names |

## Metrics

Read every key with `.get()`; **missing means not measured, never zero.**

| Key | Type | Source |
| --- | --- | --- |
| `castles_built_a` / `_b` | int | engine tally of castle births per seat |
| `final_land_a` / `_b`, `final_army_a` / `_b` | int | terminal-state truth, on every record |
| `land_margin_a` / `_b` | int | `final_land_a - final_land_b` and its negation, on truncated draws |
| probe and reducer output | per schema | **recorded games only** — see [trajectories.md](trajectories.md) |

Records migrated from v4 keep their stderr-derived values under the same key
names, so a last-frame question is still answerable on old games while a
per-turn one is honestly unanswerable. One v4 name meant two measurements: the
top-level `castles_built_a/_b` (engine tally) and metro's own counter inside
`metrics`. They disagree on 16 of the 1000 games carrying both, so the
projection keeps the engine tally under the plain name and moves the bot's
belief to `castles_built_probe_a/_b`.

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
