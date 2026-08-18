# Game record schema

Minimum fields for one stored match under
`data/games/<round>/<game_id>.json` (batch rounds) or
`data/games/<game_id>.json` (single-match store).

**Schema v5 is the only readable version.** `GameRecord.from_dict` rejects
anything older, loudly. There is never a two-branch reader: the stored v4 pool
was projected onto v5 once, in place, by a one-shot migration script — exactly
as v4 refused to read v3. No pre-v5 record survives, so that script is gone.

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

- every file in `bots/<id>/`, excluding `__pycache__`, `*.pyc`, and `probe.py`;
- every module under `bots/` it imports, transitively — this can cross bot
  directories, though no bot in `bots/` imports another one today, and every
  bot covers `_common/wire.py`;
- shell `source` targets, so `cm_*` bots cover `_common/cm_run.sh`.

Imports that do not resolve under `bots/` (stdlib, `jax`, `generals`) are
excluded — third-party versions are the lockfile's job.

### The `probe.py` exclusion

`bots/<id>/probe.py` is arena-owned per-turn introspection, loaded only by
`arena.instrument.runner` on recorded matches
([trajectories.md](trajectories.md)). It never plays, so it is outside the
closure: adding or editing a probe moves no hash and reaches no bundle.

What keeps that honest is one invariant — **unhashed code must be unreachable
from the hashed program**. `bot_source_closure` raises `ProbeInClosureError`
if any module in the closure imports `probe`, because a probe the agent could
call would change how the bot plays while leaving its rating identity
untouched.

Two games share a bot's content hash only if that bot was byte-identical. The
hash is required and must never be `"unknown"`: `fingerprint.bot_content_hash`
raises instead of returning a sentinel, and both `record_from_match_result`
and `GameRecord.from_dict` reject it. A sentinel hash would pool every
unreadable closure into one rated entity.

Inspect a roster:

```bash
python -m arena.records.fingerprint --files aegis cm_random
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

## Where observations come from

**Bots emit no telemetry.** The `[telemetry]` stderr line and every
`telemetry_extras()` method were deleted: the submitted bundle and the rated
closure are pure game logic, and nothing in `arena/` parses bot stderr. Two
sources feed `metrics` instead:

| Source | Available on | Keys |
| --- | --- | --- |
| the engine's terminal `GameInfo` | every match | finals, castle tallies, land margins |
| recorded trajectories | matches run with `--record` | probe output and series reducers ([trajectories.md](trajectories.md)) |

Engine finals are ground truth for both seats. They are also *post-capture*:
when a general falls the engine transfers the loser's cells to the winner
first, so a decisive game's finals are the winner's totals, not the last thing
either bot saw.

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
