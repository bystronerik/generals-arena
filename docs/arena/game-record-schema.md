# Game record schema (Phase 2)

Minimum fields for one stored match under `data/games/<game_id>.json`.

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

Optional later: per-turn stats, fault counts, duration seconds.

## Rules

- Store every game **before** updating ratings.
- Arena code lives in `arena/store.py` (Phase 2).
- Do not invent extra required fields without updating this page.
- Arena matches set `mode` to `"competition"` only.

## Related

- [match-runner.md](match-runner.md)
- [ratings.md](ratings.md)
- [tournament.md](tournament.md)

