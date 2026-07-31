# Remote batch 1 — post-fidelity-fix

Generated: 2026-07-31T00:30:00Z  
Bot: `army_convey` (no `classic_duel` yet)  
Block commit: `41400f6` (includes fidelity fix `0b2ed66`)

## Fidelity fix (required before any counted game)

Upstream `GeneralsIOClient._play_game` treated `ValueError` from `receive()` as
a win. That could fake a 95/100 block.

Repo-side fix in `arena/remote_client.py` (`FidelityGeneralsIOClient`):

| Path | `result_reason` | `result` | `counts_toward_block` |
| --- | --- | --- | --- |
| `game_won` | `game_won` | `win` | **yes** |
| `game_lost` | `game_lost` | `loss` | **yes** |
| `ValueError` on receive | `disconnect` | `disconnect` | no |
| Frame with ≠ 3 elements | `receive_error` | `error` | no |
| Unknown event | `receive_error` | `error` | no |

Each log also records `endpoint`, `opponent_stars` (when available), and
`finish_detail` for non-decided endings.

Unit tests: `tests/test_remote_client.py` (malformed receive is not a win).

Commits:

- `0b2ed66` — fidelity client + logging fields
- `41400f6` — tolerate “already have a username” on repeat sessions

## Pre-fix session errors (discarded)

These files are **not** counted toward the 95/100 block. They predate the
fidelity fix and never finished a game:

| File | Cause |
| --- | --- |
| `session_error_1785455557.json` | Registration rejected `[Bot]` prefix |
| `session_error_1785456312.json` | Username already bound to user id |

## Live session after fix

| Setting | Value |
| --- | --- |
| Mode | `1v1` (`botws.generals.io`) |
| `--max-games` | 10 |
| `GENERALS_LOBBY_ID` | not set |
| `GENERALS_BOT_KEY` | not set (placeholder key in client) |

Session log:

1. Connected to bot endpoint.
2. Registration skipped (“user id already has a bound username”).
3. Joined 1v1 queue; **no `game_start` after ~25 minutes** — session stopped.

## Counted vs discarded games (post-fix)

| Category | W | L | D | Notes |
| --- | ---: | ---: | ---: | --- |
| **Counted** (`counts_toward_block=true`, human opponents) | 0 | 0 | 0 | none finished |
| Discarded (`disconnect` / `error` / pre-fix errors) | — | — | — | 2 session errors only |

**Winrate toward 95/100 block:** undefined (0 counted games).

## Blockers for next batch

1. **Queue matchmaking** — 1v1 bot queue did not assign a game in 25+ minutes.
   Try `--mode lobby` with a coordinated human, or run during peak hours.
2. **`GENERALS_BOT_KEY`** — human-95-plan §5.1 notes the placeholder
   `sd09fjd203i0ejwi_changeme` may block real queue play; obtain an operator key
   and set `GENERALS_BOT_KEY` in `.env.agent`.
3. **`classic_duel`** — champion bot not implemented yet; `army_convey` remains
   the interim remote bot (build actions dropped, no deathtouch suicide).

## Next batch plan

1. Set `GENERALS_BOT_KEY` when available.
2. Gate A from human-95-plan: 1–2 `--mode lobby` games with a known human
   (`GENERALS_LOBBY_ID`), then 10× `--mode 1v1`.
3. Re-run with `--max-games 20`; only sum logs where
   `counts_toward_block == true` and `opponent_is_bot == false`.
4. Publish `remote-block1.md` after 100 counted human games (not this batch).
