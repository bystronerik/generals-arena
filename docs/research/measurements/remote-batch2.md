# Remote batch 2 — classic_duel on botws

Generated: 2026-07-31T00:55:00Z  
Bot: `classic_duel` (commit `aa57f8f`, session at `41400f6`)  
Prior batch: [`remote-batch1.md`](remote-batch1.md) (army_convey, 0 games in 25+ min queue)

## Credentials (presence only)

| Key | Present |
| --- | --- |
| `GENERALS_USER_ID` | yes |
| `GENERALS_USERNAME` | yes |
| `GENERALS_BOT_KEY` | no |
| `GENERALS_LOBBY_ID` | no |

## Pre-flight

Dry-run passed:

```bash
.venv/bin/python scripts/remote_play.py --mode dry-run --bot classic_duel
```

## Live session

| Setting | Value |
| --- | --- |
| Mode | `1v1` (`botws.generals.io`, default — not `public_server`) |
| `--max-games` | 10 (stopped after game 3 queue wait) |
| Bot | `classic_duel` |
| Username | `[Bot] BOBTHEAGENT` (registers as `BOBTHEAGENT`) |
| `GENERALS_BOT_KEY` | not set (placeholder key in client) |

Timeline:

1. **00:32:03Z** — connected, registration skipped (username already bound).
2. **00:34:07Z** — game 1 started (~2 min queue). Ended turn 0 with `receive_error`.
3. **00:34:38Z** — game 2 started (~31 s queue). Same error, same human opponent.
4. **00:34:38–00:52Z** — game 3 queue wait ~18 min with no `game_start`; session stopped.

## Per-game log summary

| File | Opponent | Human? | Result | `counts_toward_block` | Detail |
| --- | --- | --- | --- | --- | --- |
| `20260731T003407Z_classic_duel_receive_error.json` | Among | yes | `error` | **no** | `unexpected event 'chat_message'` |
| `20260731T003438Z_classic_duel_receive_error.json` | Among | yes | `error` | **no** | `unexpected event 'chat_message'` |

Both games had `server_turns: 0`, `peak_land: 0` — the bot left before any turn played.

## Counted vs discarded

| Category | W | L | D | Notes |
| --- | ---: | ---: | ---: | --- |
| **Counted** (`counts_toward_block=true`, human opponents) | 0 | 0 | 0 | none finished |
| Discarded (`receive_error`) | — | — | — | 2 games (chat_message) |
| Queue timeout (game 3) | — | — | — | no match in ~18 min |

**Winrate toward 95/100 block:** undefined (0 counted games).

## Findings vs batch 1

| Topic | Batch 1 (`army_convey`) | Batch 2 (`classic_duel`) |
| --- | --- | --- |
| Queue matchmaking | 0 games in 25+ min | 2 games in ~2 min |
| Endpoint | `botws.generals.io` | same |
| `GENERALS_BOT_KEY` | not set | not set |
| Root cause of 0 counted | queue idle | **`chat_message` treated as fatal in `FidelityGeneralsIOClient`** |

Batch 1 queue idle may have been time-of-day or transient. Batch 2 shows botws 1v1 **can** match without `GENERALS_BOT_KEY`.

The fidelity client's `_play_game` ends on any event outside
`game_update` / `game_won` / `game_lost`. Upstream `GeneralsIOClient._play_game`
ignores unknown events (no default case). Opponents sending lobby chat causes an
immediate `receive_error` discard at turn 0.

## Blockers for batch 3

1. **`chat_message` handling** — ignore or log non-game events instead of
   `receive_error` exit. Required before any counted human game can finish.
2. **`GENERALS_BOT_KEY`** — still unset; may matter for sustained queue access
   but did not block the two quick matches here.
3. **Queue intermittency** — game 3 waited ~18 min after two fast matches;
   peak-hours retest or `--mode lobby` with a coordinated human for Gate A.

## Recommendations

| Option | When to use |
| --- | --- |
| **Fix `chat_message` in fidelity client** | First — unblocks all live games |
| **Stay on `botws`** (default) | Matchmaking worked; do not switch to `public_server` yet |
| **`--mode lobby` + human** | Gate A smoke test after chat fix; set `GENERALS_LOBBY_ID` |
| **`GENERALS_BOT_KEY`** | Obtain operator key when available; test whether it stabilizes queue |

## Next batch plan

1. Patch `FidelityGeneralsIOClient._play_game` to skip `chat_message` (and other
   benign server events) like upstream.
2. Re-run `--mode 1v1 --bot classic_duel --max-games 10` on botws.
3. If queue idle recurs, try `--mode lobby` with a known human.
4. Only sum logs where `counts_toward_block == true` and `opponent_is_bot == false`.
