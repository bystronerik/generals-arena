# Remote batch 4 — long queue wait + public-server fallback

Generated: 2026-07-31T02:31:43Z  
Bot: `classic_duel` (commit `658fbb4`)  
Prior batch: [`remote-batch3.md`](remote-batch3.md) (queue idle ~34 min on botws)

## Credentials (presence only)

| Key | Present |
| --- | --- |
| `GENERALS_USER_ID` | yes |
| `GENERALS_USERNAME` | yes |
| `GENERALS_BOT_KEY` | no |
| `GENERALS_LOBBY_ID` | no |

Loaded from `.env.agent` at session start. Lobby mode was not used (no lobby id; 1v1 queue is the path for automatic human matchmaking).

## CLI change

Added `--public-server` to `scripts/remote_play.py` for one alternate endpoint attempt (`ws.generals.io` instead of default `botws.generals.io`).

## Live sessions

| Phase | Endpoint | `--max-games` | Queue wait | Matches |
| --- | --- | ---: | --- | ---: |
| 1 (retry) | `botws.generals.io` | 20 | ~31 min | 0 |
| 2 (fallback) | `ws.generals.io` (`--public-server`) | 20 | ~31 min | 0 |

Username: `[Bot] BOBTHEAGENT` (registers as `BOBTHEAGENT` on bot endpoint; registers with `[Bot]` prefix stripped on public endpoint too).

### Phase 1 — botws (primary)

1. **01:29:47Z** — first connect attempt failed (`Read timed out`, 5 s socket.io timeout); logged `session_error_1785461393.json`.
2. **01:30:12Z** — retry connected; registration skipped (username already bound).
3. **01:30:12Z** — joined 1v1 queue.
4. **01:30–02:01Z** — no `game_start`; intermittent `packet queue is empty, aborting` on receive polling (same as batch 3).
5. **02:01:02Z** — stopped manually after ~31 min idle queue.

### Phase 2 — public server (one alternate attempt)

1. **02:01:02Z** — connected to `ws.generals.io`; agent registered successfully (fresh registration on public endpoint).
2. **02:01:02Z** — joined 1v1 queue.
3. **02:01–02:31Z** — no `game_start`; queue idle.
4. **02:31:43Z** — stopped after ~31 min idle queue.

**Total wall clock:** ~62 min across both endpoints. **Endpoint used for counted games:** neither (0 matches on both).

## Per-game log summary

No new game logs from batch 4. Latest game logs remain batch 2 `receive_error` entries (pre–`658fbb4` fix).

## Counted vs discarded (batch 4 only)

| Category | W | L | D | Notes |
| --- | ---: | ---: | ---: | --- |
| **Counted** (`counts_toward_block=true`, human opponents) | 0 | 0 | 0 | no matches |
| Discarded | 0 | — | — | — |
| Connect errors | — | — | — | 1 botws timeout (retry succeeded) |
| Queue idle | — | — | — | 0 matches in ~62 min combined |

**Winrate toward 95/100 block (batch 4):** undefined (0 counted games).

## Cumulative block progress (all batches)

| Batch | Counted W | Counted L | Counted D | Notes |
| --- | ---: | ---: | ---: | --- |
| 1 (`army_convey`) | 0 | 0 | 0 | queue idle 25+ min |
| 2 (`classic_duel`) | 0 | 0 | 0 | 2 discarded (`chat_message`) |
| 3 (`classic_duel`, post-fix) | 0 | 0 | 0 | queue idle ~34 min |
| 4 (`classic_duel`, post-fix) | 0 | 0 | 0 | queue idle ~62 min (botws + public) |
| **Total** | **0** | **0** | **0** | **0 / 100 counted** |

**Cumulative winrate:** undefined (0 counted games).

## Findings

1. **`chat_message` fix still unvalidated live** — no human games finished after `658fbb4`; batch 2 opponents (`Among`) would have been counted wins/losses if the fix had been in place.
2. **Queue matchmaking is highly intermittent** — batch 2 matched 2 games in ~2 min on botws; batches 1, 3, and 4 saw 0 matches across 25–62 min waits on the same credentials.
3. **Public server (`ws.generals.io`) did not improve match rate** — connected and registered, but 0 matches in ~31 min; prefer botws for future runs per `human-95-plan.md`.
4. **`GENERALS_BOT_KEY`** — still unset.

## Loss / tuning notes

No counted wins or losses this batch. No tuning input.

When the next human game completes on `658fbb4`, verify:

- `counts_toward_block == true`
- `result_reason` is `game_won` or `game_lost` (not `receive_error`)
- `server_turns > 0`
- `finish_detail` absent or not `chat_message`

## Next batch plan

1. Re-run at peak US/EU hours, or coordinate `--mode lobby` with a human and set `GENERALS_LOBBY_ID`.
2. Obtain `GENERALS_BOT_KEY` when available.
3. Stay on `botws.generals.io` (default); use `--public-server` only as a diagnostic fallback.
