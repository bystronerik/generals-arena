# Remote 95/100 evaluation blocker

The live 95/100 **scored** block is not blocked on credentials anymore.
`GENERALS_USER_ID` is present in `.env.agent` (local only; do not commit).

Setup reference:
[`docs/engine/remote-play-setup.md`](../../engine/remote-play-setup.md).

## Offline champion status

| Field | Value |
| --- | --- |
| Champion | `army_convey` |
| Selection | `human-95-plan.md` absent; round1 Elo leader (90% winrate, 0 losses) |
| Challengers | `late_rush`, `fog_scout` (round1 rank 2–3) |
| Revision | Parameter revision 2 — stalemate opponent escalation (turn ≥ 600) |
| Stress test | [`champion-stress.md`](champion-stress.md) — 20 champion games, **14W-0L-6D** |
| Classic duel | **Ready** — rev2 gather tuning: 60% vs `army_convey`, 83.3% stress grid |

The offline dry run can run now:

```bash
.venv/bin/python scripts/remote_play.py --mode dry-run --bot classic_duel
```

## Blocker

~~Result fidelity (win-on-disconnect)~~ **Fixed** in `0b2ed66` /
`FidelityGeneralsIOClient` — see [`remote-batch1.md`](remote-batch1.md).

~~`classic_duel`~~ **Ready** at `aa57f8f`.

Current live blockers for the 95/100 human block:

1. ~~**`chat_message` receive_error**~~ — **Fixed** in `658fbb4`; upstream-style ignore for benign events. Live proof pending (batch 3–4 queue idle). See [`remote-batch3.md`](remote-batch3.md).
2. **Queue intermittency** — batch 1 had 0 matches in 25+ min; batch 2 matched 2 games in ~2 min; batches 3–4 waited 34–62 min with 0 matches. Do **not** idle public 1v1 queue without a lobby id. Use [`scripts/remote_lobby_watch.py`](../../../scripts/remote_lobby_watch.py) or set `GENERALS_LOBBY_ID` in `.env.agent` and `--mode lobby`.
3. **`GENERALS_BOT_KEY`** — not set in `.env.agent`; placeholder key may affect sustained queue access (§5.1). Set when an operator key is available.

### Lobby watch (preferred over idle 1v1)

When an operator sets `GENERALS_LOBBY_ID` in `.env.agent`, run:

```bash
python scripts/remote_lobby_watch.py --bot classic_duel --max-games 20
```

Poll interval: 15 s. Exit when counted human games reach `--max-games` or the
lobby id is cleared. Optional `--max-watch-minutes 15` for a timed wait with
no lobby id.

Only logs with `counts_toward_block: true` and human opponents count toward
Phase 3.
