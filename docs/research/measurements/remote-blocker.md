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
2. **Public lobby support in `client/`** — remote eval waits on public-lobby join in `generals_client`, not private custom games. Do **not** run two bots in a private lobby to prove the wire; that adds latency and does not count toward the human block. Use competition-module / classic harness for local bot-vs-bot.
3. **Queue intermittency** — batch 1 had 0 matches in 25+ min; batch 2 matched 2 games in ~2 min; batches 3–4 waited 34–62 min with 0 matches. After public-lobby support lands, retry 1v1 or public lobby; do not idle queue for hours without a match.
4. **`GENERALS_BOT_KEY`** — not set in `.env.agent`; placeholder key may affect sustained queue access (§5.1). Set when an operator key is available.

### Out of scope: private bot-vs-bot

Private custom lobbies with two bots and force-start were attempted in batch 5
and **stopped**. They do not help the 95/100 human goal. A second bot identity
(`GENERALS_USER_ID_B` in local `.env.agent` only) was created during that probe;
do not build on it. Local match proof belongs in `arena/classic_match.py` and
competition-module harnesses.

### Next step (blocked on `client/`)

When `client/` adds **public lobby** support, re-run:

```bash
.venv/bin/python scripts/remote_play.py --mode 1v1 --bot classic_duel --max-games 10
```

Or use `remote_lobby_watch.py` once public lobby ids are supported the same way.

Only logs with `counts_toward_block: true` and **human** opponents count toward
the 95/100 human block.
