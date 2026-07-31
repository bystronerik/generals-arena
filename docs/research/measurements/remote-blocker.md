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
| Classic duel | **Ready** at `aa57f8f` — remote champion bot per [`human-95-plan.md`](../strategies/human-95-plan.md) |

The offline dry run can run now:

```bash
.venv/bin/python scripts/remote_play.py --mode dry-run --bot classic_duel
```

## Blocker

~~Result fidelity (win-on-disconnect)~~ **Fixed** in `0b2ed66` /
`FidelityGeneralsIOClient` — see [`remote-batch1.md`](remote-batch1.md).

~~`classic_duel`~~ **Ready** at `aa57f8f`.

Current live blockers for the 95/100 human block:

1. **`chat_message` receive_error** — batch 2 matched 2 human games on botws but
   both ended at turn 0 because `FidelityGeneralsIOClient` treats
   `chat_message` as fatal (upstream ignores unknown events). See
   [`remote-batch2.md`](remote-batch2.md).
2. **Queue intermittency** — batch 1 had 0 matches in 25+ min; batch 2 matched
   2 games in ~2 min then waited ~18 min for game 3. Retest after chat fix.
3. **`GENERALS_BOT_KEY`** — not set in `.env.agent`; placeholder key may affect
   sustained queue access (§5.1). Set when an operator key is available.

Only logs with `counts_toward_block: true` and human opponents count toward
Phase 3.
