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
| Classic duel | In progress (remote classic-rules harness per [`human-95-plan.md`](../strategies/human-95-plan.md)) |

The offline dry run can run now:

```bash
.venv/bin/python scripts/remote_play.py --mode dry-run --bot army_convey
```

## Blocker

The **95/100 human block count** stays gated on **result fidelity** (win-on-disconnect):
disconnect and receive-error paths must not be recorded as wins. See
[`human-95-plan.md`](../strategies/human-95-plan.md) §5.4.

Until that fix ships, do not treat live remote logs as valid for the Phase 3
headline score even though `GENERALS_USER_ID` is configured.
