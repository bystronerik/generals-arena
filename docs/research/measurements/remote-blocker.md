# Remote 95/100 evaluation blocker

The live 95/100 evaluation is blocked until `GENERALS_USER_ID` is set in the
environment.

Set up the credential as described in
[`docs/engine/remote-play-setup.md`](../../engine/remote-play-setup.md).
Do not commit the credential.

## Offline champion status

| Field | Value |
| --- | --- |
| Champion | `army_convey` |
| Selection | `human-95-plan.md` absent; round1 Elo leader (90% winrate, 0 losses) |
| Challengers | `late_rush`, `fog_scout` (round1 rank 2–3) |
| Revision | Parameter revision 2 — stalemate opponent escalation (turn ≥ 600) |
| Stress test | [`champion-stress.md`](champion-stress.md) — 20 champion games, 0 losses, 6 draws |
| Validation | 3/3 wins vs `smoke` and `expand_plus` (seeds 0–2); `fog_scout` still draws |

The offline dry run can run now:

```bash
.venv/bin/python scripts/remote_play.py --mode dry-run --bot army_convey
```

## Blocker

Live remote play remains blocked on **`GENERALS_USER_ID`**. No credential is
committed to this repo.
