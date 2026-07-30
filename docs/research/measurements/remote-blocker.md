# Remote 95/100 evaluation blocker

The live 95/100 evaluation is blocked until `GENERALS_USER_ID` is set in the
environment.

Set up the credential as described in
[`docs/engine/remote-play-setup.md`](../../engine/remote-play-setup.md).
Do not commit the credential.

The offline dry run can run now:

```bash
.venv/bin/python scripts/remote_play.py --mode dry-run --bot army_convey
```
