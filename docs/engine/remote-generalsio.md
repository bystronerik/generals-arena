# Remote generals.io (not competition)

`competition-module/generals/remote/` talks to **live generals.io**, not the Generals Competition sandbox.

## What it is for

- Autopilot / client play on classic generals.io servers.
- Example: `generals.remote.autopilot` with a user id and lobby id.

## What it is not

- It does **not** submit to [generals.bot](https://www.generals.bot/).
- It does **not** use the competition stdio protocol in `competition/protocol.py`.
- Competition rules (build castles, deathtouch from 800, 1200-turn cap, 18–21 maps) may not match live server rooms.

## Local competition matches

Use [`local-matchup.md`](local-matchup.md) and `--mode competition` instead.
