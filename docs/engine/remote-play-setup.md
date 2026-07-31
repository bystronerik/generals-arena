# Remote play setup (live generals.io)

How to run arena heuristic bots on **classic generals.io** through
`client/generals_client` (EIO v4, no `bot_key`). This is not competition mode and
must not feed `data/ratings/`.

For evaluation goals and logging fields, see
[`remote-eval-heuristics.md`](remote-eval-heuristics.md). For the unified API, see
[`unified-bot-api.md`](unified-bot-api.md).

---

## Prerequisites

```bash
source .venv/bin/activate
pip install -e competition-module    # local competition matches only
pip install -e client                # generals_client wire for live play
pip install -r requirements.txt
```

Offline-verify the adapter (no credentials):

```bash
python scripts/remote_play.py --mode dry-run --bot classic_duel
```

---

## Environment variables

| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `GENERALS_USER_ID` | **yes** (live) | — | Secret id you invent; treated like a password |
| `GENERALS_USERNAME` | no | `[Bot] arena_<bot>` | Username on generals.io; **must** use `[Bot]` prefix |
| `GENERALS_LOBBY_ID` | no | `arena-test` | Private lobby id for `--mode lobby` or lobby watch |

Rules:

- Read secrets from the environment only. Do not pass `user_id` on the CLI
  (shell history risk).
- `.env` and `.env.agent` at the repo root are gitignored. The script loads
  them if present, without overwriting variables already in the shell.
- Never commit credentials or game JSON from live sessions.
- `GENERALS_BOT_KEY` is **not** used. Remote play goes through `generals_client`.

Example:

```bash
export GENERALS_USER_ID='<long random string you invent>'
export GENERALS_USERNAME='[Bot] arena_army_convey'
export GENERALS_LOBBY_ID='arena-test'
```

---

## Recommended bots

Top round-1 heuristics that translate cleanly to classic rules:

| Bot | Why |
| --- | --- |
| `army_convey` | Army funneling is rule-independent |
| `late_rush` | Expansion + rush (deathtouch branch is dead on classic) |
| `fog_scout` | Scouting transfers well |
| `expand_plus` | Clean expansion baseline |
| `classic_duel` | Remote evaluation champion |

Avoid `castle_builder`, `castle_rush`, and `phase_switch` remotely — build
actions (`pass=2`) are silently ignored on generals.io.

---

## Running

Private lobby (safe first test — a human joins the same lobby id):

```bash
python scripts/remote_play.py --bot army_convey --mode lobby
```

Public 1v1 queue (only after lobby test passes). The script requeues with
exponential backoff when the queue does not assign a game within
`--queue-timeout-seconds` (default 600 s):

```bash
python scripts/remote_play.py --bot army_convey --mode 1v1 --max-games 5 \
  --queue-timeout-seconds 600 --session-minutes 120
```

Use `--session-minutes` to cap total wall-clock time for a 1v1 block. On exit
(normal, interrupt, or cap) the script prints a session summary and writes
`session_summary_*.json` under `data/remote_games/`.

Lobby watch — poll `.env.agent` until `GENERALS_LOBBY_ID` is set, then play
lobby games without idle 1v1 queue time:

```bash
python scripts/remote_lobby_watch.py --bot classic_duel --max-games 20
```

Optional: exit if no lobby id appears within 15 minutes:

```bash
python scripts/remote_lobby_watch.py --max-watch-minutes 15
```

The watcher never prints secret values (user id or lobby id). It counts only
human games with `counts_toward_block: true` and `opponent_is_bot: false`, and
stops when the target is reached or `GENERALS_LOBBY_ID` is cleared from
`.env.agent`.

Run offline checks before connecting:

```bash
python scripts/remote_play.py --bot fog_scout --mode lobby --verify-offline
```

### Server URL

Both scripts accept `--server-url` (default `https://botws.generals.io`) and
`--public-server` (alias for the same public bot endpoint). The URL is passed
through `FidelityRemoteSession` to `generals_client`.

### Block report

Aggregate counted human games and emit Wilson bounds plus star-band splits:

```bash
python scripts/remote_report.py
python scripts/remote_report.py --output docs/research/measurements/remote-report.md
```

Only games with `counts_toward_block: true` and `opponent_is_bot: false` enter
the headline win rate.

---

## Game logs

Each finished game writes one JSON file under `data/remote_games/` (gitignored).
Fields match the table in [`remote-eval-heuristics.md`](remote-eval-heuristics.md)
§5.

Remote results do **not** update arena Elo in `data/ratings/`.

---

## Blockers checklist

1. **`GENERALS_USER_ID` missing** — script exits with setup instructions.
2. **`generals_client` not installed** — run `pip install -e client`.
3. **No network** — live modes need outbound access to `botws.generals.io`.
4. **Build-economy bot selected** — every build tick becomes a silent pass.
5. **95% vs humans** — requires 100+ logged games against humans; do not claim
   that target without that sample.

---

## Architecture

- `arena/bot_api.py` — unified observation/action types and mappers.
- `arena/remote/bridge.py` — `UnifiedBot` + `ArenaGameClient` over `generals_client`.
- `arena/remote/client.py` — `FidelityRemoteSession`, session runners, JSON logging.
- `arena/remote/env.py` — repo paths, dotenv load, credential checks, server URL.
- `arena/remote/report.py` — aggregate human-block stats, Wilson bound, star bands.
- `scripts/remote_play.py` — CLI, game runner, session summary on exit.
- `scripts/remote_report.py` — markdown report CLI over `data/remote_games/`.

Strategy code stays in `bots/<name>/agent.py`; wire details never leak into bots.
