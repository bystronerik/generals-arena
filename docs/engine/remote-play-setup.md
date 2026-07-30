# Remote play setup (live generals.io)

How to run arena heuristic bots on **classic generals.io** through
`competition-module/generals/remote/`. This is not competition mode and must
not feed `data/ratings/`.

For evaluation goals and logging fields, see
[`remote-eval-heuristics.md`](remote-eval-heuristics.md).

---

## Prerequisites

```bash
source .venv/bin/activate
pip install -e competition-module    # includes python-socketio client
pip install -r requirements.txt
```

Offline-verify the adapter (no credentials):

```bash
python scripts/remote_play.py --mode dry-run --bot army_convey
```

---

## Environment variables

| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `GENERALS_USER_ID` | **yes** (live) | — | Secret id you invent; treated like a password |
| `GENERALS_USERNAME` | no | `[Bot] arena_<bot>` | Username shown on generals.io; use `[Bot]` prefix |
| `GENERALS_LOBBY_ID` | no | `arena-test` | Private lobby id for `--mode lobby` |

Rules:

- Read secrets from the environment only. Do not pass `user_id` on the CLI
  (shell history risk).
- `.env` and `.env.agent` at the repo root are gitignored. The script loads
  them if present, without overwriting variables already in the shell.
- Never commit credentials or game JSON from live sessions.

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

Avoid `castle_builder`, `castle_rush`, and `phase_switch` remotely — build
actions (`pass=2`) are silently ignored on generals.io.

---

## Running

Private lobby (safe first test — a human joins the same lobby id):

```bash
python scripts/remote_play.py --bot army_convey --mode lobby
```

Public 1v1 queue (only after lobby test passes):

```bash
python scripts/remote_play.py --bot army_convey --mode 1v1 --max-games 5
```

Run offline checks before connecting:

```bash
python scripts/remote_play.py --bot fog_scout --mode lobby --verify-offline
```

---

## Game logs

Each finished game writes one JSON file under `data/remote_games/` (gitignored).
Fields match the table in [`remote-eval-heuristics.md`](remote-eval-heuristics.md)
§5.

Remote results do **not** update arena Elo in `data/ratings/`.

---

## Blockers checklist

1. **`GENERALS_USER_ID` missing** — script exits with setup instructions.
2. **No network** — live modes need outbound access to `botws.generals.io`.
3. **Build-economy bot selected** — every build tick becomes a silent pass.
4. **95% vs humans** — requires 100+ logged games against humans; do not claim
   that target without that sample.

---

## Architecture

- `arena/remote_adapter.py` — `Observation` → stdio dataclass; wraps
  `bots/<name>/agent.py` as `generals.agents.Agent`.
- `scripts/remote_play.py` — CLI, credential checks, JSON logging.

The remote client calls `agent.act(observation)` with one argument; adapters
must accept `key=None`.
