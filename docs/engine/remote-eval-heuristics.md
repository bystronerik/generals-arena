# Remote evaluation of heuristic bots (classic generals.io)

How to play our heuristic bots against real humans on **live generals.io**,
what that does and does not tell us, and what to log.

Read [`remote-generalsio.md`](remote-generalsio.md) first for the short
version. This page is the evaluation plan.

> **Remote play is a different game.** Live generals.io uses the
> `generals_client` wire in `client/` (EIO v4, no `bot_key`). It is not the
> Generals Competition sandbox and it does not use the competition ruleset. A
> remote result is evidence about general play quality, never evidence about
> competition standing. Never feed remote games into the arena Elo book in
> `data/ratings/`.

---

## 1. Unified bot API

Strategy code lives in `bots/<name>/agent.py` and uses one observation and
action shape for both local competition and live generals.io. Wire details stay
in the arena bridges — bots never import `competition-module` or
`generals_client`.

Full layout, types, and bridge table:
[`unified-bot-api.md`](unified-bot-api.md).

| Path | Role |
| --- | --- |
| `arena/bot_api.py` | `UnifiedObservation`, `UnifiedAction`, mappers |
| `bots/<name>/main.py` | Stdio bridge for `--mode competition` |
| `arena/remote_bridge.py` | `UnifiedBot` over `generals_client` for live play |
| `scripts/remote_play.py` | CLI, credentials, JSON logging |

Setup and CLI flags: [`remote-play-setup.md`](remote-play-setup.md).

One remote detail that still matters for evaluation: **build actions
(`pass=2`) are rewritten to pass on live generals.io.** Castle-economy bots
waste every build tick remotely. See §2.

---

## 2. Rule mismatch: competition vs classic

| Rule | Competition (`mode="competition"`) | Live generals.io | Effect on our bots |
| --- | --- | --- | --- |
| Castles | none pre-placed; players **build** (`pass=2`) | neutral **cities** exist with a garrison to capture; no build action | `castle_builder`, `castle_rush` lose their whole economy plan; build ticks become no-ops |
| Deathtouch | any move onto the enemy general wins from turn 800 | never; a general capture always needs more army | the deathtouch branch in `general_hunter` / `phase_switch` is dead remotely |
| Turn cap | hard draw at 1200 | no cap of this kind; games end by elimination or surrender | phase clocks keyed to 800/1200 are meaningless |
| Map | rectangle, sides 18–21, ~25% mountains | varies by lobby and player count | fixed-size assumptions break |
| Players | strictly 1v1 | 1v1, FFA, and team modes | use 1v1 only; remote state assumes one opponent index |
| Turn counter | `turn` from the engine | half-turn server ticks | never compare turn numbers across the two |
| Terrain kinds | plain, mountain, castle, general | also swamps, deserts, lookouts, observatories in modern rooms | `generals_client` decodes a subset; exotic tiles may misread |
| Fog | on, vision radius 1 | on, plus room variants | comparable, the one thing that transfers |

**What transfers:** expansion efficiency, army conveying, scouting, and
target selection. **What does not:** every rule-specific timing (800, 1200),
castle economy, and deathtouch tactics.

### Which bots are worth running remotely

| Bot | Remote viability | Note |
| --- | --- | --- |
| `expand_plus` | good | pure move logic, translates cleanly |
| `fog_scout` | good | scouting is rule-independent |
| `army_convey` | good | conveying is rule-independent |
| `classic_duel` | good | remote evaluation champion |
| `late_rush` | good | expansion + rush; deathtouch branch is dead on classic |
| `smoke` | only as a sanity check | not competitive by design |
| `general_hunter` | partial | the pre-800 snipe path works; the deathtouch path never fires |
| `phase_switch` | partial | phase clock is calibrated to a 1200-turn game |
| `castle_builder`, `castle_rush` | **not viable** | build ticks become silent passes |

Start with `expand_plus`. It is the cleanest read on "is our expansion any
good against humans".

---

## 3. Wire bridge (implemented)

- `arena/bot_api.py` — unified types and mappers (`from_generals_client_state`,
  `translate_action_for_remote`).
- `arena/remote_bridge.py` — `UnifiedBot` + `ArenaGameClient` over
  `generals_client`.
- `arena/remote_client.py` — `FidelityRemoteSession`, JSON logging to
  `data/remote_games/`.
- `scripts/remote_play.py` — CLI, credential checks, offline verify.

Legacy harness only: `arena/remote_adapter.py` (`StdioStrategyAdapter` for
in-process tests against competition-module `generals.agents.Agent`). Live play
uses `UnifiedBot`, not the adapter.

### 3.1 Observation translation

`arena/bot_api.from_generals_client_state` maps `generals_client.state.GameState`
into `UnifiedObservation`. Field mapping is shared with the stdio bridge in
`bots/<name>/main.py` — same grids and scalars our strategies already expect.

Two traps in `type_grid`:

- The general mask can include **your own general** when visible. Any code that
  looks for "the enemy general" must test owner, not the general mask alone.
- Remote `castles` are classic **cities**, which can be neutral and garrisoned.
  Our bots read `type == 3` as "a castle someone built". A bot that assumes
  castles are never neutral will misjudge them remotely.

### 3.2 Action translation

Strategies return `(pass, r, c, dir, split)`. `translate_action_for_remote`
rewrites `pass = 2` (build) to `pass = 1` and increments a `builds_dropped`
counter for the run log. Do not send a build to live generals.io — it silently
produces no action.

Direction codes agree across stdio, competition protocol, and
`generals_client`: index 0 = up, 1 = down, 2 = left, 3 = right.

### 3.3 Offline verification (required before any live run)

The bridge cannot be tested against the server in CI. It **can** be verified
offline, and must be:

```bash
python scripts/remote_play.py --mode dry-run --bot expand_plus
python scripts/remote_play.py --bot expand_plus --mode lobby --verify-offline
```

Checks: synthetic `GameState` round-trip, build rewrite and counter, and the
bot loads and acts without credentials.

---

## 4. Running it (no credentials in the repo)

`user_id` is a secret you invent — any long random string. It is not issued by
generals.io; `register_username` binds a username to it once. Treat it like a
password.

```bash
source .venv/bin/activate
pip install -e competition-module   # local competition matches
pip install -e client               # generals_client wire for live play
pip install -r requirements.txt

export GENERALS_USER_ID='<your long random secret>'
export GENERALS_USERNAME='[Bot] arena_army_convey'
export GENERALS_LOBBY_ID='arena-test'

# offline verify first (no credentials needed)
python scripts/remote_play.py --mode dry-run --bot army_convey

# private lobby, human joins the same lobby id (safe first test)
python scripts/remote_play.py --bot army_convey --mode lobby

# public 1v1 queue against real players (only after the lobby test passes)
python scripts/remote_play.py --bot army_convey --mode 1v1 --max-games 5
```

Rules for handling the secret:

- Read it from the environment only. Never a CLI flag (it lands in shell
  history) and never a file in the repo. `.env` and `.env.agent` are already
  in `.gitignore`; keep it there.
- Usernames **must** start with `[Bot]` for `generals_client`. See
  [`unified-bot-api.md`](unified-bot-api.md) § Username policy.
- `GENERALS_BOT_KEY` is **not** used. Remote play goes through
  `generals_client` on `botws.generals.io` (EIO v4).
- Follow the generals.io bot convention of a `[Bot]` username prefix so
  opponents know what they are playing. Do not farm the public ladder
  unattended; run bounded sessions.

Full env table and blockers: [`remote-play-setup.md`](remote-play-setup.md).

---

## 5. What to log, and where

Remote games must **not** go into `data/games/` or the Elo book — different
ruleset, and mixing them corrupts the competition leaderboard. Use a separate
store, `data/remote_games/<timestamp>_<bot>_<result>.json`, and a separate
summary table.

Per game:

| Field | Why |
| --- | --- |
| `replay_id` | the client exposes it; the only way to review the game |
| `bot_id`, `bot_commit_or_tag` | which version played |
| `room_mode` (`lobby` / `1v1`) | lobby games are not ladder-representative |
| `opponent_username`, `opponent_is_bot` | a human result and a bot result are different evidence |
| `result` (`win` / `loss` / `disconnect`) | disconnects are not losses |
| `server_turns` | half-turn count; never compare to competition turns |
| `peak_land`, `peak_army`, `final_land`, `final_army` | the expansion-quality signal |
| `saw_enemy_general_at` | turn of first sighting, or null — the scouting signal |
| `builds_dropped` | non-zero means a castle bot was run remotely by mistake |
| `faults`, `timeouts` | our bot failing to answer in time |
| `counts_toward_block`, `result_reason` | human-95 block eligibility (see fidelity session) |

Session-level: wall-clock duration, games played, and any bridge exception
with a traceback.

### Metrics worth reading

1. **Winrate versus humans** — the headline, but noisy: opponent strength on
   the public ladder varies enormously and is not recorded by rating.
2. **Land at turn N** (e.g. server turn 100) — the cleanest transferable
   measure of expansion efficiency, and directly comparable between our own
   bots.
3. **Scouting rate** — fraction of games where the enemy general was ever
   sighted. This transfers to competition play, where finding the general
   before turn 800 is the deciding skill.
4. **Fault and timeout rate** — the competition allows 150 ms per move and
   forfeits at 50 faults. A remote session is a cheap way to find a bot that
   is too slow on a large board.

Interpretation limits to state in any note: no opponent rating, no seed
control, no repeatability, and a different ruleset. Remote results can falsify
a claim ("this bot times out", "this bot never scouts") but cannot rank bots
for the competition. Ranking stays with the local arena.

---

## 6. Sequencing

Remote evaluation is worth doing **after** the local pool produces decisive
games, not before. Reasons:

- A bot that draws every local game has no measurable strategy to validate.
- The wire bridge is a fixed cost that pays off once, so pay it when there is
  something worth testing.
- Bots must first pass the local gate (`--mode competition` match finishes)
  and the fault check.

Suggested order: fix the local pool per
[`optimize-existing.md`](../research/strategies/optimize-existing.md) → run the
tournament in [`tournament-plan.md`](../research/strategies/tournament-plan.md)
→ offline-verify the bridge → private lobby test → bounded public 1v1 session
with `expand_plus`.

## 7. Related

- [`unified-bot-api.md`](unified-bot-api.md) — one API for stdio and remote
- [`remote-play-setup.md`](remote-play-setup.md) — credentials and CLI
- [`remote-generalsio.md`](remote-generalsio.md) — what the remote module is
- [`local-matchup.md`](local-matchup.md) — the competition path
- [`../competition/protocol.md`](../competition/protocol.md) — stdio frames
- [`../../RULES.md`](../../RULES.md) — competition ruleset
