# Remote evaluation of heuristic bots (classic generals.io)

How to play our heuristic bots against real humans on **live generals.io**,
what that does and does not tell us, and what to log.

Read [`remote-generalsio.md`](remote-generalsio.md) first for the short
version. This page is the evaluation plan.

> **Remote play is a different game.** `competition-module/generals/remote/`
> connects to live generals.io. It is not the Generals Competition sandbox and
> it does not use the competition ruleset. A remote result is evidence about
> general play quality, never evidence about competition standing. Never feed
> remote games into the arena Elo book in `data/ratings/`.

---

## 1. Two different agent interfaces

The repo has two unrelated agent shapes. This is the main integration cost.

| | Competition (local arena) | Remote (live generals.io) |
| --- | --- | --- |
| Entry point | `bots/<name>/run.sh`, a subprocess | in-process Python object |
| Contract | line protocol in `competition-module/competition/protocol.py` | `generals.agents.Agent` subclass |
| Method | `Agent.act(obs)` on the plain dataclass in `bots/<name>/main.py` | `Agent.act(observation, key)` on `generals.core.observation.Observation` |
| Observation type | lists of ints parsed from stdin | JAX/NumPy arrays, `NamedTuple` |
| Driver | `competition/matchup.py` | `generals.remote.generalsio_client` |

Two verified details that will bite an implementer:

1. **The remote client calls `act` with one argument.**
   `GeneralsIOClient._generate_action` calls `self.agent.act(observation)`,
   while the `Agent` abstract base declares `act(self, observation, key)`. Any
   adapter must therefore define `act(self, observation, key=None)`.
2. **Build actions cannot be sent remotely.** `_generate_action` only emits an
   attack when `action[0]` is falsy. A build (`pass = 2`) is truthy, so it
   returns `None` and the bot silently does nothing that tick. A castle bot
   does not degrade gracefully remotely — it wastes every build turn.

---

## 2. Rule mismatch: competition vs classic

| Rule | Competition (`mode="competition"`) | Live generals.io | Effect on our bots |
| --- | --- | --- | --- |
| Castles | none pre-placed; players **build** (`pass=2`) | neutral **cities** exist with a garrison to capture; no build action | `castle_builder`, `castle_rush` lose their whole economy plan; build ticks become no-ops |
| Deathtouch | any move onto the enemy general wins from turn 800 | never; a general capture always needs more army | the deathtouch branch in `general_hunter` / `phase_switch` is dead remotely |
| Turn cap | hard draw at 1200 | no cap of this kind; games end by elimination or surrender | phase clocks keyed to 800/1200 are meaningless |
| Map | rectangle, sides 18–21, ~25% mountains | varies by lobby and player count | fixed-size assumptions break |
| Players | strictly 1v1 | 1v1, FFA, and team modes | use 1v1 only; `GeneralsIOstate` assumes one opponent index |
| Turn counter | `turn` from the engine | `timestep = turn - 1`, and server ticks are half-turns | never compare turn numbers across the two |
| Terrain kinds | plain, mountain, castle, general | also swamps, deserts, lookouts, observatories in modern rooms | `generalsio_state.get_observation` only decodes `>= 0` (player), `-1` neutral, `-2` mountain, `-3` fog, `-4` structure-in-fog; anything else is misread |
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
| `smoke` | only as a sanity check | not competitive by design |
| `general_hunter` | partial | the pre-800 snipe path works; the deathtouch path never fires |
| `phase_switch` | partial | phase clock is calibrated to a 1200-turn game |
| `castle_builder`, `castle_rush` | **not viable** | build ticks become silent passes |

Start with `expand_plus`. It is the cleanest read on "is our expansion any
good against humans".

---

## 3. Adapter design

Implemented:

- `arena/remote_adapter.py` — observation translation and `StdioStrategyAdapter`.
- `scripts/remote_play.py` — CLI, credential checks, JSON logging to `data/remote_games/`.

Setup guide: [`remote-play-setup.md`](remote-play-setup.md).

### 3.1 Observation translation

The adapter builds the plain dataclass our bots expect from the remote
`NamedTuple`. Field mapping, all verified against
`competition-module/generals/core/observation.py` and
`competition-module/competition/protocol.py`:

| Our field | Built from | Rule |
| --- | --- | --- |
| `H`, `W` | `observation.armies.shape` | |
| `turn` | `int(observation.timestep)` | remote counts half-turns; see §2 |
| `my_land` | `int(observation.owned_land_count)` | |
| `my_army` | `int(observation.owned_army_count)` | |
| `opp_land` | `int(observation.opponent_land_count)` | |
| `opp_army` | `int(observation.opponent_army_count)` | |
| `owner_grid` | `owned_cells` → 1, `opponent_cells` → 2, else 0 | perspective is already baked in |
| `type_grid` | `fog_cells` → 0, `mountains` → 2, `castles` → 3, `generals` → 4, `structures_in_fog` → 5, else 1 | apply in that precedence order |
| `army_grid` | `observation.armies` as ints | 0 inside fog |

Two traps in `type_grid`:

- `observation.generals` is a mask of **all visible generals, including your
  own**. Our protocol also marks both, and `owner_grid` disambiguates, so this
  matches — but any code that looks for "the enemy general" must test
  `generals & opponent_cells`, not `generals` alone.
- Remote `castles` are classic **cities**, which can be neutral and garrisoned.
  Our bots read `type == 3` as "a castle someone built". A bot that assumes
  castles are never neutral will misjudge them remotely.

### 3.2 Action translation

Our bots return `(pass, r, c, dir, split)` as a Python tuple. The remote
client expects an indexable array of five ints, so return
`numpy.array([...], dtype=int)`.

Direction codes agree: our `DIRECTIONS = [(-1,0), (1,0), (0,-1), (0,1)]`
matches `DIRECTIONS = [Direction.UP, Direction.DOWN, Direction.LEFT,
Direction.RIGHT]` in the remote client, so `dir` passes through unchanged.

Build actions must be intercepted: rewrite `pass = 2` to `pass = 1` and
increment a `builds_dropped` counter for the run log. Do not send a 2 — it
silently produces no action.

### 3.3 Sketch

```python
# arena/remote_adapter.py  (not implemented yet)
class StdioStrategyAdapter(Agent):
    """Wrap a bots/<name>/agent.py strategy for remote in-process play."""

    def __init__(self, bot_dir: Path, bot_id: str):
        # import bots/<name>/agent.py by path; do not copy strategy code here
        self.strategy = None          # constructed on the first observation,
        self.builds_dropped = 0       # because H/W are unknown until then
        self.faults = 0

    def act(self, observation, key=None):   # key=None is required; see §1
        obs = to_stdio_observation(observation, self.player_id)
        if self.strategy is None:
            self.strategy = Agent(player_id=self.player_id, H=obs.H, W=obs.W)
        action = self.strategy.act(obs)
        if action[0] == 2:                  # build: unsupported remotely
            self.builds_dropped += 1
            action = (1, 0, 0, 0, 0)
        return np.array(action, dtype=int)

    def reset(self):
        self.strategy = None
```

`player_id` is not on the remote observation — the perspective is already
applied, so pass `0` and rely on `owner_grid` values (1 = me, 2 = opponent),
exactly as the stdio bots already do.

### 3.4 Offline verification (required before any live run)

The adapter cannot be tested against the server in CI, and the sandbox has no
network. It **can** be verified offline, and must be:

1. Build a synthetic `generals.core.observation.Observation` by hand, run
   `to_stdio_observation`, and assert every grid and scalar.
2. Take one observation from a local `--mode competition` match, round-trip it
   through the adapter, and assert the strategy returns the identical action it
   returned over stdio. Same input, same decision.
3. Assert that a `pass = 2` action is rewritten and counted.

---

## 4. Running it (no credentials in the repo)

`user_id` is a secret you invent — any long random string. It is not issued by
generals.io; `register_agent(username)` binds a username to it once. Treat it
like a password.

```bash
source .venv/bin/activate
pip install -e competition-module      # brings python-socketio[client]>=5.11.4

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
- The client relevant calls are `register_agent(username)` once,
  then `join_private_lobby(lobby_id)` or `join_1v1_queue()`, driven by
  `autopilot(agent, user_id, lobby_id)` for indefinite play.
- Use the bot endpoint (`https://botws.generals.io/`, the client default),
  not the human endpoint.
- Follow the generals.io bot convention of a `[Bot]` username prefix so
  opponents know what they are playing. Do not farm the public ladder
  unattended; run bounded sessions.

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

Session-level: wall-clock duration, games played, and any adapter exception
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
- The adapter is a fixed cost that pays off once, so pay it when there is
  something worth testing.
- Bots must first pass the local gate (`--mode competition` match finishes)
  and the fault check.

Suggested order: fix the local pool per
[`optimize-existing.md`](../research/strategies/optimize-existing.md) → run the
tournament in [`tournament-plan.md`](../research/strategies/tournament-plan.md)
→ implement and offline-verify the adapter → private lobby test →
bounded public 1v1 session with `expand_plus`.

## 7. Related

- [`remote-generalsio.md`](remote-generalsio.md) — what the remote module is
- [`local-matchup.md`](local-matchup.md) — the competition path
- [`../competition/protocol.md`](../competition/protocol.md) — stdio frames
- [`../../RULES.md`](../../RULES.md) — competition ruleset
