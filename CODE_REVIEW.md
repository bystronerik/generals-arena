# Code review — generals-arena

Read-only review of branch `main` at **`34ec2bf`** ("Add parallel in-process
competition match workers for arena batches", 2026-07-31 17:56 +07), clean
working tree. The review began against the then-uncommitted working state of
this commit; after it landed, all findings were re-verified against the
committed tree. One finding from the first pass (eager `arena/__init__.py`
imports, S1) was **fixed by this commit itself** and is kept below marked
*resolved* for the record.

Scope: the repo's own code (`arena/`, `scripts/`, `bots/`, `tests/`). The
`competition-module/` and `client/` submodules are vendored dependencies and were
not reviewed line-by-line.

---

## 1. Executive summary

- **Confirmed runtime bug (high):** `python scripts/remote_play.py` crashes with
  `NameError` in its *default* mode (dry-run) — [remote_play.py:76-77](scripts/remote_play.py:76)
  calls `_default_username` / `_default_lobby_id`, but the imported names are
  `default_username` / `default_lobby_id`. Reproduced live during this review.
- **Heavy import coupling — RESOLVED in `34ec2bf`:** `arena/__init__.py`
  used to eagerly import every submodule, making even the pure-JSON
  `scripts/leaderboard.py` require JAX and `generals_client`. The commit
  emptied the package `__init__`; re-verified: `import arena.ratings` now loads
  only `arena.ratings` + `arena.store`, no JAX, no `generals_client`.
- **Twin match runners (high, duplication):** `arena/competition_match.py` and
  `arena/classic_match.py` are ~80 % the same file — duplicated env setup, agent
  spawning, winner mapping, and game loop — and have already drifted
  (`PYTHON` env var, stderr handling).
- **Twin record schemas (medium):** `GameRecord` (store.py) and
  `ClassicGameRecord` (classic_tournament.py) duplicate the whole
  serialization stack (`to_dict`/`from_dict`/`_optional_*`/save/load/path).
- **Twin measurement scripts (medium):** `scripts/measure_heuristics.py` and
  `scripts/measure_classic.py` duplicate `aggregate_stats`, `winner_bot_id`,
  report writing, and an inlined copy of `store.utc_now_iso` — and define two
  `bot_run_sh` functions with *different semantics* under the same name.
- **Two bot generations (medium):** 10 older bots re-declare `PASS`,
  `DIRECTIONS`, `_is_passable`, reserve logic, etc. locally, while the 5
  migrated bots use `bots/_common/`. The shared library exists; half the fleet
  ignores it.
- **Dead code after the in-process migration (medium):** the subprocess path in
  `run_match.py` (`run_matchup`, `parse_matchup_output`, `parse_castles_built`,
  `MATCHUP_PY`, the `[matchup]` regexes) is no longer called by any production
  code — only by its own tests.
- **The core data layer is in good shape:** `store.py`, `ratings.py`,
  `parallel.py`, and `remote_report.py` are clean, well-typed, well-docstringed,
  and covered by a fast test suite (155 tests, ~1 s collection; `34ec2bf` added
  [tests/test_parallel.py](tests/test_parallel.py) for the job-cap helpers).
  Docs under `docs/` are unusually good and match the code.

---

## 2. Repo map with module roles

```text
generals-arena/
├── arena/                     Python package: match running, storage, ratings, remote play
│   ├── store.py               GameRecord schema + JSON IO for data/games/          [core, clean]
│   ├── ratings.py             Elo (elote) book + leaderboard writer               [core, clean]
│   ├── run_match.py           one competition match → stored record; also holds
│   │                          telemetry parsing AND a now-dead subprocess runner  [mixed]
│   ├── competition_match.py   in-process competition game loop (JAX env + stdio bots)
│   ├── classic_match.py       in-process classic-approximate game loop (≈ copy of above)
│   ├── tournament.py          pair×seed grid runner (competition) + manifest
│   ├── tournament_worker.py   picklable ProcessPool worker (extracted from tournament.py)
│   ├── classic_tournament.py  pair×seed grid runner (classic) + own record schema + own pool
│   ├── parallel.py            physical-core caps, thread pinning, run_pool helper
│   ├── bot_api.py             UnifiedObservation/Action, strategy loader, StrategySession
│   ├── remote_adapter.py      legacy competition-module Agent wrapper + offline verify
│   ├── remote_bridge.py       generals_client BaseBot/GameClient wrappers
│   ├── remote_client.py       FidelityRemoteSession, per-game JSON logging, session loops
│   ├── remote_env.py          dotenv load, credentials, server URL resolution
│   ├── remote_block.py        human-block game counting (95/100 criterion)
│   └── remote_report.py       remote log aggregation → markdown report
├── scripts/                   CLIs; mostly thin wrappers over arena/
│   ├── smoke_match.py, tournament.py, leaderboard.py, remote_report.py   [thin — good]
│   ├── measure_heuristics.py  measurement grid + report writer            [fat — logic lives here]
│   ├── measure_classic.py     classic grid + report writer                [fat — near-copy of above]
│   ├── remote_play.py         live generals.io session CLI                [has the NameError bug]
│   └── remote_lobby_watch.py  poll .env.agent for lobby id, then play
├── bots/                      stdio bots, one dir each: agent.py + main.py + run.sh
│   ├── _common/               wire loop, tactics, strategy helpers, oppmodel, cm adapter
│   ├── <old gen: smoke, expand_plus, classic_duel, army_convey, …>  self-contained agents
│   ├── <new gen: blitz, boom, metro, aegis, proteus>                use _common heavily
│   └── cm_*/                  wrappers over competition-module JAX agents
├── tests/                     20 files, 155 tests, core-focused, fast          [good]
├── docs/                      rules, engine, per-bot, research notes           [good, current]
├── data/                      games / classic_games / ratings / remote_games (gitignored)
├── competition-module/, client/   git submodules (vendored)
├── .cursor/skills/            agent workflow skills (process, not runtime code)
└── prompts/, RULES.md, AGENTS.md, README.md
```

**Runtime wiring.** Local competition path: `scripts/*` → `arena.run_match.run_and_store`
/ `arena.tournament.run_tournament` → `arena.competition_match` (in-process JAX env
driving `bots/<name>/run.sh` over stdio) → `arena.store` → `arena.ratings`.
Classic path mirrors it via `classic_match`/`classic_tournament` into
`data/classic_games/` (never Elo — correctly enforced). Remote path:
`scripts/remote_play.py` → `remote_bridge`/`remote_client` (generals_client
websocket) → per-game JSON in `data/remote_games/` → `remote_report`.
Bots talk only the stdio line protocol (`bots/_common/wire.py`), or the unified
in-process API (`arena/bot_api.py`) for remote play.

**Exists but unused / stale (Phase 1 orientation findings):**

| Item | Location | Status |
| --- | --- | --- |
| Subprocess matchup runner | [run_match.py:200-257](arena/run_match.py:200) (`run_matchup`), [:30](arena/run_match.py:30) (`MATCHUP_PY`), [:32-39](arena/run_match.py:32) (regexes), [:68-74](arena/run_match.py:68) (`parse_castles_built`), [:178-197](arena/run_match.py:178) (`parse_matchup_output`) | No production callers since `run_and_store` switched to `run_competition_match`; only [tests/test_run_match_parsers.py](tests/test_run_match_parsers.py) exercises the parsers |
| `iter_winners` | [store.py:254](arena/store.py:254) | Never called |
| `validate_record_dict` | [store.py:249](arena/store.py:249) | Never called (thin wrapper over `from_dict`) |
| Back-compat aliases | [remote_adapter.py:44-47](arena/remote_adapter.py:44) — of the four, only `list_remote_bots` has a caller; [bot_api.py:54](arena/bot_api.py:54) `StdioObservation` alias unused | Prune to the one used name |
| `resolve_server_url(public_server=…)` | [remote_env.py:72-82](arena/remote_env.py:72) | Both branches return `PUBLIC_SERVER_URL`; the flag (and both `--public-server` CLI options) is a no-op |
| `LobbyWatchSession.counted_at_start` | [remote_lobby_watch.py:99](scripts/remote_lobby_watch.py:99) | Assigned (always 0 by construction — mtime ≥ now) and never read |
| `started_at` parameter | [remote_lobby_watch.py:76-79](scripts/remote_lobby_watch.py:76) `_lobby_cleared_while_running` | Parameter ignored |
| ~~Stale package index~~ | [arena/\_\_init\_\_.py](arena/__init__.py) | **Resolved in `34ec2bf`** — the module list was removed along with the eager imports (see S1) |

---

## 3. Findings by phase

### Phase 2 — Structure

**S1 (high) — RESOLVED in `34ec2bf`: `arena/__init__.py` no longer imports submodules.**
During the first review pass, the package `__init__` eagerly imported all 13
submodules, so importing *any* `arena.*` module transitively required JAX
(top-level `import jax.numpy` in
[competition_match.py:19](arena/competition_match.py:19) /
[classic_match.py:21](arena/classic_match.py:21)) and `generals_client`
([remote_bridge.py:9-11](arena/remote_bridge.py:9)) — both "optional" per the
README. Commit `34ec2bf` replaced the body with a docstring directing callers
to import submodules directly. Re-verified after the commit:
`import arena.ratings, arena.store` loads exactly those two modules — no JAX,
no `generals_client`. The same commit also made
[measure\_heuristics.run\_one](scripts/measure_heuristics.py:186) import
`run_and_store` lazily, consistent with the fix. No further action needed.

**S2 (medium) — `run_match.py` is three modules in one.**
It holds (a) the dead subprocess runner (see Phase 1 table), (b) telemetry
parsing/record enrichment ([BotTelemetry](arena/run_match.py:49),
[parse_bot_telemetry:90](arena/run_match.py:90),
[apply_telemetry_to_record:113](arena/run_match.py:113)), and (c) the
run-and-store orchestration + CLI. The layering shows: the ProcessPool worker
[tournament_worker.py:26](arena/tournament_worker.py:26) has to import parsing
helpers *from a CLI module*. **Fix:** move telemetry parsing to a new
`arena/telemetry.py`; delete or quarantine the subprocess path. Effort: small.

**S3 (medium) — measurement/report logic lives in `scripts/`, unlike everything else.**
The repo's convention is thin CLIs over `arena/` (`scripts/tournament.py` is 16
lines; remote reporting correctly splits into [arena/remote_report.py](arena/remote_report.py)
+ a 56-line CLI). But [measure_heuristics.py](scripts/measure_heuristics.py) (463 lines)
and [measure_classic.py](scripts/measure_classic.py) (305 lines) keep
aggregation and markdown/JSON report generation in the script layer, untestable
via the `arena` package and duplicated between the two (see D8–D11).
**Fix:** extract an `arena/reporting.py` (stats aggregation + winrate-table
rendering) used by both. Effort: medium.

**S4 (medium) — the classic pipeline forked the competition pipeline instead of parameterizing it.**
`classic_match.py` vs `competition_match.py`, and `classic_tournament.py` vs
`tournament.py` + `parallel.py`, are parallel implementations of the same
concepts (detail in Phase 3). Notably [classic_tournament.py:280-291](arena/classic_tournament.py:280)
hand-rolls a `ProcessPoolExecutor` and does **not** use
`parallel.worker_initializer`, so classic workers run without the BLAS/XLA
thread pinning that competition workers get — a real behavioral inconsistency,
not just style. **Fix:** route classic runs through `run_pool` and a shared
match-loop core (see D1–D5). Effort: medium.

**S5 (low) — 12 copies of the `sys.path` bootstrap.**
Every script *and* four arena modules carry
`_REPO_ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(...)`
(some named `REPO_ROOT`, some `_REPO_ROOT`). Inside the package it exists only
to support `python arena/run_match.py` direct execution. **Fix:** document
`python -m arena.run_match` / `python -m arena.tournament` as the invocation and
keep the bootstrap only in `scripts/` (or a single `scripts/_bootstrap.py`).
Effort: small; mostly deletion.

**S6 (info) — bots layout is fine.**
`bots/<name>/{agent.py,main.py,run.sh}` with `_common/` for shared code is the
right shape for submission-style stdio bots, and keeping bots free of `arena`
imports is a defensible isolation choice (the `Observation` duplication it
causes is noted as D13). `proteus` importing sibling cores works because
`bots/` is on `sys.path` in both stdio and in-process modes — documented in its
docstring. No change recommended to the directory scheme.

**Target structure** (changes only; everything else stays):

```text
arena/
├── __init__.py            # emptied (S1) — done in 34ec2bf
├── telemetry.py           # ← BotTelemetry + parsers from run_match (S2)   [small / med]
├── match_loop.py          # ← shared spawn/loop/teardown core of
│                          #   competition_match + classic_match (D1–D5)    [med / high]
├── records.py (optional)  # ← shared record base or helpers for
│                          #   GameRecord/ClassicGameRecord (D6–D7)         [med / med]
├── reporting.py           # ← aggregate_stats, winrate tables from
│                          #   measure_heuristics/measure_classic (S3, D8–D11) [med / med]
└── run_match.py           # slims to run_and_store + CLI; dead subprocess
                           #   path deleted                                  [small / med]
scripts/
├── measure_heuristics.py  # grid definition + CLI only
└── measure_classic.py     # grid definition + CLI only
```

### Phase 3 — Duplication

Runner-level (the big cluster — all between
[competition_match.py](arena/competition_match.py) and
[classic_match.py](arena/classic_match.py)):

- **D1 (high)** `_venv_path_env`: three copies —
  [run_match.py:168-175](arena/run_match.py:168),
  [competition_match.py:55-61](arena/competition_match.py:55),
  [classic_match.py:61-66](arena/classic_match.py:61). Already drifted: the
  classic copy doesn't export `PYTHON`, so classic bots resolve `python3` from
  `PATH` while competition bots get the exact venv interpreter. One function in
  a shared module (e.g. `arena/match_loop.py`).
- **D2 (high)** `_spawn_agent`:
  [competition_match.py:64-91](arena/competition_match.py:64) vs
  [classic_match.py:69-94](arena/classic_match.py:69). Same body; drift is the
  stderr handling (`PIPE` for telemetry vs passthrough) and the log tag — both
  expressible as parameters.
- **D3 (high)** the game loop itself:
  [competition_match.py:141-197](arena/competition_match.py:141) vs
  [classic_match.py:131-170](arena/classic_match.py:131) — build agents, make
  env/board, spawn, `while turn < env.truncation: get_obs ×2 → ask_agent ×2 →
  transition → check done`, teardown in `finally`. The competition version adds
  timeout + castle counting + stderr drain. A single
  `run_stdio_match(env, seed, a, b, *, timeout, count_castles, capture_stderr)`
  covers both; `classic_match`/`competition_match` become thin config wrappers.
- **D4 (medium)** `competition_winner_seat`
  ([competition_match.py:103-110](arena/competition_match.py:103)) and
  `classic_winner_seat` ([classic_match.py:97-105](arena/classic_match.py:97))
  are character-identical. Keep one `winner_seat`.
- **D5 (low)** `Winner = Literal["a", "b", "draw"]` defined three times:
  [store.py:28](arena/store.py:28),
  [competition_match.py:39](arena/competition_match.py:39),
  [classic_match.py:19](arena/classic_match.py:19). Import from `store` (or the
  new shared module).

Record/store level:

- **D6 (medium)** `GameRecord` construction block duplicated:
  [run_match.py:288-307](arena/run_match.py:288) vs
  [tournament_worker.py:48-71](arena/tournament_worker.py:48) — same 20 lines
  including the magic literal `schema_version=2` (also hard-coded in both;
  should be a `CURRENT_SCHEMA_VERSION` constant in `store.py`). A
  `record_from_match_result(...)` helper in `store.py` or `telemetry.py`
  removes both copies.
- **D7 (medium)** `ClassicGameRecord` ([classic_tournament.py:38-148](arena/classic_tournament.py:38))
  re-implements the entire `GameRecord` machinery from
  [store.py:59-184](arena/store.py:59): `to_dict`, `from_dict` validation,
  `_optional_int`/`_optional_float` (verbatim copies at
  [classic_tournament.py:101-110](arena/classic_tournament.py:101) vs
  [store.py:131-140](arena/store.py:131)), `*_game_path`, `save_*`, `load_*`.
  Cheapest fix: import `_optional_*` + share save/load via a small generic
  helper; fuller fix: one record type with a `mode` discriminator and an extras
  dict.

Grid/measurement level:

- **D8 (medium)** `aggregate_stats`:
  [measure_heuristics.py:201-234](scripts/measure_heuristics.py:201) vs
  [measure_classic.py:52-85](scripts/measure_classic.py:52) — the same
  per-bot W/L/D/winrate/mean-turns aggregation modulo the record type. Move to
  `arena/reporting.py` keyed on `(bot_a, bot_b, winner, turns)`.
- **D9 (low)** `winner_bot_id`:
  [measure_heuristics.py:159-164](scripts/measure_heuristics.py:159) vs
  [measure_classic.py:44-49](scripts/measure_classic.py:44).
- **D10 (medium)** `bot_run_sh` — two functions, same name, **different
  semantics**: [measure_heuristics.py:89-92](scripts/measure_heuristics.py:89)
  (name → `bots/<name>/run.sh`, special-cases `expander_python`) vs
  [measure_classic.py:35-41](scripts/measure_classic.py:35) (accepts names,
  dirs, or `.sh` paths). This is a trap for anyone extending either script.
  One resolver in `arena` with the union behavior.
- **D11 (medium)** report writing: the winrate markdown table and JSON payload
  scaffolding in [measure_heuristics.py:296-403](scripts/measure_heuristics.py:296)
  vs [measure_classic.py:112-201](scripts/measure_classic.py:112); plus a third
  copy of the same `| Rank | Bot | Elo |` table in
  [ratings.py:200-217](arena/ratings.py:200) vs
  [measure_heuristics.py:283-293](scripts/measure_heuristics.py:283)
  (`round_leaderboard_snippet`). Share the row-rendering helpers.
- **D12 (low)** seat-swap logic exists twice with two shapes:
  `swap_sides` mirroring in [tournament.py:176-178](arena/tournament.py:176) /
  [classic_tournament.py:121-124](arena/classic_tournament.py:121) vs the
  legacy `both_seat_orders` in
  [measure_heuristics.py:116-127](scripts/measure_heuristics.py:116).
  The legacy grid is flagged as legacy; fine to leave until it's deleted.

Duplicated knowledge (not just code):

- **D13 (medium)** the observation shape exists twice:
  [bots/\_common/wire.py:14-25](bots/_common/wire.py:14) `Observation` and
  [arena/bot_api.py:37-51](arena/bot_api.py:37) `UnifiedObservation` — same 10
  fields. The isolation motive (bots must not import `arena`) is legitimate,
  but then the dependency should point the other way: `bot_api` importing the
  dataclass from `bots/_common/wire.py` (arena already puts `bots/` on
  `sys.path`), leaving exactly one definition.
- **D14 (medium)** `PASS`/`DIRECTIONS`/`_is_passable` re-declared in ~10
  old-generation agents (e.g. [classic_duel/agent.py:11-12](bots/classic_duel/agent.py:11),
  [army_convey/agent.py:8-9](bots/army_convey/agent.py:8), splitter, garrison,
  fog_scout, choke_control, castle_rush, late_rush, phase_switch, smoke) while
  [strategy_common.py:7-8](bots/_common/strategy_common.py:7) and
  [bot_api.py:31-34](arena/bot_api.py:31) also define them. `smoke` staying
  dependency-free is fine (it's the protocol canary); the other nine should
  import from `_common`. Caveat: these bots are behavior-pinned by measurement
  rounds — do it as a mechanical constant-swap and re-run the verification gate.
- **D15 (low)** `DEATHTOUCH_TURN = 800` appears in 12 bot files plus the
  [StrategyConfig default](bots/_common/strategy_common.py:100). It's a rules
  constant (RULES.md), not a per-bot tunable — one definition in `_common`.
- **D16 (low)** cell-type codes (0 fog, 1 plain, 2 mountain, 3 castle,
  4 general, 5 structure-in-fog) and owner codes (0/1/2) are bare ints across
  [bot_api.py:139-168](arena/bot_api.py:139), [:222-260](arena/bot_api.py:222),
  wire, tactics, and every agent. An `IntEnum`-style constants block in
  `bots/_common` (re-exported by `bot_api`) would make the mapping code
  self-documenting; grepping shows the encodings agree today.
- **D17 (low)** git-HEAD lookup twice:
  [store.git\_commit\_or\_tag:230-246](arena/store.py:230) (short SHA,
  "unknown" fallback) vs [remote\_client.git\_head:78-88](arena/remote_client.py:78)
  (full SHA, `None` fallback). One helper with a `short=` flag.
- **D18 (low)** dotenv parsing twice:
  [remote\_env.load\_dotenv\_files:24-38](arena/remote_env.py:24) vs
  [remote\_lobby\_watch.\_parse\_env\_file/\_apply\_env\_file:43-69](scripts/remote_lobby_watch.py:43).
  The watcher legitimately needs "parse without applying" — so `remote_env`
  should expose `parse_env_file()` and both call sites use it.
- **D19 (info)** the `[telemetry]` line format is produced by
  [wire.\_telemetry\_line:51-64](bots/_common/wire.py:51) and parsed by
  regexes in [run\_match.py:41-46](arena/run_match.py:41). Two sides of one
  wire format is acceptable, but neither file mentions the other; add
  cross-referencing comments (or share the field list) so they can't drift
  silently.
- **D20 (low)** [remote\_adapter.StdioStrategyAdapter:64-113](arena/remote_adapter.py:64)
  shadow-copies all nine `StrategySession` stat fields into itself
  (`_sync_from_session`) purely to mirror state that `self.session` already
  holds. Delegating reads (properties or just using `self.session.…`) deletes
  ~40 lines and a consistency hazard.

### Phase 4 — Unification & patterns

**U1 (medium) — one bug-proven pattern gap: scripts reimplement helpers they already import.**
The `remote_play.py` NameError (Phase 1) is the symptom: the dry-run block
re-derived defaults under private names instead of calling the imported
`default_username`/`default_lobby_id`. Canonical pattern: scripts never define
`_private` fallbacks for things `arena.remote_env` exports.

```python
# before (scripts/remote_play.py:76) — crashes
registered_as = ensure_bot_username(username or _default_username(bot))
# after
registered_as = ensure_bot_username(username or default_username(bot))
```

**U2 (medium) — CLI error-exit conventions are mixed.**
Same validation, three behaviors: [tournament.py:316-321](arena/tournament.py:316)
returns 2 for bad `--jobs`; [classic_tournament.py:352-354](arena/classic_tournament.py:352)
returns 1 for the same check; [measure_heuristics.py](scripts/measure_heuristics.py:525)
returns 1 for missing bots but 2 for bad args. Canonical: use
`parser.error(...)` for argument validation (uniform exit 2, usage printed),
reserve return 1 for runtime failures.

```python
# before
if args.jobs < 1:
    print("[classic_tournament] --jobs must be >= 1", file=sys.stderr)
    return 1
# after
if args.jobs < 1:
    parser.error("--jobs must be >= 1")
```

**U3 (medium) — private-attribute access across boundaries.**
[remote\_play.py:306-307](scripts/remote_play.py:306) and
[remote\_client.py:298-299](arena/remote_client.py:298), [:351-352](arena/remote_client.py:351)
read `session._score_wins` / `_score_losses` from outside the class. Add
`wins`/`losses` properties on `FidelityRemoteSession` and use them everywhere.

**U4 (low) — two timestamp formats in stored records.**
Arena games use [store.utc\_now\_iso:143](arena/store.py:143)
(second precision, `Z` suffix); remote game logs use raw
`datetime.now(timezone.utc).isoformat()` ([remote\_client.py:238](arena/remote_client.py:238),
microseconds, `+00:00`) — and measure scripts inline a third copy of the
`utc_now_iso` body ([measure\_heuristics.py:306](scripts/measure_heuristics.py:306),
[measure\_classic.py:122](scripts/measure_classic.py:122)). Canonical:
`store.utc_now_iso()` everywhere a timestamp is persisted.

**U5 (low) — logging style.**
Local pipeline modules print with a `[module]` prefix (good, consistent);
remote modules print bare strings; [bot\_api.py:23](arena/bot_api.py:23) is the
only `logging` user. For a research CLI repo, prefix-printing is a fine
canonical choice — bring the remote modules in line (`[remote_play] …`) and
either drop the lone logger or route it to stderr explicitly; today a
`StrategySession` fault is invisible unless the host app configured logging.

**U6 (low) — naming and idiom nits.**
`REPO_ROOT` vs `_REPO_ROOT` for the same bootstrap;
`git_head` vs `git_commit_or_tag` (D17); `SessionSummary` field `bot_name`
serialized as key `"bot_id"` ([remote\_client.py:57-58](arena/remote_client.py:57))
while every other record uses `bot_id`; the
`__import__("pathlib")` one-liner at [remote\_adapter.py:31](arena/remote_adapter.py:31)
instead of a normal import. Each is a two-minute fix best bundled with
neighboring work.

**U7 (info) — where consistency is already good.**
Type hints: essentially 100 % of `arena/` and `scripts/` functions carry return
annotations (measured via AST) — genuinely consistent. Docstrings: present on
all public modules and most public functions. Dependency style: the
`param: Path | None = None` + module-level default (`GAMES_DIR`, `RATINGS_DIR`,
`REMOTE_GAMES_DIR`) injection pattern is applied uniformly and makes the test
suite's tmp-dir usage clean. The old-gen/new-gen bot split (untyped vs typed)
is the only real divergence, and it tracks bot generations rather than
randomness. No action needed beyond D14.

---

## 4. Prioritized action list

Quick wins (minutes each):

1. **Fix the `remote_play.py` NameError** — rename the two call sites
   ([remote_play.py:76-77](scripts/remote_play.py:76)) to the imported names.
   The default CLI invocation is currently broken (re-confirmed at `34ec2bf`).
   *(high)*
2. ~~Empty out `arena/__init__.py`~~ — **done in `34ec2bf`** and verified;
   no follow-up needed.
3. Delete dead code: `run_matchup` + `MATCHUP_PY` + `parse_castles_built`
   (keep `parse_matchup_output` only if you want the documented manual
   `matchup.py` flow verifiable — otherwise drop its tests with it),
   `iter_winners`, `validate_record_dict`, unused aliases in
   `remote_adapter`/`bot_api`, `counted_at_start`, the ignored `started_at`
   param, and the no-op `public_server` plumbing (or make the flag meaningful).
   *(medium)*
4. Hoist `schema_version=2` into `store.CURRENT_SCHEMA_VERSION`; use
   `store.utc_now_iso()` in both measure scripts; single `Winner` type;
   single `winner_seat()`. *(low–medium)*
5. `FidelityRemoteSession.wins/.losses` properties; `parser.error` for arg
   validation across the five CLIs; refresh the `arena/__init__` module list if
   you keep it. *(low)*

Structural refactors (ordered by value):

6. **Unify the match loop** (D1–D3, S4): extract
   `arena/match_loop.py` with shared `_venv_path_env`, `_spawn_agent`, and a
   parameterized run loop; make `competition_match` / `classic_match` thin
   wrappers. This also fixes the classic-side thread-pinning gap by routing
   `classic_tournament` through `parallel.run_pool`. ~½ day; the
   `pytest -m slow` harness tests plus one `--mode competition` verification
   match per AGENTS.md gate it. *(effort M / benefit H)*
7. **Extract `arena/reporting.py`** (S3, D8–D11): aggregation + table
   rendering shared by both measure scripts and `ratings.write_leaderboard`;
   collapse the two `bot_run_sh` resolvers into one. ~½ day. *(M / M)*
8. **Extract `arena/telemetry.py`** from `run_match.py` (S2, D6) and add
   `record_from_match_result(...)` used by both `run_and_store` and
   `tournament_worker`. A couple of hours. *(S / M)*
9. **De-duplicate the record layer** (D7): at minimum share
   `_optional_*` and save/load helpers between `GameRecord` and
   `ClassicGameRecord`; full unification only if classic records grow. *(M / M)*
10. **Point `bot_api` at the wire `Observation`** (D13) and move
    `PASS`/`DIRECTIONS`/`DEATHTOUCH_TURN`/cell-code constants into `_common`
    (D14–D16), then mechanically migrate the nine old-gen bots to import them.
    Do this last, one bot per commit, re-running the competition verification
    match each time — these bots are measurement-pinned. ~1 day spread out.
    *(M / M, churn-sensitive)*
11. Trim the 12 `sys.path` bootstraps to the script layer and standardize on
    `python -m arena.…` for package modules (S5). *(S / S)*

Not recommended: restructuring `bots/` or splitting `arena/` into subpackages.
The current flat layout is legible at this size; the wins above come from
merging parallel implementations, not from moving files.
