# Per-turn recorder and telemetry aggregation plan

Give heuristic tuning per-turn evidence and a later RL bot trajectories to
train on, without touching the rating identity: a game's per-turn history is
recorded **beside** its `GameRecord`, keyed by `game_id`, off by default.

Status: **decided, 2026-08-01 — ready to implement.** All design questions
were resolved the same day (§8: four decided explicitly, four applied
defaults); no implementation yet. Modelled on
[ratings-refactor-plan.md](ratings-refactor-plan.md); like that document, every
audit claim below was verified against the repo on the stated date, and numbers
marked *(measured)* come from throwaway probes run on 2026-08-01.

Two constraints set after the first draft, and folded in throughout:

1. **No backwards compatibility, but keep previous round data.** The old
   telemetry machinery is deleted, not preserved alongside — there is no EOF
   `[telemetry]` line, no stderr parsing, and no old-metrics emulation in new
   code. The same rule applies to the record schema itself: v4 fields nothing
   reads are **dropped**, not carried (§2.4 — the field-consumer audit is
   A12). Since old rounds must be kept, the 10,501 stored records are
   migrated to the lean v5 by a one-shot projection script — no dual-version
   reader ever exists, exactly as v4 refused to read v3.
2. **Nothing recorder- or telemetry-related ships in a bundle.** The submitted
   bot is only the logic needed to play: `bots/_common/wire.py` becomes a pure
   protocol loop, and per-turn introspection moves out of `bots/` entirely,
   into arena-owned probe modules. The bundle-of-hash-X-is-the-program-rated-
   as-X invariant ([`arena/bundle.py:18-21`](../../arena/bundle.py:18)) is
   preserved exactly — both the bundle and the rated closure are the clean
   bot.

---

## 0. Summary

| | Today | Target |
| --- | --- | --- |
| Per-turn engine truth | held in `run_stdio_match`, discarded every turn | recorded as seed + action sequence + scalar series, replayable to full states |
| Per-turn bot internals | none — one stderr line at stdin EOF | arena-owned instrumented runner + per-bot probes under `arena/`, opt-in |
| Telemetry in the bundle | `wire.py` telemetry + `telemetry_extras()` ship in every zip | none — the closure and the bundle are pure game logic |
| Telemetry typing | ad-hoc `k=v` strings, one hardcoded bool coercion | declared schema (type + meaning + reducers); unknown keys fail loudly |
| Aggregation | last stderr frame only | engine-truth finals on every record; series reducers on recorded games |
| Hash blast radius of instrumentation edits | every `wire.py` edit forks all ~20 bot hashes | zero after a one-time cleanup fork — probes live outside every closure |
| Storage | — | `data/trajectories/<round>/<game_id>.*`, gitignored, opt-in per run |
| Record schema | v4: 15 required + 9 optional fields, several with zero readers | lean v5: identity + outcome required, observability in `metrics`; stored records migrated one-shot, no dual-version reader |
| Acceptance | — | replay(seed, actions) reproduces the recorded game, state-equal |
| Default cost | — | recording off: zero; on: ≤ 2% wall clock per match |

Non-goals (restated from the request): no model code, no training loops, no
checkpoint formats, no change to the rating model or the fit.

---

## 1. Current-state audit

### What exists

**A1 — Bots emit exactly one telemetry line, at stdin EOF.**
[`wire.py:80-85`](../../bots/_common/wire.py:80): the loop returns on EOF and
only then writes `_telemetry_line`
([`wire.py:51`](../../bots/_common/wire.py:51)) to stderr. Format
`[telemetry] player= turn= my_land= my_army= opp_land= opp_army= [k=v ...]`,
extras stringified with no schema at
[`wire.py:61-63`](../../bots/_common/wire.py:61). No per-turn history exists
anywhere in the repo.

**A2 — Correction: 9 bots define `telemetry_extras()`, not 11.**
*(measured — `grep -rn "def telemetry_extras" bots/`)*: `proteus`, `aegis`,
`blitz`, `metro`, `boom`, `expand_plus`, `castle_builder`, `general_hunter`,
and `classic_duel`. `classic_duel` is remote-only and its games never become
`GameRecord`s. Current key inventory across all nine, with observed types:

| Key | Bots | Value shape |
| --- | --- | --- |
| `enemy_general_sighted` | blitz, expand_plus, castle_builder, general_hunter, classic_duel | 0/1 |
| `first_sighting_turn` | the sighting bots, when sighted | int |
| `phase` | aegis, blitz, boom | token string |
| `was_attacked`, `countering` | aegis | 0/1 |
| `strikes` | blitz | int counter |
| `castles_built`, `pushing` | metro | int counter, 0/1 |
| `reason`, `guard` | boom | token string, int |
| `active`, `label`, `switches` | proteus | token, token, int counter |
| `first_city_capture_turn` | classic_duel | int (classic-only) |

**A3 — Aggregation keeps the last line and flattens extras stringly.**
[`telemetry.py:65-85`](../../arena/records/telemetry.py:65)
(`parse_bot_telemetry`) keeps the last line per player;
[`telemetry.py:112-118`](../../arena/records/telemetry.py:112) flattens extras
into `GameRecord.metrics` as `<key>_a` / `<key>_b`;
[`telemetry.py:56-62`](../../arena/records/telemetry.py:56)
(`_coerce_metric_value`) special-cases exactly one key
(`enemy_general_sighted → bool`), tries `int`, and silently stores anything
else as a string.

**A4 — The match loop already holds everything a recorder needs, and discards
it.** [`loop.py:183-203`](../../arena/matches/loop.py:183): each turn it has
both fog observations (`obs_0`, `obs_1`), both decoded actions (`a_0`, `a_1`),
the full fog-free `state` and `info` after `transition`, and castle births
([`loop.py:193-199`](../../arena/matches/loop.py:193)). It returns only
aggregates ([`loop.py:213-222`](../../arena/matches/loop.py:213)).
`GameState` carries `armies (H,W) int`, `ownership (2,H,W) bool`, plus
`castles`/`mountains`/`generals` masks
([`competition-module/generals/core/game.py:34-57`](../../competition-module/generals/core/game.py:34)).
In particular the engine can state the final land/army truth for both seats
without any help from the bots — today those record fields come from bot
stderr instead.

**A5 — Bot stderr goes to one temp file per seat, concatenated at match end.**
[`loop.py:157-163`](../../arena/matches/loop.py:157) and
[`loop.py:115-120`](../../arena/matches/loop.py:115) (`_read_temp_stderr`).
Bot debug output and telemetry share one channel, and the whole thing is held
in memory as `MatchLoopResult.stderr`.

**A6 — Store: flat JSON per game, schema v4.** *(measured)* **10,501 records /
42 MB** (≈4.2 KB average) under `data/games/`.
[`store.py:37`](../../arena/records/store.py:37) pins v4;
[`store.py:110`](../../arena/records/store.py:110) keeps `metrics` free-form;
[`store.py:270`](../../arena/records/store.py:270) (`save_game`) writes one
file per game — which is also what makes the process pool safe: each
[`worker.run_one_worker`](../../arena/tournaments/worker.py:32) writes only its
own game ([`worker.py:81`](../../arena/tournaments/worker.py:81)), no shared
writer, futures collected in
[`parallel.run_pool`](../../arena/tournaments/parallel.py:85).

**A7 — Local matches have no per-move clock.**
[`ask_agent`](../../competition-module/competition/matchup.py:79) is a blocking
`readline` with no deadline; the only clock is the whole-match `timeout` at
[`loop.py:177`](../../arena/matches/loop.py:177). The 150 ms budget exists only
on the judge (RULES.md §08). This bounds how much in-process instrumentation
can perturb outcomes locally: it cannot cause a timeout-pass, only shift wall
clock.

**A8 — Scope boundaries already hold, by construction, for the paths that must
never rate.** [`classic.run_classic_match`](../../arena/matches/classic.py:48)
returns a tuple and stores nothing;
[`tournaments/classic.py:30`](../../arena/tournaments/classic.py:30) writes
`data/classic_games/`; the remote path
([`arena/remote/env.py:13`](../../arena/remote/env.py:13)) writes
`data/remote_games/` and does not import `arena.matches.loop` at all
*(measured — grep)*. Only
[`competition.run_competition_match`](../../arena/matches/competition.py:41)
feeds `data/games/`, and it hard-rejects `mode != "competition"`
([`competition.py:54`](../../arena/matches/competition.py:54)).

**A9 — The closure (= the bundle = the rating identity) contains the
telemetry code.** Per [game-record-schema.md](game-record-schema.md) §Bot
content hash, the closure is every file in the bot's own directory plus every
`bots/` module it imports transitively — which covers the `telemetry_extras()`
methods inside each `agent.py` *and* `bots/_common/wire.py`'s emission code.
[`bundle.py:118`](../../arena/bundle.py:118) ships exactly that closure. Two
consequences: today's bundles carry instrumentation the judge never uses
(violating constraint 2), and **any edit to `wire.py` forks the content hash —
and therefore the rated entity — of every bot at once**. Both are fixed by the
same move (§2.2): telemetry leaves the closure entirely, after which
instrumentation edits never fork a hash again. The one-time cleanup that gets
there forks everything once and needs one regeneration round (~60 min, ratings
plan §6 step 8).

**A10 — Replay inputs are already stored.** `seed`, `mode`, and
`engine_version` are required GameRecord fields;
[`make_board`](../../competition-module/competition/matchup.py:101) builds the
board deterministically from `PRNGKey(seed)`, and the JAX `transition` is a
pure function of `(state, actions)`. The only thing missing for full
reconstruction is the action sequence — which A4 shows the loop already holds.

**A11 — The suite has almost no headroom.** *(measured)* 277 tests pass in
**6.87 s warm** against the 7 s ceiling
([`AGENTS.md`](../../AGENTS.md) tester §4). Separately *(measured)*:
`jax` + `generals` import costs 0.25 s, the **first jitted `transition` step
costs 1.35 s**, and 100 subsequent steps cost 4 ms. No current test jits the
transition *(measured — grep)*, so any in-suite replay test pays that ~1.6 s
itself. §6 works around this.

**A12 — Several v4 fields have zero readers.** *(measured — grep over
`arena/` and `scripts/`, tests excluded)* Consumers of each `GameRecord`
field:

| Field | Readers |
| --- | --- |
| `bot_a`/`bot_b`, content hashes, `winner` | the fit ([`counts.py:173-179`](../../arena/records/ratings/counts.py:173)) |
| `mode`, `schema_version`, `engine_version` | eligibility ([`policy.py:95-107`](../../arena/records/ratings/policy.py:95)) |
| `turns` | reporting ([`reporting.py:84-105`](../../arena/records/reporting.py:84)), round reports |
| `seed`, `game_id` | round reports, tournament output sort ([`tournaments/competition.py:297`](../../arena/tournaments/competition.py:297)) |
| `truncated`, `terminated`, `castles_built_*`, `metrics["land_margin_*"]` | round reports only ([`measure_heuristics.py:160-175`](../../scripts/measure_heuristics.py:160)) |
| `started_at`, `finished_at`, `duration_seconds` | **nothing** |
| `final_land_*`, `final_army_*` | **nothing** (the same-named keys in `arena/remote/` are a different record type) |
| `round` | **nothing** beyond construction validation ([`store.py:119`](../../arena/records/store.py:119)) |

Two derivability facts that matter for slimming: `terminated` is exactly
`winner != "draw"` under the loop's mapping
([`loop.py:104-112`](../../arena/matches/loop.py:104) with
[`loop.py:217`](../../arena/matches/loop.py:217)) — fully redundant. But
`truncated` is **not** derivable from `winner`: a simultaneous general capture
(RULES.md §02) is a draw with `truncated=False`, and losing that distinction
would silently misclassify "both bots stalled" vs "mutual kill". Also,
`game_id` embeds a UTC minute-stamp
([`store.py:252-258`](../../arena/records/store.py:252)), so dropping the
timestamp fields loses only sub-minute provenance.

### Sizing the worst case

Board ≤ 21×21 = 441 cells; grids are `H*W*3` ints per turn; cap 1200 turns;
worst case 10.5k games × 2 seats:

| Encoding | Per game | × 10,501 games | Verdict |
| --- | ---: | ---: | --- |
| Full state per turn, JSON | ~10 MB | ~100 GB | ruled out |
| Full state per turn, npz int16 raw | ~3.2 MB | ~33 GB | ruled out as the *stored* form |
| **Actions + scalar series, jsonl.gz** | **~12 KB** | **~130 MB** | chosen — states are reconstructed by replay (A10) |
| Bot trace per seat, jsonl.gz | ~25 KB | ~260 MB | acceptable, and opt-in |

The decisive fact is A10: since `(seed, engine_version, actions)` determines
every state, storing states is caching, not recording. Dense tensors for RL
are a *materialization* produced by replay on demand, not a stored primary.

---

## 2. Recorder design (Part A)

### 2.1 One source or two? — Both, because they answer different questions

**Recommendation: build both, as two separate channels with one storage root.**

- **Engine-side truth recorder** (in `run_stdio_match`): seed, both decoded
  actions per turn, per-turn scalar series (land/army per seat), castle-build
  events, and periodic state digests. Fog-free, both seats, ground truth.
  **Unlocks:** RL trajectories (states + both actions + outcome, via replay),
  truthful per-turn land/army curves (a bot's `opp_*` view is fog-limited),
  and game reconstruction/debugging. Independently of recording, the loop
  starts computing **final land/army from the terminal state** and returning
  them on `MatchLoopResult` — so every new record carries engine-truth
  `final_land_*` / `final_army_*` in `metrics` (§2.4) with no bot cooperation
  at all.
- **Bot-side probe trace** (arena-owned instrumented runner, §2.2): the bot's
  internal state sampled **every turn** by a probe that lives outside the
  bot's closure. **Unlocks:** heuristic tuning evidence — when the phase
  switched, when the belief formed, why a move family was chosen. The engine
  cannot produce this; it is precisely the part of the game state that lives
  inside the bot.

Neither substitutes for the other: engine truth cannot say what the bot
believed, and bot beliefs are not ground truth. The engine-side recorder is
the **canonical** one — it alone carries the replay guarantee — and the
bot-side trace is an attachment to it.

### 2.2 Bot-side transport — an instrumented runner, not a channel in the bot

Constraint 2 rules the first draft's design out: any mechanism *inside*
`wire.py` or `agent.py` — per-turn stderr lines, an env-var-gated trace
writer, even a dormant five-line hook — ships to the judge, because the
closure includes every file in the bot dir plus `_common/wire.py` (A9).
"Clean bundle" therefore means the instrumentation must not exist in `bots/`
at all.

**Design (probe location decided 2026-08-01, §8 O7): an arena-owned
instrumented entry point, with each bot's probe living next to its agent as
`bots/<name>/probe.py` — excluded from the closure by rule.**

```
arena/instrument/
  runner.py               python -m arena.instrument.runner <bot_dir> — drives
                          the stdio protocol for one seat, instrumented
bots/<name>/probe.py      def extras(agent) -> dict   # passive reader,
                          NOT part of the closure, the hash, or the bundle
```

How it works:

- `bots/_common/wire.py` is reduced to the pure protocol loop: handshake,
  observation parsing, `agent.act`, action write. `_telemetry_line` and the
  EOF emission are **deleted**, along with every `telemetry_extras()` method
  in the nine `agent.py` files. That is the entire bundled surface — nothing
  else remains to strip.
- The instrumented runner imports the *same* modules the clean entry point
  does — `Observation` and the parsing helpers from `bots/_common/wire.py`,
  the bot's `Agent` from its `agent.py` — and drives the identical
  read-obs/act/write-action cycle itself, loading `bots/<name>/probe.py` (a
  bot without one gets engine-side recording only), calling `extras(agent)`
  after each `act`, and appending one trace line per turn. Probes are
  **passive readers** of agent attributes (the current `telemetry_extras`
  bodies, §1 A2, are already exactly that).
- **The closure rule gains one exclusion, with a guard.**
  [`fingerprint.bot_source_closure`](../../arena/records/fingerprint.py)
  currently takes every file in the bot's directory; it will skip the fixed
  name `probe.py` (alongside the existing `__pycache__`/`*.pyc` exclusions),
  and [`bundle.py`](../../arena/bundle.py) inherits the exclusion for free
  since it ships exactly the closure. The invariant that keeps this honest:
  **unhashed code must be unreachable from the hashed program** — so
  `fingerprint` raises, loudly, if any module in the closure imports `probe`.
  A probe that influenced play would otherwise be invisible to the rating
  identity; the guard turns that bug into a hard error instead of a silent
  under-hash. Consequences worth having: adding or editing a `probe.py`
  **does not fork the bot's hash**, and the rule change itself forks nothing
  (no bot has a `probe.py` today, so every current hash is unchanged).
- Spawn selection lives in the harness:
  [`spawn_agent`](../../arena/matches/loop.py:71) launches `bash run.sh` as
  today for normal runs, and `python -m arena.instrument.runner <bot_dir>
  --trace <file>` when the caller asked to record. `cm_*` bots (submodule
  wrappers, no probes) always spawn via `run.sh`; recorded matches involving
  them get engine-side trajectories only.

The three original constraints, re-checked under this design:

- **stdio untouched:** the runner speaks the same protocol from the same
  parsing code; the wire format does not change.
- **Timing / outcome perturbation:** a probe is a dict read plus a buffered
  ~80-byte write per turn (µs), and locally there is no per-move clock (A7).
  The strong check is behavioural, not temporal: because the engine recorder
  logs the decoded action sequence, a paired-seed run — clean `run.sh` vs
  instrumented runner, same seed — must produce **identical action sequences**
  (§6 T3). That is a stricter guarantee than the first draft could offer.
- **Bundle-safe:** trivially now — `arena/` is never bundled, `wire.py` is
  clean, and [`bundle.py`](../../arena/bundle.py) needs no changes beyond its
  docstring. The bundle gets strictly smaller than today's.

What this buys beyond compliance: **instrumentation stops being
hash-coupled.** Today every telemetry tweak would fork all ~20 entities (A9).
After the one-time cleanup fork, probe and runner edits are invisible to
fingerprints, bundles, and the leaderboard — and the probe sits in the same
directory as the agent it reads, so a refactor is likely to touch both
together.

The honest trades, stated. First, the closure rule — the definition of rating
identity — gets edited, and a bug there ships or under-hashes silently; the
import guard above plus fingerprint tests (§6 T9) are the containment. Second,
probes still fail on agent refactors that rename attributes; they fail loudly
(an `AttributeError` aborts the recorded match rather than logging garbage),
and the schema gate (§3.1) catches renamed keys. Accepted — instrumentation
that is out of date should break, not guess.

### 2.3 Format, volume, retention

**Recommendation: gzip-compressed jsonl, one directory per round, one file set
per game, off by default.**

```
data/trajectories/<round>/<game_id>.traj.jsonl.gz      # engine-side, canonical
data/trajectories/<round>/<game_id>.trace.a.jsonl.gz   # bot-side, seat A (when probed)
data/trajectories/<round>/<game_id>.trace.b.jsonl.gz   # bot-side, seat B (when probed)
```

Engine trajectory, line-oriented (`v` guards the format):

```jsonl
{"v": 1, "game_id": "…", "seed": 7, "mode": "competition", "round": "…", "engine_version": "…", "bot_a": "…", "bot_b": "…", "H": 19, "W": 21}
{"t": 1, "a": [0,3,4,1,0], "b": [1,0,0,0,0], "land": [2,1], "army": [12,11]}
{"t": 100, "digest": "sha256:…"}
{"end": {"winner": "a", "turns": 412, "terminated": true, "truncated": false}}
```

- `a`/`b` are the **decoded actions the engine actually applied** (`a_0`,
  `a_1` at [`loop.py:186-187`](../../arena/matches/loop.py:186)) — invalid
  moves included, since silent passes are the engine's job to resolve.
- `digest` every 100 turns and at the end: sha256 over the canonical
  concatenation of `armies`/`ownership`/`castles` bytes. Costs µs, and turns
  replay verification from "equal at the end" into "localize the first
  divergent century" if JAX ever produces one.
- Bot trace lines are `{"t": N, …probe dict}`, one per turn, written by the
  instrumented runner; the harness gzips into place after `close_agent`.
- gzip because it is stdlib — see open question O2 for the zstd trade.

Worst case (§1 table): ~130 MB engine-side + ~260 MB bot-side at the full
10.5k-game scale — and recording is opt-in per run, so the realistic volume is
per-experiment. `.gitignore` gains, in the existing scoped-glob style:

```gitignore
# Per-turn trajectories (derived, opt-in; see docs/arena/trajectories.md)
data/trajectories/**
```

Retention: no auto-GC in this pass. A round's trajectory dir lives and dies
with the experiment that recorded it — delete it when the experiment note is
published, keep it while a tuning effort or the RL work references it (open
question O5). Previous rounds' `data/games/` records are **kept** regardless —
constraint 1 forbids deleting round data, and nothing here needs to.

### 2.4 Storage separation and the lean v5 record

Trajectories are keyed by `game_id` and live entirely under
`data/trajectories/<round>/` — they never enter the record. The rating fit is
untouched by construction: `ratings/counts.py` reads only identity and outcome
fields and never touches `metrics` (A12), and nothing in the fit path learns
that trajectories exist.

The record itself slims to what A12 proves is consumed, plus `round` — unused
today, but this plan makes it the trajectory-directory key, so it earns its
place now rather than by convention.

**Schema v5:**

| | Fields |
| --- | --- |
| Required | `game_id`, `seed`, `mode`, `round`, `bot_a`, `bot_b`, `bot_a_content_hash`, `bot_b_content_hash`, `engine_version`, `winner`, `turns`, `truncated` |
| Optional | `schema_version` (5; absent or lower is a hard error), `metrics` |
| Dropped | `started_at`, `finished_at`, `duration_seconds` (zero readers; `game_id`'s embedded stamp keeps minute provenance — A12); `terminated` (≡ `winner != "draw"`, A12; `measure_heuristics` derives it) |
| Moved into `metrics` | `castles_built_a/_b` (engine tally), `final_land_*` / `final_army_*` (now terminal-state truth, §2.1), `land_margin_*` (already there) |

The split rule: **required fields are the rating identity and the outcome;
everything observational lives in `metrics`.** `truncated` stays required
because it is outcome, not observation — a 1200-cap stall and a
simultaneous-capture draw are different results that `winner` alone cannot
distinguish (A12), and the ratings plan's draws-are-truncations analysis
depends on telling them apart.

**Migration of the 10,501 stored records** (constraint 1: keep the data, no
dual-version reader): a one-shot projection script — drop the dropped fields,
move `castles_built_*`/`final_*` values into `metrics` where present, stamp
`schema_version: 5`, rewrite in place. Pure field shuffling, no
reinterpretation; runs in seconds; committed (it documents the projection)
even though it runs once. Afterwards `MIN_SCHEMA_VERSION = 5` and
`from_dict` rejects v4 as loudly as v4 rejected v3 — there is never a reader
with two branches. The fit's inputs (identity + winner) are untouched by the
projection, so ratings over the migrated pool are identical.

### 2.5 Reconstruction, precisely

"Reconstruct a game" means: given a trajectory file and the same
`engine_version` checkout,

1. rebuild the board with `make_board(GeneralsEnv(mode=mode), seed)`;
2. apply the recorded action pairs through `make_transition(env)` in order;
3. after every step, the recomputed land/army scalars equal the recorded
   `land`/`army` line; at every digest line, the state digest matches;
4. at the end, `(winner, turns, terminated, truncated)` equal both the
   trajectory's `end` line and the stored `GameRecord`.

A game reconstructs **iff** all four hold. This is the recorder's acceptance
test: record one real seeded match, replay it, assert. The replayer is also
the RL materializer — `python -m arena.records.trajectories --replay <traj>`
yields per-turn full states (and, via `get_observation`, either seat's fog
view — which is why observations are not stored: they are a pure function of
state and player).

Era guard: replay refuses to run when the current `competition-module` HEAD
differs from the trajectory's `engine_version` — same reasoning as the rating
era boundary (ratings plan §9 q2).

### 2.6 Cost budget and process safety

**Budget: ≤ 2% wall clock with recording on** — ≈60 ms against the measured
3.07 s mean match (ratings plan, measurement audit). Plausibility: per turn
the recorder appends ~10 small Python ints to a list (the scalars are already
computed as observation fields, A4); the gzip write is one ~85 KB stream at
match end; digests are 12 hashes of ~3 KB each; the probe is a dict build the
bots used to do once, now done per turn. **Measured how:** 20 matches on fixed
seeds, recorder off then on, same machine, paired per seed; report mean and
max delta in the enabling commit's message. The same paired run feeds §6 T3's
identical-action-sequence assertion.

Process safety: identical pattern to `save_game` (A6) — the worker that played
`game_id` is the only writer of that game's trajectory files, into a directory
created by the parent (`run_tournament` already creates the round dir). No
shared file, no lock, no manifest. Write to `<name>.tmp` then `os.replace`, so
a killed worker leaves no half-readable trajectory.

### 2.7 Scope guards

Classic and remote paths **do not record at all** in this pass.

- `run_stdio_match` gains `recorder: TrajectoryRecorder | None = None`; with
  `None` (the default) the loop body is unchanged and every bot is spawned via
  `run.sh` exactly as today.
- Only [`arena/matches/competition.py`](../../arena/matches/competition.py)
  ever constructs a recorder or selects the instrumented runner, and only when
  its caller passes an explicit record flag.
  [`classic.py`](../../arena/matches/classic.py) and everything under
  [`arena/remote/`](../../arena/remote) (which does not even import the loop,
  A8) are left untouched, so they *cannot* write into `data/trajectories/` —
  and they already cannot write into `data/games/` (A8).
- A guard test (§6 T6) asserts the classic path constructs no recorder and
  always spawns `run.sh`.

If the human-95 effort later wants classic traces, that is a separate decision
with its own directory (`data/classic_trajectories/`), mirroring how
`data/classic_games/` is fenced today (open question O6).

---

## 3. Telemetry schema and series aggregation (Part B)

### 3.1 Declared, typed telemetry schema

New module `arena/records/telemetry_schema.py`:

```python
class Kind(Enum): BOOL01, INT, TOKEN          # TOKEN: \S+ string, categorical

@dataclass(frozen=True)
class TelemetryKey:
    name: str
    kind: Kind
    meaning: str                              # one line, shown in docs
    reducers: tuple[Reducer, ...]             # applied to the recorded series

TELEMETRY_SCHEMA: dict[str, TelemetryKey]     # the one registry
```

- Probe dicts are validated against the schema at trace time and again at
  record build: `BOOL01 → bool`, `INT → int`, `TOKEN → str`.
- **Unknown keys raise** with the key name. In a tournament this fails the
  worker's future and aborts the round loudly — acceptable and intended:
  probes and schema live in one repo, so the fix is
  a one-line schema entry in the same commit that adds the key. The old
  behaviour — silently landing as a string — is exactly the defect being
  removed.
- Deleted outright (constraint 1, no compatibility layer):
  `_telemetry_line` and the EOF emission in `wire.py`; `parse_bot_telemetry`,
  `BotTelemetry`, `apply_telemetry_to_record`, and `_coerce_metric_value` in
  [`telemetry.py`](../../arena/records/telemetry.py).
  `record_from_match_result` survives but reads finals from the loop result
  (engine truth) and no longer looks at stderr at all. The
  `[telemetry]`-line format disappears from
  [game-record-schema.md](game-record-schema.md).

### 3.2 Series reducers

Defined over a per-turn series `x_1..x_T` (engine scalar series or probe trace
series), emitted into `metrics` **only when the series was recorded**:

| Reducer | Meaning | Metric key shape |
| --- | --- | --- |
| `final` | last frame | `<key>_a` |
| `mean` | arithmetic mean over recorded turns | `<key>_mean_a` |
| `max` | maximum | `<key>_max_a` |
| `argmax_turn` | turn index of the maximum | `<key>_argmax_turn_a` |
| `auc` | sum over turns (discrete area under curve) | `<key>_auc_a` |
| `first_turn_true` | first turn with a truthy value; **absent** if never true | `<key>_first_turn_a` |

Key assignments (initial; each lives in the schema entry, not in code
branches):

| Key | Kind | Reducers |
| --- | --- | --- |
| `enemy_general_sighted` | BOOL01 | final, first_turn_true — the series form *replaces* the old bot-computed `first_sighting_turn`, which is deleted from probes as redundant |
| `phase`, `active`, `label`, `reason` | TOKEN | final |
| `was_attacked`, `countering`, `pushing` | BOOL01 | final, mean, first_turn_true |
| `strikes`, `switches`, `castles_built` | INT (monotone counters) | final |
| `guard` | INT | final, mean, max |
| engine `land`/`army` per seat | INT | mean, max, argmax_turn |
| engine `land_margin` (land_a − land_b) | INT | final, auc, first_turn_true(>0) |

`first_city_capture_turn` is classic-only and dies with `classic_duel`'s
`telemetry_extras` — a classic probe can resurrect it if the human-95 effort
ever records (O6).

### 3.3 The absent-value rule survives

`missing means not measured, never zero`
([game-record-schema.md](game-record-schema.md)) is preserved structurally:

- With recording off (the default), new records carry engine-truth finals and
  **no bot-internals metrics at all** — not measured, so absent. There is no
  EOF-line fallback anymore; a measurement round that wants sighting or phase
  metrics runs with `--record`. Rating rounds don't need them.
- Series reducers add keys only when their series exists; `first_turn_true`
  emits nothing when the predicate never fired.
- The 10,501 migrated records keep their old stderr-derived values as
  `metrics` keys (the v5 projection moves, never invents or deletes values —
  §2.4), and every `metrics` reader uses `.get()`. So on old games "sighted
  on the last frame" is still answerable, while per-turn questions are
  honestly unanswerable — absent, not zero. A regression test (§6 T4) covers
  a migrated-shape fixture.

### 3.4 What the 9 implementations must change

**Mechanical, per-bot, small:** each `telemetry_extras()` method body moves
verbatim into `bots/<bot_id>/probe.py` as
`def extras(agent) -> dict`, and the method is deleted from `agent.py`. The
bodies are already passive attribute reads (§1 A2), so the move is a cut-paste
plus an `agent.` prefix where `self.` was. `classic_duel`'s moves nowhere — it
is remote-only and gets no probe until O6 is revisited. Each moved key gets a
`TELEMETRY_SCHEMA` entry (kind + meaning + reducers) — the §3.2 table *is*
that work, transcribed.

Two per-bot judgment calls, both trivial: drop `first_sighting_turn` (now a
reducer output, §3.2), and keep `proteus.label` / `boom.reason` single tokens
— the TOKEN kind encodes that constraint instead of leaving it implicit.

---

## 4. Module layout

```
arena/instrument/           NEW package — never bundled, never hashed
  runner.py                 instrumented stdio entry point (one seat);
                            loads bots/<name>/probe.py when present
bots/<name>/probe.py        passive per-bot probes (ex-telemetry_extras);
                            excluded from closure/hash/bundle by rule (§2.2)
arena/records/
  fingerprint.py            closure rule: probe.py excluded + import guard
  store.py                  lean GameRecord v5 (§2.4); timestamps gone
  telemetry.py              record build from MatchLoopResult + traces;
                            stderr parsing deleted
  telemetry_schema.py       NEW — TELEMETRY_SCHEMA, kinds, reducers (pure)
  trajectories.py           NEW — TrajectoryRecorder, writer/reader, replay,
                            CLI (--replay, --verify, --materialize)
scripts/
  migrate_games_v5.py       NEW — one-shot v4→v5 projection (§2.4)
arena/matches/
  loop.py                   recorder=None hook; engine-truth finals on
                            MatchLoopResult; instrumented-spawn selection
  competition.py            record flag → recorder + instrumented spawns
  run_match.py              --record CLI flag
arena/tournaments/
  competition.py            --record flag; trajectory dir per round
  worker.py                 payload key "record"; writes its own game's files
bots/_common/
  wire.py                   pure protocol loop — telemetry deleted
bots/<name>/agent.py        telemetry_extras() deleted (nine bots)
```

Wrap-don't-edit holds: nothing in `competition-module` changes; the recorder
consumes `make_board`/`make_transition`/`get_observation` exactly as the loop
already does. [`bundle.py`](../../arena/bundle.py) changes only its docstring —
the closure it ships is simply smaller and clean.

Verification gate (per [AGENTS.md](../../AGENTS.md)) for every step below that
touches match or bot code:

```bash
python competition-module/competition/matchup.py bots/expand_plus/run.sh bots/smoke/run.sh --mode competition --seed 0
```

plus, once the flag exists, one recorded in-process match:

```bash
python -m arena.matches.run_match bots/expand_plus/run.sh bots/smoke/run.sh --seed 0 --record
```

and, after step 2 below, a clean-bundle proof:

```bash
python -m arena.bundle expand_plus --force
```

---

## 5. Migration steps

Ordered; each independently committable, on `main` (no branches). The lean
schema lands first (pure arena code, settles the storage target everything
else writes to); the hash-forking closure cleanup comes immediately after —
early, so every later step builds on the clean closure and there is exactly
one fork instead of a fork per follow-up tweak.

| # | Step | Touches bot closures? | Commit contains |
| --- | --- | --- | --- |
| 1 | **Schema v5 + one-shot migration.** Slim `store.py` to §2.4's field set; `MIN_SCHEMA_VERSION = 5`; writers stop producing timestamps; `measure_heuristics` derives `terminated` and reads castles/margins from `metrics`; run the committed projection script over the 10,501 stored records. Update `game-record-schema.md`. | no | store + writers + migration script + tests T4 |
| 2 | **Strip the closure.** `fingerprint` excludes `probe.py` + gains the imports-probe guard (forks nothing by itself — no probe exists yet); delete `_telemetry_line` + EOF emission from `wire.py`; delete all nine `telemetry_extras()` methods; move their bodies to `bots/<bot_id>/probe.py`; delete `telemetry.py`'s stderr parsing; `MatchLoopResult` + record build gain engine-truth finals in `metrics`. **Forks every bot hash — once** (the wire/agent edits, not the probes). Gate: matchup run + `python -m arena.bundle expand_plus` proving the clean bundle. | **yes — all bots, one time** | fingerprint + wire + agents + probes + tests T3, T8, T9 |
| 3 | **Telemetry schema.** `telemetry_schema.py`; probe validation through it; unknown keys raise. | no | module + tests T1, T4 |
| 4 | **Trajectory format + writer/reader.** `trajectories.py` (no loop hook yet), `.gitignore` entry, new `docs/arena/trajectories.md`. | no | module + tests T2, T5 |
| 5 | **Engine-side recorder + instrumented spawn.** `recorder=None` hook in `loop.py`; instrumented-runner spawn selection; `--record` through `run_match.py`, `tournaments/competition.py`, `worker.py` payload; scope guards. Off by default. | no | hook + runner + flag + tests T6 + gate run with `--record` |
| 6 | **Replay + acceptance.** Replay/verify/materialize in `trajectories.py` CLI; era guard; overhead measurement (§2.6) recorded in the commit message. | no | replay + test T7 (marked) + paired-seed on/off numbers |
| 7 | **Series aggregation.** Record build merges recorded series through the schema's reducers when trajectories exist; unrecorded path emits engine finals only. | no | reducer wiring + test T4 extension |
| 8 | **Regeneration + docs sync.** Fold step 2's hash fork into the next measurement round (one regeneration round repopulates the leaderboard, ratings plan §6 step 8). Remaining docs updates per §7. | no | round report + docs |

Step 2 is the only one with a blast radius, and its cost (one ~60-minute
regeneration round) is paid by a round that measurement work would run anyway.
Old rounds' games stay on disk throughout — step 1's projection rewrites their
shape, never their content, and their ratings are bit-for-bit unaffected
(§2.4).

---

## 6. Test plan

Constraint first: the suite is at **6.87 s warm** against the 7 s ceiling
(A11), so there is ~0.13 s of headroom — and a real replay test costs ~1.6 s
of one-time jit (A11). Per AGENTS.md the ceiling is defended, not argued past:
the replay test therefore lives behind a marker and runs in the verification
gate, not in the default suite.

| # | Test | Asserts | Cost |
| --- | --- | --- | --- |
| T1 | Schema coercion | Each kind coerces; unknown key raises with the key name; every probe's keys are declared (probes are importable without a game) | <10 ms |
| T2 | Writer round-trip | Synthetic 30-turn trajectory → write gz → read → equal; `.tmp`+rename leaves no partial file on injected failure | ~20 ms |
| T3 | Clean wire / equivalent runner | `run_stdio` under a fake stdin emits **nothing** to stderr (extends `test_bot_wire.py`); the instrumented runner, driven over the same fake stdin frames, produces a byte-identical stdout action stream and one parseable trace line per turn | ~20 ms |
| T4 | v5 schema + migration + no-trajectory path | The projection maps a canned v4 record to the expected v5 (moved keys in `metrics`, dropped keys gone, identity/outcome byte-identical); `from_dict` rejects v4 loudly; a new unrecorded record carries engine finals and no probe metrics; reducers emit nothing for absent series; `first_turn_true` absent when never true | ~20 ms |
| T5 | Series reducers | final/mean/max/argmax_turn/auc/first_turn_true on hand-computed fixtures, including the empty and single-turn series | <10 ms |
| T6 | Scope guard | Classic match construction passes no recorder and spawns `run.sh` (never the instrumented runner); no code path resolves a trajectory dir under `data/games/`, `data/classic_games/`, or `data/remote_games/` | ~10 ms |
| T7 | **Replay determinism** (`@pytest.mark.replay`, excluded from default run) | Record a tiny seeded match (smoke vs smoke, small fixed-dims env, ≤60 turns), replay it, assert §2.5's four conditions | ~1.7 s, gate-only |
| T8 | Bundle is clean | Extends `test_bundle.py`: the built zip contains no `arena/` file, no `probe.py`, and a `wire.py` with no telemetry symbols; smoke still passes | ~ms on top of existing |
| T9 | Closure exclusion invariants | Extends `test_fingerprint.py`: adding or editing `bots/<x>/probe.py` leaves the hash unchanged; `probe.py` is absent from `bot_source_closure`; a closure module importing `probe` makes `fingerprint` raise (the unhashed-code-unreachable guard, §2.2) | ~10 ms |

Default-suite addition: **≤ 0.1 s** total (T1–T6, T8, T9). T7 joins the
verification gate command list in §4 and runs on every recorder-touching
commit:

```bash
python -m pytest tests -m replay -q
```

The full-strength equivalence check — clean vs instrumented spawn on the same
seed producing identical engine-recorded action sequences — costs two real
matches (~6 s) and therefore lives in the §5 step-5 measurement run, not in
the suite; T3 covers the same property at protocol level for milliseconds.

---

## 7. Docs this work forces

Small and single-purpose, per AGENTS.md:

- **New:** `docs/arena/trajectories.md` — file formats, replay contract, the
  opt-in flag, probes, retention. Single page.
- [game-record-schema.md](game-record-schema.md) — rewritten for v5: the lean
  field table, the required-is-identity-and-outcome rule, the migration note,
  and the "Bot stderr telemetry" section deleted; `metrics` documented as
  engine-truth finals plus probe-and-reducer output (the latter only on
  recorded games).
- [match-runner.md](match-runner.md), [tournament.md](tournament.md) — the
  `--record` flag and the instrumented spawn.
- `docs/bots/adding-a-bot.md` — telemetry section becomes "add
  `bots/<name>/probe.py` for per-turn introspection; it is excluded from the
  hash and the bundle, and must never be imported by the agent".
- [game-record-schema.md](game-record-schema.md) §Bot content hash — the
  closure contract gains the `probe.py` exclusion and its imports-probe
  guard (part of the v5 rewrite above).
- [`docs/research/learned-bot-plan.md`](../research/learned-bot-plan.md) — its
  "richer per-turn trajectory dumps if/when" open follow-up resolves to a
  pointer at `trajectories.md`.
- [`docs/index.md`](../index.md) — one line for the new page.
- [`AGENTS.md`](../../AGENTS.md) — file-placement table gains the
  `data/trajectories/` row (derived, gitignored).
- [`docs/research/strategies/test-core-skill.md`](../research/strategies/test-core-skill.md)
  — core surface table gains the trajectory writer/reducers and the marked
  replay test.

---

## 8. Decisions

The four questions that required a call were answered on 2026-08-01; the rest
are applied defaults, recorded here so the implementation has one source of
truth.

### Decided (2026-08-01)

**O3 — Replay test placement: gate-only marker, confirmed.**
In-suite would blow the 7 s ceiling by ~1.6 s of jit that cannot be made
cheap (A11). `@pytest.mark.replay`, excluded from the default run, mandatory
in the verification gate for recorder-touching commits (§6 T7). The
raise-the-ceiling alternative was declined.

**O4 — Hash fork timing: fold into the next measurement round, confirmed.**
§5 step 2 forks all ~20 bot entities and needs a regeneration round to
repopulate the leaderboard. Step 2 lands immediately before the next planned
measurement round, folding the regeneration in at zero incremental wall
clock. Nothing else is blocked while waiting: steps 1, 3, and 4 touch only
arena code and can land first (only steps 5+ need the clean closure for the
instrumented spawn).

**O6 — Classic-path recording: out of scope, confirmed.**
The scope guard (§2.7) stays absolute — only the competition path can record.
`classic_duel`'s `first_city_capture_turn` telemetry dies in step 2 with
nowhere to go; if the human-95 effort later wants traces, that is a classic
probe plus a separate `data/classic_trajectories/` root with the same fencing
as `data/classic_games/`, as its own effort.

**O7 — Probe location: `bots/<name>/probe.py`, closure-excluded.**
Decided **against** the plan's original default (probes under `arena/`), for
locality: the probe lives next to the agent it reads, so refactors touch both
together. The cost accepted with it: the closure rule in
[`fingerprint.py`](../../arena/records/fingerprint.py) — the definition of
rating identity — gains an exclusion for the fixed name `probe.py`, and the
schema-doc contract is amended to match. Containment for the risk that made
this the non-default: the **unhashed-code-unreachable guard** (`fingerprint`
raises if any closure module imports `probe`, §2.2) plus T9's invariants
(probe edits never move a hash; `probe.py` never appears in a closure or
bundle). The rule change itself forks nothing — no bot has a `probe.py`
today.

### Applied defaults

Overridable, but the plan now assumes these.

| # | Question | Default applied |
| --- | --- | --- |
| O1 | Dense-tensor layout for RL training | **Don't store dense states at all.** Actions-only canonical form plus a `--materialize` replayer emitting npz per game into a gitignored cache; decide shard layout when the RL work (learned-bot-plan) actually starts. |
| O2 | gzip vs zstd | **gzip** — stdlib, and at ~130 MB worst case the difference is noise. Revisit only if RL-scale materialization measurably bottlenecks on decompression. |
| O5 | Trajectory retention | **Manual.** A round's trajectory dir is deletable once its experiment note is published, kept while tuning or RL work references it; no auto-GC. Revisit when `data/trajectories/` first crosses ~2 GB. |
| O8 | The "11 bots" figure | **9 is the count of record** (§1 A2; `classic_duel` among them never reaches `data/games/`). If more bots were intended (e.g. `garrison`, `late_rush`), each is a new `probe.py` plus a schema entry, per §3.4. |
