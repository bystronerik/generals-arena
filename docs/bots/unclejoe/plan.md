# unclejoe implementation plan

Plan for a new bot `bots/unclejoe/`, forked from `bots/joe-rs/` (the
dependency-free Rust port, compiled at intake — not the Python `bots/joe/`).
Status: **plan only** — no code exists yet.

Concept: keep the ported NN policy (greedy argmax over the joe EMA weights in
`artifact/`) as the default move source, and add a tactics layer built around
a small exact search that overrides the NN only where the best move is cheaply
computable and near-certain. Each turn, cheap trigger predicates decide
whether a tactical situation is on; if one is, a shallow bounded search
resolves it exactly and returns the move; otherwise the NN move passes
through unchanged.

Ground rules honored throughout: `bots/joe/`, `bots/joe-rs/`, and
`competition-module/` internals are never edited (the only joe-rs-adjacent
change is a docs checklist line in
[export.md](../joe-rs/export.md)), and the crate stays dependency-free so
intake still compiles it from source.

## 0. Prerequisite: strategy spec first

A spec is required. The repo workflow (AGENTS.md subagent roles) has
bot-author work "from a spec under `docs/research/strategies/`", and every
prior bot follows it. Step 1 is therefore
`docs/research/strategies/unclejoe.md`, containing: the concept (NN
pass-through + conservative-superset triggers + bounded exact search), the
exact trigger definitions (§4), the search's state/resolution model and caps
(§3), the latency budget table (§5), the evaluation plan (§7), and explicit
non-goals (no search outside the two triggers, no value-head use, no
half-measures like "search suggests, NN decides"). The design content in this
plan is what moves into that spec.

## 1. Copy strategy

Start from `cp -R bots/joe-rs bots/unclejoe` minus `target/`, then adapt:

| Item | Action |
| --- | --- |
| `src/` (io/board/nn/xla_math) | **Copy verbatim.** No edits except `main.rs` and `board/mod.rs` (below). The `mod io` / `use std::io as stdio` alias pattern carries over unchanged. |
| `src/main.rs` | **Adapt:** rename env var `JOE_RS_ARTIFACT` → `UNCLEJOE_ARTIFACT`, stderr tags `[joe-rs]` → `[unclejoe]`, add `mod search;` and `mod tactics;` and the override call in `Seat::act`, drop the `parity` subcommand. Keep `selfcheck` and `bench`. |
| `src/board/mod.rs` | **Adapt:** one added line, `pub mod memory;`, for the new per-game memory file (§3). No copied board file changes. |
| `src/parity.rs` | **Drop.** The parity harness (JAX-oracle `.npz` corpus, tier-2 pins, mutation kills) stays owned by joe-rs; duplicating it doubles the re-export checklist for zero information since `nn/` is byte-identical. unclejoe's sequence-level guard is the wire-replay test with tactics disabled (§6). |
| `src/board/memory.rs`, `src/search/`, `src/tactics/` | **New** (§3). |
| `Cargo.toml` | **Adapt:** `name = "unclejoe"`. Keep `[dependencies]` empty (hard requirement: intake compiles one crate from source), keep the release profile; the `[profile.mutation]` section goes with `mutation_check.py`. |
| `Cargo.lock` | Copy (trivial with zero deps) — or regenerate; either is fine. |
| `rust-toolchain.toml`, `.cargo/config.toml` | **Copy verbatim** (1.97.1 pin, `target-cpu=x86-64-v3`). |
| `artifact/` | **Byte-copy** `model.safetensors` + `manifest.json` from joe-rs. Do not edit the manifest — provenance lives in docs and the version registry, and a byte-identical manifest keeps the `safetensors_sha256` pin verifiable against joe-rs's. |
| `run.sh` | **Adapt:** binary path `target/release/unclejoe`, `UNCLEJOE_ARTIFACT` export, same source-stamp rebuild logic. **Rewrite all comments to contain no path strings** — the closure scan pulls any path mentioned in run.sh prose into the rated closure (joe-rs's run.sh drags its port-plan doc in this way). Refer to docs by prose name only, never with path syntax. |

Per-tool decisions (`bots/joe-rs/tools/` — six files, all naming joe-rs
source paths):

| Tool | Decision |
| --- | --- |
| `package_submission.py` | **Adapt.** Change `bot_id`/`binary` to `unclejoe`, `BOT_DIR`, the generated bundle `run.sh` (env var + binary name), and the smoke-frame check (the fog-everywhere smoke position triggers no tactic, so the expected reply class is unchanged). It keeps using the shared `arena/rust_bundle.py`. |
| `convert_artifact.py` | **Drop.** unclejoe never converts from `ema.eqx`; it derives from joe-rs's already-converted artifact. |
| `capture_fixtures.py`, `make_synthetic_long.py`, `make_smoke_fixture.py` | **Drop.** Corpus production stays with joe-rs; unclejoe's wire-replay test reads the shared corpus at `data/joe/joe-rs-parity/` read-only. |
| `mutation_check.py` | **Drop.** Its mutation list pins exact `nn/net.rs` / `board/obs.rs` source lines and drives the joe-rs parity pytest; that harness proves the *shared* code once, in joe-rs. unclejoe's tactics code gets direct unit tests instead (§6). |
| **New:** `tools/sync_artifact.py` | Small script: byte-copy `bots/joe-rs/artifact/*` into `bots/unclejoe/artifact/`, verify `safetensors_sha256` from the manifest against the copied file, refuse on mismatch. This is the single deliberate resync knob (§2). |

## 2. Identity and provenance

- **Own rating identity.** The content hash covers `src/`, `Cargo.*`,
  `run.sh`, and `artifact/model.safetensors` (with `target/` skipped), so
  unclejoe registers as a new lineage in `data/bot_versions/` at its first
  rated round. That is correct and intended — it is a different program.
- **Exporter fan-out.** unclejoe becomes a second derived target of the joe
  artifact, one hop further removed (joe → joe-rs → unclejoe). Policy:
  **unclejoe tracks joe-rs**, not joe directly — the whole point of the
  joe-rs vs unclejoe contrast is isolating the tactics layer, which requires
  identical weights. Concretely: append one step to the "After a joe
  re-export" checklist in [export.md](../joe-rs/export.md) —
  `python bots/unclejoe/tools/sync_artifact.py && cargo build --release
  --manifest-path bots/unclejoe/Cargo.toml` — so a joe re-export cannot
  silently leave unclejoe on stale weights. `sync_artifact.py`'s sha
  verification makes a *partial* sync (manifest copied, safetensors not)
  fail loudly. A resync forks unclejoe's content hash, same as it forks
  joe-rs's — expected, and the registry records it.
- **run.sh closure hygiene** as in §1: no path strings in comments, so
  unclejoe's closure is exactly its code + artifact.

## 3. The new modules: `board/memory.rs`, `src/search/`, `src/tactics/`

Three additions, split so that per-game memory, the exact-search engine, and
the trigger/override policy each have one owner. The layering DAG extends to
`xla_math io → board → search → tactics → nn → main` (as in joe-rs, a module
may name the ones above it and must not name the ones below it; `nn/` never
learns the new modules exist). `search` depends on `board` only; `tactics`
reads `board::memory` and drives `search`; `main` owns the memory update
and calls `tactics`.

- `board/memory.rs` — persistent per-game memory the wire frame alone can't
  give. It lives in `board` because it is board state, not policy: derived
  turn by turn from the frame with no knowledge of triggers or search.
  - `enemy_general: Option<usize>` — set the first turn a cell has
    `type == TYPE_GENERAL && owner == OWNER_OPP`; a general never moves, so
    this never goes stale (only its army does).
  - `mountains: Vec<bool>` — ever-seen mountains (same accumulation rule as
    `AugState.mountains`, but a separate copy rather than a reach into the
    NN pipeline's state; it is a 441-bool array).
  - `general_visible_now: bool`, `last_seen_general_army: i32`,
    `last_seen_turn: i32` — for the fog bounds below.
- `search/sim.rs` — the local forward model. State it searches over: a
  board patch (the full ≤21×21 grid is fine — 441 cells; only *move
  generation* is windowed, not storage) of
  `(passable, owner ∈ {me, opp, neutral, unknown}, army: i32)` built from
  the current frame plus `board::memory`. It implements the engine's exact
  resolution semantics from RULES.md:
  - simultaneous moves with the §02 priority ladder — chasing > reinforcing >
    smaller-army-first, ties falling through;
  - combat per §05 — strictly-more captures, tie keeps the defender, mover
    leaves ≥1 behind, full or half split;
  - growth per §04 applied by `obs.turn` parity — generals/castles +1 every
    other turn, all cells +1 every 50 turns;
  - **deathtouch from turn 800** (§07) — any executed move onto the enemy
    general wins regardless of army; defense is a chase of the attack's
    source;
  - both generals captured same turn = draw (a draw counts as failure for
    both the kill proof and the defense proof — conservative in both
    directions).
- `search/minimax.rs` — depth-limited minimax over *(our move, their reply)*
  pairs, our moves maximizing, theirs minimizing, resolved through `sim.rs`.
  The search takes a goal (prove kill / prove survival) and a bound set as
  arguments and knows nothing about triggers.
  Depth: at most **3 of our moves**. Move generation is windowed: our
  candidate moves come only from own cells with army > 1 within Manhattan
  distance `depth + 1` of the target general (the enemy's for kill, ours for
  defense), each with 4 directions × {full, half}; the opponent's candidate
  moves come from every enemy-*possible* cell in the same window — including
  fogged cells under the pessimistic model below — plus "pass". Terminal
  conditions: enemy general captured (win, for the kill search), our general
  captured or drawn (loss, for the defense search), or depth exhausted (no
  proof). The search returns a move **only when the outcome is proven for
  every opponent reply at every ply**; anything short of a proof means no
  override.
  - **Hard caps:** a node budget (proposed: 100,000 nodes) and a wall-clock
    check every 1,024 nodes against a 20 ms deadline. Hitting either cap =
    decline, NN move passes through. Both are constants in `search/`,
    quoted in the spec.
- `src/tactics/` — the policy layer: `tactics/triggers.rs` holds the two
  predicates (§4), and `tactics/mod.rs` the per-turn orchestration —
  evaluate the triggers against the frame and a read-only `&Memory`, invoke
  `search` with the matching goal, and return `Option<Action>` for
  `Seat::act` to apply. No simulation or minimax code lives here, and
  tactics never mutates memory: the memory update is `Seat::act`'s job
  (below), so it cannot go stale when tactics is disabled or declines early.
- **Fog, pessimistically.** The wire scalars give `opp_army` exactly, so the
  **hidden-army budget** `hidden = opp_army − Σ(visible enemy army)` is a
  sound bound on everything in fog. The model: any never-seen-mountain,
  currently-fogged cell (`TYPE_FOG` or `TYPE_STRUCTURE_IN_FOG`, minus
  remembered mountains) may be enemy-owned holding up to `hidden` army, at
  every such cell independently. For the **defense** search this is the
  adversary's side, so pessimism makes proofs *harder* (correct direction:
  only override on moves that save the general even against the worst hidden
  army). For the **kill** search the same pessimism applies to enemy
  reinforcements and chase sources; a kill is proven only if it lands
  against `last-seen general army + growth since seen + worst-case
  resolved-first reinforcement`. One structural helpfulness: every cell
  8-adjacent to a tile we own is visible (vision is the 3×3 pool), so all
  1-ply threats to our general are always in view; fog only enters at depth
  ≥ 2.
- **Integration point in `Seat::act`:** update `board::memory` from the
  fresh frame first — unconditionally, every turn, exactly like `AugState`:
  both are per-game state that must advance regardless of who chooses the
  move and regardless of the tactics kill-switch. Then run the full existing
  pipeline — frame→raw, masks, `augment_obs`, forward, argmax — then consult
  tactics (frame + `&Memory`, read-only) and, if it proved a move, emit that
  instead of the NN's. Never skip the forward: it keeps the state code
  untouched, keeps the value telemetry, and its cost is already budgeted.
  Env kill-switch `UNCLEJOE_TACTICS=0` reverts to pure pass-through (used by
  the equality test and any future A/B debugging). Everything stays inside
  the existing `catch_unwind`, so a tactics panic degrades to a pass, not a
  forfeit.

## 4. Trigger predicates (exact, conservative-superset)

Both are O(board-scan) integer checks on the already-parsed frame + tactics
state. Firing costs only a search that then declines; missing a real case is
the failure mode, so both are deliberately loose.

- **Kill trigger.** `enemy_general` is known, and there exists an own cell
  with `army > 1` within Manhattan distance `D` (the search depth budget,
  ≤3; plain Manhattan distance at trigger time — path feasibility through
  mountains is the search's job, and ignoring it keeps the superset
  property) of it, with either
  - `obs.turn ≥ 800` (deathtouch: any unit is potentially lethal), or
  - our total army in that neighborhood `> last_seen_general_army` (a stale
    lower bound on the defense — an underestimate keeps the superset
    property, and the search applies the honest pessimistic upper bound
    before overriding).

  Fog caveat: "adjacent to the enemy general" is only meaningful once the
  general has been *seen* — before `enemy_general` is set the trigger can
  never fire, and that is correct, not a gap: no exact kill is provable
  against an unlocated general.
- **Immediate-defense trigger.** Let `g` be our general (always known — it
  is ours). Fire if either
  - any *visible* enemy-owned cell within Manhattan distance `D` of `g` has
    `army ≥ army(g)` (loose: ignores the leave-one-behind rule and
    multi-step attrition — superset), or
  - `obs.turn ≥ 800` and any visible enemy cell with `army > 1` is within
    `D` (deathtouch threat), or
  - any fogged cell within `D` of `g` exists **and** `hidden ≥ army(g)` (a
    hidden stack could be sitting one step outside vision).

  As noted, distance-1 threats are always visible, so the pure-fog arm only
  matters at `D ≥ 2`.

Expected fire rate is low (most turns neither general has anything within 3
tiles); the shadow-mode milestone (§8, U2) measures it before the override
goes live.

## 5. Latency

Current joe-rs full-path cost is measured, not estimated: **p99 23.7 ms, max
34.5 ms on the Modal 1-core x86-64-v3 proxy** (`joe-rs bench` over the
1,572-turn synthetic-long log — [latency.md](../joe-rs/latency.md),
2026-08-15). Worst-case unclejoe turn = that pipeline + trigger scan (two
passes over ≤441 cells of integer compares — microseconds) + search capped
at **20 ms** hard deadline ⇒ worst case ≈ 55 ms, about a third of the 150 ms
limit (RULES.md §08), and the cap is enforced by wall clock, not by hoping
the node estimate holds. Verification: keep the `bench` subcommand and re-run
`unclejoe bench` on the same synthetic-long log (triggers will fire on some
late-game frames of that corpus, exercising the real combined path), locally
and via the Modal bench script pointed at the new crate; add one constructed
worst-case position (max own cells + max fog in-window, trigger on) as a
Rust test asserting the search returns under the node cap.

## 6. Tests

- **Rust (`cargo test`, zero cost to the Python budget):** in
  the new modules' unit tests, each beside its code —
  - `board/memory.rs`: enemy-general lock-in on first sighting, mountain
    accumulation, last-seen army/turn tracking through visibility changes;
  - each trigger: fires on a position with a real threat/kill at each depth
    1–3; stays silent with the general absent, out of range, behind
    remembered mountains, or pre-800 with insufficient army; the fog arm
    fires exactly when `hidden` crosses the threshold;
  - `search/sim.rs`: priority-ladder cases straight from RULES.md §02 (chase beats
    reinforce, reinforce beats smaller-army, tie fall-through),
    tie-keeps-defender, growth parity, deathtouch and its chase defense;
  - `search/minimax.rs`: finds a forced 1-, 2-, and 3-ply kill on small fixed
    positions; finds the saving chase/reinforce defense; **declines** when
    the "kill" fails against the best reply, when a fogged cell's hidden
    budget can defend, and when the node cap trips (cap set artificially low
    in the test).
- **Python (`bots/unclejoe/tests/`, outside the content hash):** one adapted
  `test_wire_replay.py`: pipe the shared `data/joe/joe-rs-parity/` corpus
  logs through `unclejoe` **with `UNCLEJOE_TACTICS=0`** and require
  reply-for-reply equality with Python joe's recordings — proving the copied
  NN path is untouched. Mark it with a new `unclejoe` marker added to
  `pytest.ini`'s `markers` and to the `addopts` exclusion, exactly like
  `joe`. Since `testpaths` already includes `bots`, the default suite
  collects-and-deselects it at ~zero cost: **the 15 s budget is untouched**.

## 7. Verification and measurement

1. **Gate** (AGENTS.md, with the known environment traps: absolute `PYTHON`,
   cargo on PATH):

   ```bash
   export PATH="$HOME/.cargo/bin:$PATH" && \
   PYTHON=$PWD/.venv/bin/python python competition-module/competition/matchup.py \
     bots/unclejoe/run.sh bots/joe-rs/run.sh --mode competition --seed 0
   ```

   Must reach a normal end; store the game under `data/games/<round>/`
   before any refit.
2. **Contrast**, per [decision-rule.md](../../arena/decision-rule.md):
   baseline `A` = joe-rs's current registered hash (needs ≥30 games / not
   provisional), candidate `B` = unclejoe, **both arms in the same round** —
   trivially satisfied here since both bots exist side by side, no frozen
   copy needed. One round via `arena.tournaments.competition` with a shared
   ≥5-bot panel including `cm_expander`, pinned `--round-seed`,
   `--seat-policy alternate`, `--strict-versions`; ≥200 games/arm minimum,
   ~1150/arm for a ±25 CI. Read
   `fits["<round>"].delta(joe-rs, unclejoe)` and quote
   `Δ ± SE, CI₉₅, P(B>A)`, games per arm, verdict — never a rank.
   **Replicate in a second, separately scheduled round before writing the
   result down**, and note host state by hand; unclejoe is mildly
   deadline-shaped (the 20 ms search cap), which is exactly the class the
   replication rule binds hardest on.
3. Both arms must be on the **same network** — if joe re-exports
   mid-evaluation, resync both or neither (§2).

## 8. Milestones

- **U1 — faithful fork.** Copy + rename per §1, tactics module absent,
  `sync_artifact.py`, adapted run.sh/selfcheck/bench/packager. Done when:
  `cargo test` passes, wire-replay equality passes, the matchup gate
  finishes, `unclejoe bench` ≈ joe-rs's numbers.
- **U2 — memory + triggers, shadow mode.** `board/memory.rs` and both
  triggers in `tactics/`, logging fires to stderr, never overriding. Done
  when: memory and trigger unit tests pass and a gate match's stderr shows
  sane fire rates.
- **U3 — search + override.** The `search/` sub-package (`sim.rs`,
  `minimax.rs`), the `tactics/` orchestration and override in `act`,
  `UNCLEJOE_TACTICS=0`. Done when: all §6 tests pass and wire-replay
  equality still holds with tactics off.
- **U4 — latency proof.** `bench` on synthetic-long (local + Modal),
  worst-case-position cap test. Done when: p99 and max are recorded in a
  `docs/bots/unclejoe/latency.md` alongside the cap constants.
- **U5 — measurement.** §7's round, then the replication round; verdict
  quoted per the decision rule; `update-leaderboard`.
