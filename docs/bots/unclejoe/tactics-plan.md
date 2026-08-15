# unclejoe tactics plan

Plan for the tactics layer of `bots/unclejoe/`. Status: **plan only** — no
code exists yet. Prerequisite: the fork is finished per
[fork-plan.md](fork-plan.md) (milestone U1: behavior-identical to joe-rs,
gate passed, wire-replay equality proven).

Concept: keep the ported NN policy (greedy argmax over the joe EMA weights
in `artifact/`) as the default move source, and add a tactics layer built
around a small exact search that overrides the NN only where the best move
is cheaply computable and near-certain. Each turn, cheap trigger predicates
decide whether a tactical situation is on; if one is, a shallow bounded
search resolves it exactly and returns the move; otherwise the NN move
passes through unchanged.

## 1. The new modules: `board/memory.rs`, `src/search/`, `src/tactics/`

Three additions, split so that per-game memory, the exact-search engine, and
the trigger/override policy each have one owner. The layering DAG extends to
`xla_math io → board → search → tactics → nn → main` (as in joe-rs, a module
may name the ones above it and must not name the ones below it; `nn/` never
learns the new modules exist). `search` depends on `board` only; `tactics`
reads `board::memory` and drives `search`; `main` owns the memory update
and calls `tactics`.

Source edits beyond the new files: `main.rs` gains `mod search;`,
`mod tactics;`, and the override call in `Seat::act` (§2's integration
point); `board/mod.rs` gains one `pub mod memory;` line. No copied board,
io, nn, or xla_math file changes.

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
  predicates (§3), and `tactics/mod.rs` the per-turn orchestration —
  evaluate the triggers against the frame and a read-only `&Memory`, invoke
  `search` with the matching goal, and return `Option<Action>` for
  `Seat::act` to apply. No simulation or minimax code lives here, and
  tactics never mutates memory: the memory update is `Seat::act`'s job
  (§2), so it cannot go stale when tactics is disabled or declines early.
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

## 2. Integration point in `Seat::act`

Update `board::memory` from the fresh frame first — unconditionally, every
turn, exactly like `AugState`: both are per-game state that must advance
regardless of who chooses the move and regardless of the tactics
kill-switch. Then run the full existing pipeline — frame→raw, masks,
`augment_obs`, forward, argmax — then consult tactics (frame + `&Memory`,
read-only) and, if it proved a move, emit that instead of the NN's. Never
skip the forward: it keeps the state code untouched, keeps the value
telemetry, and its cost is already budgeted. Env kill-switch
`UNCLEJOE_TACTICS=0` reverts to pure pass-through (used by the equality
test and any future A/B debugging). Everything stays inside the existing
`catch_unwind`, so a tactics panic degrades to a pass, not a forfeit.

## 3. Trigger predicates (exact, conservative-superset)

Both are O(board-scan) integer checks on the already-parsed frame + memory.
Firing costs only a search that then declines; missing a real case is the
failure mode, so both are deliberately loose.

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
tiles); the shadow-mode milestone (§7, U2) measures it before the override
goes live.

## 4. Latency

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

## 5. Tests

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
- **Python:** the fork's wire-replay test
  ([fork-plan.md](fork-plan.md) §4) now runs **with `UNCLEJOE_TACTICS=0`**
  and must still show reply-for-reply equality — proving the NN path stays
  untouched under the tactics build. No new Python tests; the 15 s budget
  is untouched.

## 6. Verification and measurement

1. **Gate:** the same matchup command as [fork-plan.md](fork-plan.md) §5,
   re-run on the tactics build. Must reach a normal end; store the game
   under `data/games/<round>/` before any refit.
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
   mid-evaluation, resync both or neither
   ([fork-plan.md](fork-plan.md) §3).

## 7. Milestones

(U1, the fork itself, is [fork-plan.md](fork-plan.md) §6.)

- **U2 — memory + triggers, shadow mode.** `board/memory.rs` and both
  triggers in `tactics/`, logging fires to stderr, never overriding. Done
  when: memory and trigger unit tests pass and a gate match's stderr shows
  sane fire rates.
- **U3 — search + override.** The `search/` sub-package (`sim.rs`,
  `minimax.rs`), the `tactics/` orchestration and override in `act`,
  `UNCLEJOE_TACTICS=0`. Done when: all §5 tests pass and wire-replay
  equality still holds with tactics off.
- **U4 — latency proof.** `bench` on synthetic-long (local + Modal),
  worst-case-position cap test. Done when: p99 and max are recorded in a
  `docs/bots/unclejoe/latency.md` alongside the cap constants.
- **U5 — measurement.** §6's round, then the replication round; verdict
  quoted per the decision rule; `update-leaderboard`.
