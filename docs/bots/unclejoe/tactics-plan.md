# unclejoe tactics plan

Plan for the tactics layer of `bots/unclejoe/`. Status: **plan only** — no
tactics code exists yet. The prerequisite is met: the fork is finished per
[fork-plan.md](fork-plan.md) §7 (milestone U1, 2026-08-15 —
behavior-identical to joe-rs, gate passed, wire-replay equality proven), and
the spec is
[`../../research/strategies/unclejoe.md`](../../research/strategies/unclejoe.md).
Work starts at U2 (§7).

Revised 2026-08-15 after a design review. The goal changed: **spend the whole
150 ms turn budget** (RULES.md §08), not finish the move early. The
trigger-gated override stays; two mechanisms join it so the ~100 ms the
pipeline leaves unused does work every turn. Three mechanisms, strictly
layered:

1. **Proof-gated override** (kill, immediate defense) — the bounded exact
   search from the first revision, unchanged in semantics; its deadline
   rises from 20 ms to 60 ms.
2. **Exact-rule candidate filters** — masks computed from RULES.md
   arithmetic: two castle-build filters (crowding surcharge, lateness) and a
   1-ply refutation veto. Filters remove candidates; they never rank them.
3. **Afterstate value re-rank** — one-step policy improvement: the policy's
   top-k candidates, each advanced one ply under opponent-pass through the
   forward model, scored by the network's **own value head**; the best value
   plays. An anytime loop against an internal deadline spends the remaining
   clock.

Every move played is one of: the NN argmax, a top-k policy candidate
promoted by the network's own value head, or a proof-backed override. No
hand-written evaluation function ranks a move.

## 1. The new modules

The layering DAG stays `xla_math io → board → search → tactics → nn → main`
(as in joe-rs, a module may name the ones above it and must not name the
ones below it; `nn/` never learns the new modules exist). `search` depends
on `board` only; `tactics` reads `board::memory` and drives `search`; `main`
owns the memory update, calls `tactics`, and is the only place that runs
`nn` forwards on afterstates — the re-rank orchestration lives in
`Seat::act`, so `tactics` never names `nn`.

- `board/memory.rs` — persistent per-game memory the wire frame alone can't
  give, unchanged from the first revision. `enemy_general: Option<usize>`
  (set on first sighting; a general never moves, so it never goes stale),
  `mountains: Vec<bool>` (ever-seen, a separate copy from `AugState`'s),
  `general_visible_now: bool`, `last_seen_general_army: i32`,
  `last_seen_turn: i32` for the fog bounds, and — added in U2 —
  `first_contact_turn: Option<i32>`, the turn we first saw any enemy cell,
  latched, which arms the defense trigger (§3).
- `search/sim.rs` — the local forward model over a full ≤21×21 patch of
  `(passable, owner ∈ {me, opp, neutral, unknown}, army: i32)` built from
  the frame plus `board::memory`, implementing the engine's exact
  resolution: simultaneous moves with the §02 priority ladder (chasing >
  reinforcing > smaller-army-first, ties falling through); combat per §05
  (strictly-more captures, tie keeps the defender, mover leaves ≥1, full or
  half split); growth per §04 by `obs.turn` parity; deathtouch from turn
  800 (§07) with its chase defense; both generals captured same turn =
  draw, and a draw counts as failure for both the kill proof and the
  defense proof. It doubles as the **afterstate engine**: advancing one of
  our moves with the opponent passing is a plain call into the same
  resolution code.
- `search/minimax.rs` — the proof search, unchanged: depth-limited minimax
  over *(our move, their reply)* pairs, windowed move generation (own cells
  with army > 1 within Manhattan `depth + 1` of the target general; enemy
  moves from every enemy-*possible* cell in the window, fogged cells
  included under the pessimistic model, plus pass), returning a move **only
  when the outcome is proven for every opponent reply at every ply**. New
  in this revision: `refutes()` — a depth-1 exact check that asks whether
  any *visible* opponent reply captures our general after a given candidate
  move. Sound (it quantifies over every visible reply from the true root)
  and cheap.
- `search/afterstate.rs` — **new, and the biggest implementation item.**
  Candidate move → `sim` advance under opponent-pass → render the resulting
  position as a hypothetical wire frame (visibility per §06 recomputed from
  the resulting ownership) → advance a **cloned** `AugState` → the
  augmented observation the forward consumes. None of this path is covered
  by the joe-rs parity corpus, so §5 gives it its own fixtures.
- `tactics/triggers.rs` — the two trigger predicates, unchanged (§3).
- `tactics/filters.rs` — **new**: the castle masks and the refutation-veto
  plumbing (§3).
- `tactics/mod.rs` — orchestration: evaluate triggers against the frame and
  a read-only `&Memory`, invoke `search` with the matching goal, and return
  either a proven override or the filtered candidate list for `Seat::act`
  to evaluate. Tactics never mutates memory; the memory update is
  `Seat::act`'s job (§2).
- **Fog, pessimistically** — unchanged from the first revision, for the
  proof search only. `hidden = opp_army − Σ(visible enemy army)` bounds
  every never-seen-mountain fogged cell independently. Pessimism makes
  defense proofs harder (correct direction) and kill proofs land only
  against last-seen general army + growth + worst-case resolved-first
  reinforcement. Every cell 8-adjacent to an owned tile is visible (3×3
  pool), so 1-ply threats to our general are always in view. The
  pessimistic model is a **proof device, not a predictor**: it is never
  used to score re-rank lines.

Copied-file edits: the first revision promised zero. The afterstate work
may need **mechanical** edits to copied `board` files — a
`#[derive(Clone)]`, a `pub` — to clone-and-advance `AugState` without
mutating the live one. Rule amended: no *behavioral* change to any copied
file; every mechanical edit is listed in this section when it is made. The
wire-replay equality test (§5) is the guard that "mechanical" stays true.

Mechanical edits made so far:

- `board/mod.rs` (U2) — `pub mod memory;` and one sentence of module doc
  naming it. No code path changed.

## 2. Per-turn flow in `Seat::act`

Stamp `t0` when the frame arrives; everything below runs against
`TURN_DEADLINE_MS = 130` — 20 ms of slack under the 150 ms limit for host
jitter and reply emit.

1. Update `board::memory` from the fresh frame — unconditionally, every
   turn, exactly like `AugState`: per-game state advances regardless of who
   chooses the move and regardless of any kill-switch.
2. Run the full existing pipeline unchanged — frame→raw, masks,
   `augment_obs`, forward, argmax — and additionally read the top
   `TOP_K = 5` legal candidates in policy order and the root value. Never
   skip the forward.
3. Evaluate the triggers. On a fire, run the proof search
   (`PROOF_DEADLINE_MS = 60`, `PROOF_NODE_BUDGET = 300,000`). A proof
   overrides everything and the turn ends. A decline falls through to
   step 4 with whatever clock is left.
4. Re-rank. For each candidate in policy order: drop it if a filter masks
   it (castle rules) or `refutes()` it; otherwise build the afterstate, run
   the forward, record the value. Stop starting evaluations when the
   remaining clock is under `RERANK_RESERVE_MS = 25`. Play the best-valued
   surviving candidate; with none evaluated yet (cutoff before the first
   forward), play the first surviving candidate in policy order; if every
   candidate is filtered, the filters are moot — play the argmax.
5. Emit.

The loop is anytime by construction: the argmax is candidate #1, so an
early cutoff at any point degrades to exactly the old bot. Everything stays
inside the existing `catch_unwind`; a tactics panic degrades to a pass, not
a forfeit.

Kill-switches: `UNCLEJOE_TACTICS=0` is the master — pure pass-through,
identical to joe-rs, used by the wire-replay equality test. Three
per-mechanism switches for A/B decomposition: `UNCLEJOE_OVERRIDE=0`,
`UNCLEJOE_MASKS=0`, `UNCLEJOE_RERANK=0`. With only `UNCLEJOE_RERANK=0` the
bot plays the argmax over unmasked candidates — the cheap standalone castle
mechanism.

## 3. Triggers and candidate filters

The two trigger predicates are unchanged from the first revision — exact,
conservative-superset, O(board-scan) integer checks over the parsed frame
plus memory. Range is `D` (= depth budget, ≤3) **moves**, measured by a
bounded BFS over every cell not remembered as a mountain: army travels one
orthogonal step per turn and mountains are the only permanently impassable
cells, so nothing that could arrive in `D` moves is excluded, and armies a
wall stands between are. **Kill**: `enemy_general` known, an own cell with
`army > 1` within `D` of it, and either `obs.turn ≥ 800` (deathtouch) or
neighborhood army `> last_seen_general_army` (a stale lower bound —
underestimating keeps the superset property).
**Immediate defense**: nothing before first contact (`first_contact_turn`
set), and then a visible enemy cell within `D` of our general with
`army ≥ army(g)`; or `obs.turn ≥ 800` and any visible enemy cell with
`army > 1` within `D`; or a fogged cell within `D` with `hidden ≥ army(g)`.
Firing costs only a search that then declines; missing a real case is the
failure mode, so both stay deliberately loose. U2 measured the rate before
anything overrode anything, and the contact precondition is what it bought:
ungated, the fog arm fired on **every turn before contact in every game**,
because `hidden` with no enemy ever seen is their whole army. That gate is
the layer's one non-proof assumption — exact only through turn 13, where
the ≥17-step spawn distance makes an enemy in range impossible — and
[shadow.md](shadow.md) records it as such.

The filters are new, and a different kind of object: exact arithmetic from
RULES.md that removes candidates and never ranks them.

- **Castle crowding** (§03): the build price is 35 plus
  `max(0, 14 − 2·d)` for each own structure at Manhattan distance `d`.
  Mask any build whose total surcharge exceeds
  `CASTLE_SURCHARGE_CAP = 8` — i.e. allow only builds clear of own
  structures (a single structure at `d ≥ 3` passes; anything closer, or two
  structures crowding, is masked).
- **Castle lateness** (§03 + §04): a castle produces 1 army per 2 turns, so
  payback takes `2 × price` turns. Mask any build after
  `CASTLE_LATE_TURN = 650`.
- **Refutation veto**: mask any candidate that `refutes()` proves loses the
  general to a visible reply next ply. This closes the first revision's
  known gap — the defense search declining and the bot then playing an NN
  move the search had already seen lose.

Both castle thresholds are uncalibrated guesses — the named risk of this
revision. They ship behind `UNCLEJOE_MASKS`, U4 counts them in shadow
before they go live, and revisions to the constants are strategist work
against shadow and round data.

## 4. Latency

The budget is spent, not avoided: the design target is a turn that rides
`TURN_DEADLINE_MS = 130` and never crosses 150 (RULES.md §08). The
guarantee is the anytime structure, not an estimate.

| Constant | Value | Role |
| --- | --- | --- |
| `TURN_DEADLINE_MS` | 130 | internal deadline; 20 ms slack for host jitter + emit |
| `PROOF_DEADLINE_MS` | 60 | proof-search wall clock, checked every 1,024 nodes |
| `PROOF_NODE_BUDGET` | 300,000 | clock-independent ceiling, scaled from 100k at 20 ms |
| `TOP_K` | 5 | candidates read from the policy (argmax is #1) |
| `RERANK_RESERVE_MS` | 25 | start no afterstate eval under this remainder |
| `CASTLE_SURCHARGE_CAP` | 8 | §3 |
| `CASTLE_LATE_TURN` | 650 | §3 |

All constants live in one place in the code and are quoted here and in the
spec; a change to them is a change to both documents.

Arithmetic sanity: the pipeline is measured at p99 23.7 ms / max 34.5 ms on
the Modal 1-core x86-64-v3 proxy ([latency.md](../joe-rs/latency.md),
2026-08-15), and a forward dominates it — so the spare clock holds about
four extra forwards, which is exactly what the re-rank spends it on; on a
trigger turn the 60 ms proof search substitutes for roughly three of them,
and a decline still leaves ~2 forwards of re-rank. Sim, filters, and
afterstate rendering are integer work, far below one forward.

unclejoe is now **maximally deadline-shaped**: every turn runs to the
clock. That is the class the replication rule binds hardest on (§6), and it
makes U5's bench mandatory rather than confirmatory: `unclejoe bench` over
the synthetic-long log, locally and on the Modal proxy, reporting p99
**and max** against 150; plus a Rust test that an artificially tiny
deadline degrades the loop to the argmax.

## 5. Tests

- **Rust (`cargo test`, zero cost to the Python budget):** in the new
  modules' unit tests, each beside its code —
  - `board/memory.rs`: enemy-general lock-in on first sighting, mountain
    accumulation, last-seen army/turn tracking through visibility changes;
  - each trigger: fires on a position with a real threat/kill at each depth
    1–3; stays silent with the general absent, out of range, behind
    remembered mountains, or pre-800 with insufficient army; the fog arm
    fires exactly when `hidden` crosses the threshold;
  - `search/sim.rs`: priority-ladder cases straight from RULES.md §02,
    tie-keeps-defender, growth parity, deathtouch and its chase defense;
  - `search/minimax.rs`: finds a forced 1-, 2-, and 3-ply kill; finds the
    saving chase/reinforce defense; **declines** when the "kill" fails
    against the best reply, when a fogged cell's hidden budget can defend,
    and when the node cap trips (cap set artificially low); `refutes()`
    flags a candidate that loses the general to a visible reply and passes
    a safe one;
  - `search/afterstate.rs`: its own fixtures, since the parity corpus never
    covers this path — hand-built positions where the rendered afterstate
    frame is checked cell-for-cell (army conservation, leave-one-behind,
    §06 visibility from the new ownership), and an `AugState`
    clone-advance test proving the live state is untouched;
  - `tactics/filters.rs`: the §03 price example (general + castle both at
    `d = 2` → 35 + 10 + 10 = 55) masked and a clear build passed; the
    lateness threshold; the everything-filtered fallback to argmax;
  - the anytime loop: a tiny deadline ⇒ argmax; a mid-loop deadline ⇒
    best-so-far.
- **Python:** the fork's wire-replay test
  ([fork-plan.md](fork-plan.md) §4) runs **with `UNCLEJOE_TACTICS=0`** and
  must still show reply-for-reply equality — the guard that the NN path and
  the mechanical copied-file edits (§1) changed nothing. No new Python
  tests; the 15 s budget is untouched.

## 6. Verification and measurement

1. **Gate:** the same matchup command as [fork-plan.md](fork-plan.md) §5,
   re-run on the tactics build. Must reach a normal end; store the game
   under `data/games/<round>/` before any refit.
2. **Contrast**, per [decision-rule.md](../../arena/decision-rule.md):
   baseline `A` = joe-rs's current registered hash (≥30 games, not
   provisional), candidate `B` = unclejoe, **both arms in the same round**.
   One round via `arena.tournaments.competition` with a shared ≥5-bot panel
   including `cm_expander`, pinned `--round-seed`, `--seat-policy
   alternate`, `--strict-versions`; ≥200 games/arm minimum, ~1150/arm for a
   ±25 CI. Read `fits["<round>"].delta(joe-rs, unclejoe)` and quote
   `Δ ± SE, CI₉₅, P(B>A)`, games per arm, verdict — never a rank.
   **Replication is mandatory, not advisory**: every turn now runs to the
   clock, so unclejoe is exactly the deadline-shaped class the replication
   rule exists for. Second, separately scheduled round; host state noted by
   hand.
3. **Decomposition:** a flat or negative headline result is ambiguous
   across three mechanisms. The switches make follow-up cheap: a
   decomposition arm is a one-line `run.sh` export
   (e.g. `UNCLEJOE_RERANK=0`), which forks the content hash into its own
   registered identity — intended; the registry records it. Each
   decomposition contrast runs inside one round, per the decision rule.
   The U4 shadow counters say which decomposition to run first.
4. Both arms must be on the **same network** — if joe re-exports
   mid-evaluation, resync both or neither
   ([fork-plan.md](fork-plan.md) §3).

## 7. Milestones

(U1, the fork itself, is [fork-plan.md](fork-plan.md) §7.)

- **U2 — memory + triggers, shadow.** `board/memory.rs` and both triggers
  in `tactics/`, logging fires to stderr, never overriding. Done when:
  memory and trigger unit tests pass and a gate match's stderr shows sane
  fire rates. **Shipped 2026-08-15**: 20 new unit tests (38 total, all
  passing), four gate matches and the corpus replay measured in
  [shadow.md](shadow.md), wire-replay equality still holding. The
  measurement changed the defense trigger inside the milestone — it now
  waits for first contact (§3) — which is the one place U2 traded a superset
  for an assumption. Two more things U3 inherits: the tactics constants
  block is `src/tactics/mod.rs`
  (`SEARCH_DEPTH`, `DEATHTOUCH_TURN` so far), and the triggers measure range
  as **moves over cells not remembered as mountains**, not raw Manhattan
  distance — still a superset, since mountains are the only permanently
  impassable cells, and tighter than counting armies a wall stands between.
  One open item goes to U3: even gated on contact, the fog arm fires on 88%
  of turns in one of the four gate matches, so the defense search's *decline*
  path is the hot path in a game shaped like that one.
- **U3 — proof search + override live.** `search/sim.rs`,
  `search/minimax.rs` (including `refutes()`, not yet wired to anything),
  the `tactics/` orchestration and override in `act`, the
  `UNCLEJOE_TACTICS` master and `UNCLEJOE_OVERRIDE` switches. Done when:
  §5's sim, minimax, and trigger tests pass and wire-replay equality holds
  with tactics off.
- **U4 — afterstates + shadow re-rank.** `search/afterstate.rs`,
  `tactics/filters.rs`, the forwards wired in `Seat::act` **logging
  only**: play the argmax, log per-turn candidate values, the value gap
  between the argmax's afterstate and the best rival, would-change /
  would-mask / refutation counts. Done when: the afterstate fixtures pass
  and shadow stats over the synthetic-long corpus plus one fresh gate match
  are written to `docs/bots/unclejoe/shadow.md` with an explicit
  **go/no-go**: live re-rank requires candidate value gaps that stand clear
  of the near-zero mass of the gap distribution. A no-go ships U3 + masks
  only and leaves the budget unspent — an acceptable outcome, not a failed
  milestone.
- **U5 — re-rank live + latency proof.** The anytime loop live behind
  `UNCLEJOE_RERANK` / `UNCLEJOE_MASKS`; `bench` on synthetic-long, locally
  and on the Modal proxy; the tiny-deadline degradation test. Done when:
  p99 and max are recorded in `docs/bots/unclejoe/latency.md` under 150 ms
  with the §4 constants quoted, and wire-replay equality still holds with
  tactics off.
- **U6 — measurement.** §6's round, the mandatory replication round, then
  decomposition if the headline is flat; verdict quoted per the decision
  rule; `update-leaderboard`.
