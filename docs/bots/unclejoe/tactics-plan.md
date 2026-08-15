# unclejoe tactics plan

Plan for the tactics layer of `bots/unclejoe/`. Status: **U2, U3 and U4 are
built** (§7 records what each one shipped and what it changed about this
plan); U5 is next, and U4's go/no-go changed its shape. The prerequisite was
the fork, finished per [fork-plan.md](fork-plan.md) §7 (milestone U1,
2026-08-15 — behavior-identical to joe-rs, gate passed, wire-replay equality
proven), and the spec is
[`../../research/strategies/unclejoe.md`](../../research/strategies/unclejoe.md).
Measurements live in [shadow.md](shadow.md).

**The build in the tree today must not be rated.** U4 wired the afterstate
re-rank in shadow, and the forwards it spends put p99 at 143 ms and max at
181 ms against RULES.md §08's 150 (§7, U4). `UNCLEJOE_RERANK=0` restores U3's
cost.

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

- `board/memory.rs` — persistent per-game memory. **What lives here is
  decided by permanence, not by availability**: a fact a rule keeps true for
  the whole game is settled once and read afterwards, never re-derived from
  each frame, even when the frame states it. So: `own_general` and
  `enemy_general: Option<usize>` (a general never relocates and a capture
  ends the game, §05/§07), `mountains: Vec<bool>` (ever-seen, permanent by
  §01, a separate copy from `AugState`'s), `first_contact_turn:
  Option<usize>` (an event that cannot un-happen; it arms the defense
  trigger, §3), plus the one *aging* pair — `last_seen_general_army: i32`
  with `last_seen_turn: i32`, carried together so a bound can be aged — and
  `general_visible_now: bool`. Anything a turn can change stays a frame read
  (`tactics/common/frame.rs`), because caching it would mean inventing an
  invalidation rule, and a stale value inside a proof is a wrong answer
  rather than a slow one. Remembered **castles** join this list at U3: they
  are permanent by §03 and the local sim needs them.
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
- `tactics/kill.rs`, `tactics/defense.rs` — **one file per tactic**, which
  is a change from this plan's first shape (a single `triggers.rs` holding
  both predicates). Each file owns its trigger predicate (§3), the result
  type it returns, its tests, and — from U3 — the `search` call that the
  predicate gates. The two never call each other; they meet in `mod.rs`.
- `tactics/common/` — everything neither tactic owns alone, and the only
  implementation of each question it answers: `reach.rs` is distance in
  **moves** over cells not remembered as mountains, `frame.rs` is what a
  frame states about the army it does not show (the hidden-army bound). Both
  feed `search` in U3, but as **arguments rather than a dependency**: the DAG
  above puts `search` below `tactics`, so `tactics` computes the window and
  the fog bound with these helpers and hands them to the search, which
  re-derives neither. The
  rule that keeps it small: a helper one tactic needs stays in that tactic's
  file; a helper that encodes a *rule* comes here, because two callers
  reading one rule two ways is the failure this module prevents.
- `tactics/filters.rs` — **new**: the castle masks and the refutation-veto
  plumbing (§3).
- `tactics/mod.rs` — the composition root: the constants block, the shared
  reach scratch, the shadow counters, and the orchestration — run the
  tactics against the frame and a read-only `&Memory`, invoke `search` with
  the matching goal, and return either a proven override or the filtered
  candidate list for `Seat::act` to evaluate. Tactics never mutates memory;
  the memory update is `Seat::act`'s job (§2).
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
- `board/obs.rs` (U4) — `pub` on `BUILD_BASE_COST`, so `tactics/filters.rs`
  can read the §03 crowding surcharge as `cost − BUILD_BASE_COST` off the
  pipeline's own grid rather than spelling 35 a second time. No code path
  changed. The `AugState` clone the first revision expected turned out to be
  unnecessary — `augment_obs` takes the live state by shared reference and
  writes a caller-supplied scratch, so the afterstate hands it its own and the
  type system already proves the live one is untouched.

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

U4 counted them: over 4,037 turns the three filters removed **49** candidates
(33 crowding, 1 lateness, 15 refutation) and removed the *argmax* — the only
case where a mask can change the played move — on **9**. The mechanism is
correct and almost never load-bearing, and the rate is lumpy: one gate seed
built nine castles and took 30 of the 33 crowding masks while two others built
two and none and took zero ([shadow.md](shadow.md) §U4).

The refutation veto also acquired an exact gate in U4, which is why it costs
nothing on a quiet turn: `refutes` is depth one and the only reply that takes
a general is a move onto it, so a refutation needs an enemy cell orthogonally
adjacent to ours holding an army that can move. Our own candidate cannot
create one. A four-cell check therefore decides whether the veto runs at all.

## 4. Latency

The budget is spent, not avoided: the design target is a turn that rides
`TURN_DEADLINE_MS = 130` and never crosses 150 (RULES.md §08). The
guarantee is the anytime structure, not an estimate.

| Constant | Value | Role |
| --- | --- | --- |
| `TURN_DEADLINE_MS` | 130 | internal deadline; 20 ms slack for host jitter + emit |
| `PROOF_DEADLINE_MS` | 60 | one proof search's wall clock, checked every 1,024 nodes |
| `PROOF_NODE_BUDGET` | 300,000 | clock-independent ceiling, scaled from 100k at 20 ms |
| `SEARCH_DEPTH` | 3 | `D`: the kill proof's horizon, and the trigger scan radius |
| `DEFENSE_DEPTH` | 1 | the defense proof's horizon (U3, §7) |
| `FOG_GROWTH_MARGIN` | 3 | added to the hidden bound so an aged bound is still a bound (U3, §7) |
| `TOP_K` | 5 | candidates read from the policy (argmax is #1) |
| `RERANK_RESERVE_MS` | 25 | start no afterstate eval under this remainder |
| `CASTLE_SURCHARGE_CAP` | 8 | §3 |
| `CASTLE_LATE_TURN` | 650 | §3 |

`PROOF_DEADLINE_MS` is per search, not per turn: both triggers can fire on one
turn, and a kill search that runs to its deadline must not starve the defense
proof behind it. Each search takes `min(now + 60 ms, t0 + 130 ms)`, so the
worst trigger turn is one forward plus 60 ms plus the remainder — still inside
the internal deadline.

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

**U4 measured this table and two of its rows are wrong** ([shadow.md](shadow.md)
§U4). `RERANK_RESERVE_MS = 25` is smaller than the thing it reserves for: one
evaluation is a whole forward, 22 ms typical and 48 ms at joe-rs's own max on
the dev host. So the anytime structure bounds where the loop stops *starting*
work and not where it stops working, and the guarantee above has a hole
exactly that wide — measured at p99 143 ms, max 181 ms. `TOP_K = 5` is the
second: five afterstates plus the live forward is ~138 ms before any
rendering, so the loop cut off on 4,008 of 4,037 turns and averaged 3.9
forwards. The arithmetic below counted the spare clock correctly and then
spent it as though the reserve were free. A reserve must exceed the **worst**
cost of one evaluation, or the loop must reserve what its own last evaluation
took; and `TOP_K` should be chosen on the Modal proxy, not here.

## 5. Tests

- **Rust (`cargo test`, zero cost to the Python budget):** in the new
  modules' unit tests, each beside its code —
  - `board/memory.rs`: both generals settled once and never displaced —
    including ours surviving a frame that omits it — mountain accumulation,
    first contact latching, last-seen army/turn tracking through visibility
    changes;
  - each tactic, in its own file: fires on a position with a real
    threat/kill at each depth 1–3; stays silent with the general absent, out
    of range, behind remembered mountains, or pre-800 with insufficient
    army; the fog arm fires exactly when `hidden` crosses the threshold, and
    not at all before first contact;
  - `tactics/common/`: the step budget, walls that must be gone around and
    never through, a sealed cell; the hidden-army remainder including its
    zero floor;
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
  fire rates. **Shipped 2026-08-15**: 25 new unit tests (43 total, all
  passing), one file per tactic (§1), four gate matches and the corpus
  replay measured in
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
  `search/minimax.rs` (including `refutes()`), the `tactics/` orchestration
  and override in `act`, the `UNCLEJOE_TACTICS` master and
  `UNCLEJOE_OVERRIDE` switches. Done when: §5's sim, minimax, and trigger
  tests pass and wire-replay equality holds with tactics off.
  **Shipped 2026-08-16**: 74 unit tests, all passing — 31 new, being 17 on the
  forward model, 12 on the search, and 2 on the action codec between the layer
  and the wire. Wire-replay equality holds over 14 games and 7,092 turns with
  `UNCLEJOE_TACTICS=0`, and four gate matches are measured in
  [shadow.md](shadow.md). No mechanical edit to
  a copied file was needed after all — the afterstate clone (§1) is U4's
  problem, and U3 needed none. Five things U4 inherits:
  - **`search` does not name `tactics`.** §1's DAG and §1's `common/` bullet
    disagreed about which way the U3 dependency would run; the DAG won.
    `tactics` computes the window with `common::reach` and the fog bound with
    `common::frame`, then passes both into the search as arguments. One
    implementation of each rule, no upward edge.
  - **`refutes()` is wired, as the defense override's gate rather than as a
    filter.** A defense proof alone answers "does *some* action survive?",
    which is *yes* on nearly every quiet turn — an override on that answer
    would swap the network's move for an arbitrary safe one. So the override
    needs two facts: the argmax **provably loses** the general (`refutes`, on
    the visible-only board, so every reply counted is one the opponent really
    has) and another action **provably does not** (the pessimistic proof).
    U4's veto use — masking candidates — is unchanged and still to come.
  - **The defense proof's horizon is one ply** (`DEFENSE_DEPTH = 1`), not
    `SEARCH_DEPTH`. At one ply the pessimistic model costs nothing, because
    vision is the 3×3 pool around owned cells (§06) and every cell that can
    reach our general in one move is therefore lit. At two it costs
    everything: a fogged cell two steps out carries the whole unaccounted
    army, so almost nothing is provable and the search would spend the clock
    proving it. The trigger still scans `D = 3`, a superset; the extra fires
    cost one `refutes()` call each.
  - **Three model assumptions**, stated in `search/sim.rs` and
    `search/minimax.rs` rather than buried: opponent **builds are modeled as
    a pass** (a build is a pass that also spends 35+ army off one of their own
    cells, so it is dominated by a pass except for the new castle's ≤1 army of
    production inside the horizon); **replies from outside the window are
    modeled as a pass** (they cannot reach the general inside the horizon and
    they commute with our move, with a residue this search does not see — a
    source just outside that walks in and interferes at a later ply); and
    passability is **asymmetric** — `TYPE_STRUCTURE_IN_FOG` is closed to us
    and open to them, which is the exact reading of the two fog codes and not
    an assumption at all. That last point is also why the owner field ended up
    three-valued rather than the four §1 proposed: `unknown` is not a state
    the resolution code ever has to reason about, because a fogged cell is
    *materialized* at construction — enemy-owned, holding the bound — and the
    two passability masks carry everything the fourth value would have said.
  - **Every emitted override is a legal action, by construction.** Vision is
    the 3×3 pool around owned cells, so every neighbour of a cell we own is
    visible; the destination of any move the search generates from our own
    cells is therefore known exactly, and never a guess about fog.
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
  **Shipped 2026-08-16, and the go/no-go came back NO-GO.** 99 unit tests,
  all passing — 25 new. Four gate matches and the corpus replay measured in
  [shadow.md](shadow.md): 4,037 re-ranked turns, 15,743 afterstate forwards,
  every match ending on the same turn with the same winner as U2 and U3, and
  wire-replay equality still holding over 14 games and 7,092 turns. Five
  things U5 inherits, and the first two are why this milestone did not
  become U5:
  - **The value gaps sit inside the head's own resolution.** The re-rank
    would move the reply on 48.8% of turns, and on 83% of those the gap is
    under **one bin** of the value head — which is 128 bins over [−1, +1],
    so 0.0157 wide. Only 8.1% of turns clear one bin and 3.3% clear two. A
    difference the head cannot represent is not a preference, so H2 is not
    supported as specified. The live re-rank does not ship. A **minimum-gap
    threshold** would act on the 8% instead of the 49%, and that is the
    strategist revision this data argues for.
  - **`RERANK_RESERVE_MS = 25` is smaller than one evaluation.** An
    evaluation is a whole forward — 22 ms typical, 48 ms at joe-rs's own max
    on the dev host — so the anytime structure bounds where the loop stops
    *starting* work, not where it stops working. Measured: p99 143 ms, max
    181 ms, over the 150 ms limit. §4's "the guarantee is the anytime
    structure, not an estimate" has that hole in it. The U4 build must not
    be rated; `UNCLEJOE_RERANK=0` restores U3's cost.
  - **`TOP_K = 5` is not a budget that exists.** The loop cut off on 4,008
    of 4,037 turns and averaged 3.9 forwards. Five afterstates plus the live
    forward is ~138 ms before any rendering. Four is what the dev host pays
    for; the Modal proxy is where the number should be chosen.
  - **The masks are real but rare.** 49 candidates removed over 4,037 turns,
    and the argmax on 9 of them. A mask-only contrast needs far more games
    than the plan's ~1150/arm to resolve, which changes what a decomposition
    arm on `UNCLEJOE_MASKS` can be expected to show (§6).
  - **The opponent's totals are carried, not recomputed.** Most of their army
    is in fog, so the rendered afterstate frame carries the base frame's
    `opp_land` / `opp_army` and applies only the deltas it can see. The
    residue — production from enemy castles under fog — is *the same for
    every candidate on a turn*, and a re-rank compares candidates within one
    turn, so it cancels. `search/afterstate.rs` states the argument where the
    approximation is made.
- **U5 — re-rank live + latency proof.** The anytime loop live behind
  `UNCLEJOE_RERANK` / `UNCLEJOE_MASKS`; `bench` on synthetic-long, locally
  and on the Modal proxy; the tiny-deadline degradation test. Done when:
  p99 and max are recorded in `docs/bots/unclejoe/latency.md` under 150 ms
  with the §4 constants quoted, and wire-replay equality still holds with
  tactics off. **U4's no-go changes this milestone's shape**: the re-rank
  does not go live as designed, so U5 is either (a) masks-only, with the
  re-rank compiled but switched off, or (b) a revised re-rank — minimum-gap
  threshold, a reserve that covers a whole forward, and a `TOP_K` chosen on
  the proxy — which is a new spec and a new U4-shaped measurement before it
  is a U5. Either way the latency numbers are still mandatory, because the
  U4 build as measured does not hold 150 ms.
- **U6 — measurement.** §6's round, the mandatory replication round, then
  decomposition if the headline is flat; verdict quoted per the decision
  rule; `update-leaderboard`.
