# Morpheus-rs refactor plan — a flat module list into real subpackages

Status: **proposed, not started (2026-08-10).** Nothing under `crates/**` has
moved. This is a plan for a **pure move**: files change place, code does not
change meaning. Every stage is separately shippable and separately revertible.

Scope: `bots/morpheus-rs/crates/core`, 30 modules and 17,261 lines behind a
flat `lib.rs`, four of them large enough that "which file is this in" has
stopped being a useful question — `tactics.rs` (2,560), `parity.rs` (1,755),
`runtime.rs` (1,327), `search.rs` (1,057).

It engages with [`rewrite-plan.md`](rewrite-plan.md) §6, §10 and §18 and with
[`packaging.md`](packaging.md), because several of the current choices are
argued there and two of them constrain what this refactor may do.

## 0. What the current layout already got right, and what it did not

The flat list is not an accident of neglect; it is what a milestone-ordered
port produces. M1 landed `state`/`transition`/`action`/`observe`, M2 added
`hashing`/`memory`/`symmetry`/`tensor`, M3 `network`/`gemm`/`inference`, and so
on — each milestone appended files to a directory and a line to `lib.rs`, and
at no point was there a reason to stop and group. The *grouping already
exists*; it is recorded in the rewrite plan's milestone numbers and in
`tools/mutation_check.py`'s surface map, and nowhere in the tree.

Two things the current layout gets right and this plan keeps:

- **The crate has no dependencies** and the module names mirror the Python
  oracle's module names one-for-one (`tactics.py` → `tactics.rs`). That
  one-for-one naming is a parity asset: when a surface disagrees, the file to
  read on each side has the same name. Any regrouping has to preserve the
  *leaf* names even as it changes their parents.
- **The two-crate workspace** (`crates/core` lib + `crates/bot` bin) is load
  bearing: `crates/bot` is what `[[bin]]`, `run.sh`, `build.sh` and the
  packager all name, and `crates/core` is what `cargo test` and the parity
  subcommands exercise.

What it does not get right is everything above the leaf: there is no place to
put the fact that `state`/`transition`/`action`/`observe`/`memory` are one
layer, no compiler-visible statement that `tactics` may not call `runtime`, and
no seam inside the four big files even though the parity harness has been
treating their parts as separate surfaces since M5.

## 1. Proposed tree

**Decision: module subdirectories inside `crates/core`. Not several workspace
crates.** The workspace stays exactly two crates.

```
bots/morpheus-rs/crates/core/src/
  lib.rs                    module list + the layering rule, nothing else

  support/                  primitives the crate implements itself because it
    mod.rs                  has no dependencies — the budget's visible cost
    rng.rs
    sha256.rs

  io/                       the two things the process reads from outside
    mod.rs                  itself: the judge's frames, and files on disk
    wire.rs
    json.rs

  board/                    the deterministic game layer — bit-exact against
    mod.rs                  the oracle, no floats, no policy (M1/M2)
    state.rs
    transition.rs
    action.rs
    observe.rs
    memory.rs
    symmetry.rs
    hashing.rs

  nn/                       weights in, eleven heads out; the artifact's
    mod.rs                  contract and the kernels that serve it (M3)
    gemm.rs
    tensor.rs
    network.rs
    inference.rs
    safetensors.rs

  belief/                   what the bot thinks the fog contains (M4)
    mod.rs                  (particles, weights, ESS, resampling)
    proposal.rs
    recovery.rs
    reservoir.rs
    summary.rs

  tactics/                  hand-written rules; split along the five parity
    mod.rs                  surfaces the harness already separates
    params.rs
    geometry.rs
    pathing.rs
    query.rs
    weights.rs
    oscillation.rs
    play_mask.rs
    seek.rs
    castle.rs
    defense.rs
    kill.rs
    candidates.rs
    shaping.rs
    constrain.rs

  search/                   the tree and what fills it
    mod.rs
    tree.rs
    matrix.rs
    controller.rs
    select.rs
    backup.rs
    evaluator.rs

  runtime/                  the deadline: what to attempt and what to skip
    mod.rs
    config.rs
    clock.rs
    estimator.rs
    degrade.rs
    metrics.rs
    controller.rs
    deployment.rs
    telemetry.rs

  parity/                   the harness half of the binary; never plays
    mod.rs
    ints.rs
    codec.rs
    bench.rs
    surfaces/
      mod.rs
      board.rs
      net.rs
      numpy.rs
      belief.rs
      tactics.rs
      search.rs
```

One line each for why the grouping is that grouping:

| group | why these files are one thing |
| --- | --- |
| `support/` | Both exist only because the dependency budget declined `sha2` and `rand`; putting them together makes that cost one directory instead of two scattered files. |
| `io/` | Format codecs with zero game knowledge — the wire protocol and the JSON the config, manifest and telemetry are written in. |
| `board/` | The competition transition and everything bit-exact that depends only on it; the layer whose parity tolerance is "zero". |
| `nn/` | Everything defined by the trained artifact rather than by the rules: the 49-plane input contract, the graph, the kernels, the container. |
| `belief/` | The particle filter and the three things only it uses — proposal, recovery, reservoir — plus the summary that hands it to `nn::tensor`. |
| `tactics/` | The hand-written rules. Split five ways along `playmask`/`candidates`/`planners`/`shaping`/`constrain`, because the parity harness and the mutation map already treat those as separate surfaces. |
| `search/` | The tree, the regret math, the batched simulation loop, and the evaluators plugged into it. |
| `runtime/` | Everything that exists because there is a 150 ms judge limit: knobs, clock, estimators, degradation, metrics, and the controller that spends the budget. |
| `parity/` | The one group with no play-time role at all — the reason it is a directory is so that is obvious from the tree. |

### Why subdirectories and not workspace crates

The crate graph *would* work: the current `use crate::…` edges form a clean DAG
(`support`/`io` → `board` → `nn` → `belief` → `tactics` → `search` → `runtime`,
with `parity` above everything). A crate split is therefore possible. It is
still the wrong choice here, for four reasons in descending weight:

1. **A module move is codegen-neutral; a crate split is not, and this crate has
   already been burned by that.** M3 (§18) ends with a measurement that would
   not sit still: the depthwise stage read 1.10 ms for most of the milestone
   and 0.28 ms afterwards, from edits *elsewhere in the crate*, with three
   candidate causes tested and refuted and whole-crate LTO left as the only
   remaining explanation — unproven. `release` is `lto = "fat"`,
   `codegen-units = 1`. Moving code between modules of one crate cannot change
   what the optimizer sees; moving it across a crate boundary changes
   instantiation sites, inlining candidates and the LTO unit's shape. A refactor
   whose entire claim is "nothing changed" should not be the thing that
   re-opens M3's open question, and a latency regression here is not
   hypothetical — M7's x86 qualification reads p99.9 = 140 ms against a limit of
   150.
2. **Crate boundaries force `pub` where `pub(crate)` is the honest answer.**
   Splitting means every helper that crosses a boundary becomes public API of
   its crate, and the compiler's dead-code and visibility analysis stops being
   able to tell "used by the bot" from "used by one sibling". Subdirectories
   *add* privacy tools rather than removing them: a child module can see its
   ancestors' private items, so `search/select.rs` can carry
   `impl SearchController` against private fields declared in `search/mod.rs`
   without widening anything. That single Rust fact is what makes the big-file
   splits below cost nothing in visibility.
3. **The usual payoff — parallel and incremental compilation — is not available
   here.** `codegen-units = 1` plus fat LTO serializes the expensive half of a
   release build regardless of crate count, and the release build is what
   `run.sh`, `run_parity.sh`, the packager and `mutation_check.py` all run.
   Debug builds would get faster; nothing in the loop uses them.
4. **It buys nothing the packaging path wants.** `package_submission.py` walks
   `SOURCE_TREES = ("crates",)` with `rglob`, so either shape ships; more
   crates means more `Cargo.toml` members and a `Cargo.lock` with more path
   entries, against a file budget already at 0.4% of its cap. There is no
   argument on this axis in either direction, which means the first three
   decide it.

The one real thing a crate split would give — *mechanical* enforcement that
`tactics` cannot call `runtime` — is worth having and is not worth a crate
graph. It is available at ~90% strength as a convention stated in `lib.rs` and
checked by review, and the layering is already true today with no enforcement
at all.

### mod.rs, not `foo.rs` beside `foo/`

Both are legal in edition 2021; pick one and be uniform. This plan picks
`tactics/mod.rs` so that every path in the mutation map, the docs and the
content-hash file list is `…/src/<group>/<leaf>.rs` with no exceptions — the
tooling names files a lot here, and one shape is worth more than the
editor-tab ergonomics of the alternative. This is a coin flip; what matters is
that it is not mixed.

## 2. Old → new mapping

All 30 modules. "split" means the file's contents are divided; every other row
is a `git mv` plus an import edit.

| current | lines | destination | note |
| --- | ---: | --- | --- |
| `lib.rs` | 42 | `lib.rs` | rewritten as nine `pub mod` lines plus the layering rule |
| `rng.rs` | 772 | `support/rng.rs` | move |
| `sha256.rs` | 249 | `support/sha256.rs` | move |
| `wire.rs` | 334 | `io/wire.rs` | move |
| `json.rs` | 325 | `io/json.rs` | move |
| `state.rs` | 101 | `board/state.rs` | move |
| `transition.rs` | 645 | `board/transition.rs` | move |
| `action.rs` | 211 | `board/action.rs` | move |
| `observe.rs` | 209 | `board/observe.rs` | move |
| `memory.rs` | 283 | `board/memory.rs` | move |
| `symmetry.rs` | 319 | `board/symmetry.rs` | move |
| `hashing.rs` | 268 | `board/hashing.rs` | move |
| `gemm.rs` | 226 | `nn/gemm.rs` | move |
| `tensor.rs` | 499 | `nn/tensor.rs` | move — it defines `BeliefSummary` and the 49-plane contract, both properties of the artifact, not of the board |
| `network.rs` | 917 | `nn/network.rs` | move, **not split** — see §3 |
| `inference.rs` | 362 | `nn/inference.rs` | move |
| `safetensors.rs` | 247 | `nn/safetensors.rs` | move |
| `belief.rs` | 758 | `belief/mod.rs` | move |
| `proposal.rs` | 499 | `belief/proposal.rs` | move |
| `recovery.rs` | 848 | `belief/recovery.rs` | move |
| `reservoir.rs` | 233 | `belief/reservoir.rs` | move |
| `particle_summary.rs` | 181 | `belief/summary.rs` | move + the only leaf rename in the plan (see §3) |
| `tactics.rs` | 2,560 | `tactics/` ×14 | **split** |
| `matrix.rs` | 353 | `search/matrix.rs` | move |
| `tree.rs` | 738 | `search/tree.rs` | move |
| `search.rs` | 1,057 | `search/` ×4 | **split** |
| `evaluator.rs` | 299 | `search/evaluator.rs` | move, merged with `search.rs`'s two evaluators |
| `runtime.rs` | 1,327 | `runtime/` ×6 | **split** |
| `deployment.rs` | 457 | `runtime/deployment.rs` | move |
| `telemetry.rs` | 187 | `runtime/telemetry.rs` | move — its only dependency is `runtime`, and the trace's key set *is* the controller's component list |
| `parity.rs` | 1,755 | `parity/` ×10 | **split** |

### `tactics.rs` → `tactics/` (14 files)

The seam is the five parity surfaces plus the shared helpers they all read.
Line ranges are from the tree at the time of writing and are there to make the
split reviewable, not to be trusted after the first edit.

| new file | from | ~lines | contents |
| --- | --- | ---: | --- |
| `params.rs` | 40–127 | 90 | every `pub const` — they are already one contiguous block at the top of the file, and they are the bot's tuning surface |
| `geometry.rs` | 128–260 | 135 | `Cell`, `DecodeTables`, `Grids`, `move_dest`, `move_segment`, `is_reverse_*`, `turn_of` |
| `pathing.rs` | 494–602 | 110 | `move_progress`, `path_progress`, `DistanceField`, `path_distance_field` |
| `query.rs` | 331–386, 603–795 | 250 | read-only board questions: visibility, generals, structures, `army_concentration`, `reveal_count_grid`, `believed_enemy_general` |
| `weights.rs` | 387–493 | 106 | the scalar shaping terms: `fog_urgency`, `wave_weight`, `attack_weight`, `tip_thrash_factor`, `stack_gather_factor`, `direction_bias` |
| `oscillation.rs` | 263–330 | 70 | `oscillation_history`, `blocks_oscillation` |
| `play_mask.rs` | 916–1140 | 225 | `garrison_floor`, `max_threat_arrival`, `apply_garrison_floor`, `apply_castle_anchor`, `play_mask` → surface `playmask` |
| `seek.rs` | 796–915 | 120 | `seek_goals`, `enemy_seek_target`, `wave_assembly_cell` |
| `castle.rs` | 1141–1272 | 130 | `castle_build_site`, `castle_tithe_move`, `opponent_mobile` — the castle savings pipeline |
| `defense.rs` | 1273–1389 | 117 | `general_threat`, `defend_general_move` — emergency defense |
| `kill.rs` | 1390–1508 | 120 | `kill_plan`, `winning_kill_move` — the kill-window planner |
| `candidates.rs` | 1509–1700 | 190 | the index sets, `mandatory_action_indices`, `policy_ordered_candidates` → surface `candidates` |
| `shaping.rs` | 1701–2158 | 460 | `heuristic_action_scores` (289 lines on its own), `blend_prior`, `apply_pre_contact_prior`, `best_prior_legal_action` → surface `shaping` |
| `constrain.rs` | 2159–2418 | 260 | `constrain_nn_action`, `capture_moved_army` → surface `constrain` |

`seek`/`castle`/`defense`/`kill` are the four planners the `planners` surface
covers; they are four files rather than a `planners/` sub-directory because
each is a named subsystem in the rewrite plan (§1) and a third level of nesting
buys nothing at ~120 lines apiece.

The 141 lines of `#[cfg(test)] mod tests` go with the function each test names.

### `search.rs` → `search/` (4 files, plus `evaluator.rs` merged in)

| new file | from | ~lines | contents |
| --- | --- | ---: | --- |
| `mod.rs` | 32–190, 1050–1057 | 190 | consts, `EvalItem`, `SearchConfig`, `PendingPath`, `EnemyPriorRequest`, `Selection`, `singleton_belief` |
| `controller.rs` | 192–500 | 310 | `SearchController::new`, root handling, enemy-prior cache, `materialize_enemy_priors`, `widen_enemy_if_needed` |
| `select.rs` | 500–830 | 330 | `select_path` and `complete_select_path` — the 300-line walk that is its own subject |
| `backup.rs` | 830–980 | 150 | `evaluate_leaves`, `backup_path`, `run_batch`, `best_action*` |
| `evaluator.rs` | 51–110, 982–1050 + all of `evaluator.rs` | 430 | the `SearchEvaluator` trait and all four implementations — `Uniform`, `Scripted`, `Network`, `ShapedUniform` — in one place instead of two |

`select.rs` and `backup.rs` carry `impl SearchController` blocks against fields
declared in `mod.rs`; no field or method visibility changes, because a child
module already sees its parent's private items.

### `runtime.rs` → `runtime/` (6 files)

| new file | from | ~lines | contents |
| --- | --- | ---: | --- |
| `mod.rs` | header | 40 | the module's doc comment and re-exports |
| `config.rs` | 37–100, 190–270 | 150 | `COST_COMPONENTS`, every `DEFAULT_*`, `RuntimeConfig`, `offline_seed_ms` |
| `clock.rs` | 141–190 | 50 | `Clock`, `MonotonicClock`, `FakeClock` |
| `estimator.rs` | 102–140, 271–330 | 100 | `FallbackLevel`, `nearest_rank_p99`, `NearestRankP99Estimator` |
| `degrade.rs` | 440–495 | 55 | `argmax_masked`, `highest_prior_legal`, `select_degraded_action` |
| `metrics.rs` | 330–440, 1187–1232 | 160 | `TurnMetrics`, `prior_probe_fields`, `publish_metrics` |
| `controller.rs` | 495–1187 | 650 | `RuntimeController` and its `decide` / `run_belief_update` / `run_search` / `materialize` |

### `parity.rs` → `parity/` (10 files)

| new file | from | ~lines | contents |
| --- | --- | ---: | --- |
| `mod.rs` | 613–624, 1690–1707 | 120 | `run()` — the `kind` match, now one line per arm |
| `ints.rs` | 54–140 | 90 | the `Ints` reader and `itoa` |
| `codec.rs` | 140–520 | 380 | every `read_*`/`write_*`/`push_*` for state, observation, memory, belief, particles, draws |
| `bench.rs` | 522–612 | 95 | `bench_belief` and `VaryingEvaluator` |
| `surfaces/board.rs` | 625–759 | 140 | `transition`, `order`, `observe`, `mask`, `cost`, `memory`, `hash`, `tensor`, `symmetry` |
| `surfaces/net.rs` | 759–861 | 100 | `net`, `prior`, `toplegal` |
| `surfaces/numpy.rs` | 861–892 | 35 | `npsum`, `argsort` — the two host-conditional oracle behaviours |
| `surfaces/belief.rs` | 892–1108 | 215 | `initbelief`, `summary`, `propose`, `filter`, `rejuvenate`, `maxent`, `reservoir` |
| `surfaces/tactics.rs` | 1108–1301 | 195 | `playmask`, `shaping`, `candidates`, `planners`, `constrain` |
| `surfaces/search.rs` | 1301–1690 | 400 | `matrix`, `decide`, `search`, `evict`, `runtime` |

**This is the only split that needs new code, and it is nine lines of it.**
Each match arm becomes
`pub(super) fn <kind>(ints: &mut Ints, out: &mut Vec<i64>, ctx: &mut Ctx) -> Result<(), String>`
with its body moved verbatim; `run()` keeps one match mapping `kind` to a
function. `run()` currently holds a `let mut net_session: Option<Session>`
outside the case loop, lazily filled by the `net` and `decide` arms so the
artifact is loaded once per subcommand rather than once per case. That becomes
`struct Ctx { net: Option<Session> }` threaded through the calls. It is the one
place in the whole refactor where the reviewer should read the diff rather than
trust the move, and it is why this is the last stage.

After the split the largest file in the crate is `network.rs` at 917 lines,
then `runtime/controller.rs` (~650), `recovery.rs` (848) and `rng.rs` (772) —
i.e. the crate's ceiling drops from 2,560 to under 1,000 without a single
behaviour edit.

## 3. What stays put, and why

**The two-crate workspace.** `crates/bot/src/main.rs` stays one file at that
path. It is named by `[[bin]] path`, by `run.sh`, by
`tools/submission/build.sh`, and by `packaging.md`; it is 498 lines of
composition root plus `selfcheck`, `bench` and the parity dispatch; and
splitting a composition root is how you
get a composition root you have to search for.

**`network.rs`, whole, at 917 lines.** It is the fifth-largest file and the
obvious next candidate, and it should not move. Its content is one forward
pass — twelve identical blocks and eleven heads — and M3's per-stage timings
are measured *inside* it against a build-sensitivity problem the milestone
never closed. Splitting the graph across files is a change nobody can prove
neutral by reading, and the gain is a 917-line file becoming three ~300-line
files that are always read together.

**Leaf module names.** `transition.rs` stays `transition.rs`, `tactics`' rules
keep the names the Python oracle gave them. The one exception is
`particle_summary.rs` → `belief/summary.rs`, because `belief::particle_summary`
stutters and the Python name survives in the parity surface (`summary`) either
way. If that costs a moment's grep, drop the rename — it is not worth an
argument.

**Inline `#[cfg(test)] mod tests`.** They stay inline, in the file whose
private items they reach, rather than moving to `crates/core/tests/`.
`mutation_check.py` also depends on the shape: it locates `#[cfg(test)]` in the
file it is mutating and refuses a mutation whose pattern only matches below it.

**`parity` inside the shipped binary.** It stays a plain module of
`crates/core`, compiled into `morpheus-rs` and shipped. Feature-gating it out
would shrink the zip and change what the judge builds; that is a packaging
change, not a move, and the archive is 1.04 MB against a 50 MB cap.

**`tools/` and its `submission/build.sh`.** `build.sh` must not sit beside
`run.sh` — `matchup.py::build_agent` executes any `build.sh` it finds there and
then crashes formatting its log line, which breaks the repo's own verification
gate (`packaging.md`, M0.5/M8). Nothing in `tools/` moves.

**`bots/morpheus-rs/tests/` as a Python pytest harness at the bot root.** It is
outside the content hash by the `tests/` rule, and moving it inside `crates/`
would put it back in.

**Zero dependencies, and the hand-written `sha256`/`gemm`/`json`/`rng`.**
Grouping them under `support/` and `io/` does not reopen the question of
whether they should be crates; see §6.

**`artifact/`, `deployment.json`, `.cargo/config.toml`, `rust-toolchain.toml`,
`Cargo.toml`, `Cargo.lock`** at the bot root. All are inside the content hash
or load-bearing for the offline build, and none has a layout problem.

## 4. Staged steps

Nine stages. Each one leaves the tree compiling, `cargo test` green, the full
parity corpus green, the mutation check at its recorded count, and the AGENTS.md
gate finishing. **Work may stop after any stage**; none depends on a later one.

Verification after *every* stage, in this order:

```bash
cargo build --release --manifest-path bots/morpheus-rs/Cargo.toml
```

```bash
cargo test --release --manifest-path bots/morpheus-rs/Cargo.toml
```

```bash
bots/morpheus-rs/tools/run_parity.sh
```

```bash
PYTHON=.venv/bin/python python competition-module/competition/matchup.py bots/morpheus-rs/run.sh bots/cm_expander/run.sh --mode competition --seed 0
```

`run_parity.sh` runs the full corpus *and* `mutation_check.py`, which is what
makes the gate self-defending here: a mutation whose file moved without its map
entry moving comes back `stale` and the tool exits non-zero (M4's rule, which
has now paid for itself twice). A stage that forgets its tooling edits fails its
own gate rather than silently degrading the harness.

Use `git mv` for every move so blame survives; the file-level history of
`tactics.rs` is a record of four milestones' worth of reasoning.

**Stage 0 — baseline, no moves.** Record, in the commit message or a scratch
note: the current `content_hash` of `bots/morpheus-rs`, the parity case count
and surface count, and the mutation tally (140/167 at M6). Everything after
this is measured against those three numbers. Confirm the four commands above
are green *before* touching anything, so a later red is unambiguous.

**Stage 1 — `support/` and `io/`.** Four files, no splitting: `rng`, `sha256`,
`wire`, `json`. Deliberately first and deliberately boring: these have the
highest fan-in in the crate, so this stage exercises every mechanism the later
ones need — the `use` sweep, `lib.rs`, `mutation_check.py`'s `FILE_SURFACES`
keys and per-mutation `file` fields, `run.sh`'s stamp picking up new paths, the
packager's `rglob`, and the content-hash change — on a diff small enough to
read in full.

**Stage 2 — `board/`.** Seven files, no splitting. The largest single `use`
sweep in the plan (`state`, `memory` and `wire` are named nearly everywhere).

**Stage 3 — `nn/`.** Five files, no splitting. Re-read the M3 latency numbers
after this one: it is the stage nearest the code whose measurements moved
under whole-crate LTO, and `warmup_ms` in the selfcheck is a free per-forward
probe (26 forwards, ~4.7 ms each on the M3 Pro). It should not move; check.

**Stage 4 — `belief/`.** Five files, one leaf rename.

**Stage 5 — `search/`.** The first split: `search.rs` into four, `evaluator.rs`
merged into `search/evaluator.rs`, `tree`/`matrix` moved in. Watch the merge —
see §5's note on mutation pattern uniqueness.

**Stage 6 — `tactics/`, in three sub-stages**, each independently green:

- **6a — the shared leaves.** `params`, `geometry`, `pathing`, `query`,
  `weights`, `oscillation` out; the five surface functions stay in
  `tactics/mod.rs` for now. This is where the fourteen-way split earns its
  keep or does not: if the leaves do not come out cleanly, stop here with a
  1,700-line `mod.rs` that is still better than the 2,560 it started as.
- **6b — the planners and the mask.** `play_mask`, `seek`, `castle`,
  `defense`, `kill`.
- **6c — the decision surfaces.** `candidates`, `shaping`, `constrain`.
  `tactics/mod.rs` ends as a module list plus re-exports.

47 of the 167 mutations name `tactics.rs`; each sub-stage re-points its share.

**Stage 7 — `runtime/`.** Six-way split, plus `deployment` and `telemetry`
moved in.

**Stage 8 — `parity/`.** Last, because it is the only stage with new glue
(§2's `Ctx`), the only one whose failure cannot affect the playing bot, and the
one whose value is lowest. It is also the easiest to drop entirely if the
appetite runs out: a 1,755-line `parity.rs` beside eight clean directories is
a perfectly defensible resting place.

**Stage 9 — docs, tooling comments, and the registry step.** §5's edit list,
then one lineage step for the whole refactor.

## 5. Blast radius outside the crate

### `tests/parity_cases.py` — three comments, no code

The Python harness is coupled to the **integer stream and the surface names**,
not to the module tree. It never spawns anything but
`morpheus-rs parity <kind>`, and `<kind>` does not move. The layouts it mirrors
positionally are the encodings, which this refactor does not touch.

- line 18 — `crates/core/src/parity.rs` → `crates/core/src/parity/`
- line 172 — `# --- serialization (mirrors crates/core/src/parity.rs)` →
  `parity/codec.rs`
- line 1176 — `read_draws in crates/core/src/parity.rs` → `parity/codec.rs`

If Stage 8 is skipped, this file needs no edit at all.

### `tools/convert_artifact.py` — one comment

Line 20 names `crates/core/src/network.rs` as where the graph lives →
`crates/core/src/nn/network.rs`. The converter reads the TorchScript artifact
and writes safetensors; nothing in it inspects the Rust tree.

### `tools/mutation_check.py` — the largest single edit in the plan

Two things, both mechanical, both gate-enforced:

1. **`FILE_SURFACES`** — 21 keys, each a bare filename joined as `SRC / name`.
   They become paths: `"transition.rs"` → `"board/transition.rs"`,
   `"tactics.rs"` → fourteen entries, and so on. `SRC / "tactics/shaping.rs"`
   resolves, so no code changes.
2. **167 mutation `file` fields**, distributed as: `tactics.rs` 47,
   `network.rs` 17, `search.rs` 11, `tree.rs`/`transition.rs`/`matrix.rs` 10
   each, `recovery.rs`/`belief.rs` 9 each, and a long tail. For a whole-file
   move this is a find-replace; for a split, each entry must be re-pointed at
   the file its `before` pattern actually landed in — and a wrong answer comes
   back `stale`, non-zero, named.

Two rules for this file:

- **Keep each new file's surface set identical to its parent's.** A fourteen-way
  `tactics/` split makes it tempting to narrow `constrain.rs` to
  `("constrain", "decide")`. Don't, in this pass: the map is documented as
  deliberately generous, and an over-narrow entry reports a *caught* mutation as
  a survivor. Tightening it is a separate, measured change.
- **Re-check pattern uniqueness wherever two files merge.** `replace(..., 1)`
  hits the first occurrence, and Stage 5 merges `evaluator.rs` into
  `search/evaluator.rs` — the only place in the plan where two files become
  one. `search.rs`'s eleven patterns must still match the site they were
  written for.

### Docs — about a dozen references

- `packaging.md` line 16 — the layout block's `crates/core/src/  the ported
  bot; wire, board, network, parity` becomes the nine-group listing.
- `parity-harness.md` lines 122, 676 (`crates/core/src/parity.rs` → `parity/`
  and, for 676's "add a kind" recipe, `parity/surfaces/<group>.rs` plus the
  `run()` arm in `parity/mod.rs`), 490 (`matrix.rs` → `search/matrix.rs`),
  695 (`transition.rs`, `observe.rs`, `action.rs`, `memory.rs`, `state.rs` →
  `board/…`).
- `telemetry.md` line 27 — `crates/core/src/telemetry.rs` →
  `crates/core/src/runtime/telemetry.rs`. Line 75's `main.rs` is unaffected.
- `inference.md` lines 93 (`crates/core/network.rs` plus `gemm.rs` → `nn/`)
  and 222 (`inference.rs` → `nn/inference.rs`).
- `search-and-tactics.md` line 49 — `matrix.rs` → `search/matrix.rs`.
- `docs/index.md` — one new line for this file.

**`rewrite-plan.md` is not edited.** Its §16–§23 are dated milestone records
that describe a tree as it was on the day; rewriting paths inside them would
falsify a history this repo has been careful to keep honest. It gets one
addition instead: a pointer in §12 ("Where things live") to this plan. Its two
live `.rs` references (`telemetry.rs`, `matrix.rs`) are inside milestone prose
and stay.

### `run.sh` and the packaging path

- **`run.sh`'s `source_stamp()` is already correct.**
  `find "$DIR/crates" -type f -name '*.rs'` recurses, so subdirectories are
  picked up with no edit. The stamp hashes each file's path alongside its
  contents, so the first build after each stage rebuilds — which is what you
  want, and which the gate exercises anyway.
- **`tools/package_submission.py` assumes nothing about layout.**
  `SOURCE_TREES = ("crates",)`, walked with `rglob("*")`, skipping any path
  with `target` in its parts; there is no file list, no expected count, and no
  flat glob. The only consequences are quantitative: `_minify_rust` runs one
  subprocess per `.rs`, so ~30 → ~65 files roughly doubles a packaging step
  measured in seconds, and the archive goes from 43 files to roughly 80 —
  against `MAX_FILES = 10,000`, i.e. 0.4% → 0.8% of the cap. `check_limits`
  needs no change.
- **`tools/minify/`** is a per-file stdin→stdout filter; more, smaller files is
  the case it already handles. Watch one thing: it parses and re-prints, and
  every new `mod.rs` will carry a `//!` header that `remove_docs` strips. The
  packaging smoke compiles and plays the minified output, so this is checked
  rather than assumed — run
  `python bots/morpheus-rs/tools/package_submission.py --force --gate` once at
  Stage 9.
- **`tools/dependency_budget.py`** has no relationship to this tree at all; its
  `crates` references are crates.io graphs.
- **`tools/submission/build.sh`** names `crates/bot` only through the workspace
  build; unaffected. Its `cd`-before-`cargo` rule (the FMA trap) is untouched.

### The content hash, and the artifact already playing on generals.bot

**Yes, the rename changes the bot's content hash, and it changes it on the
first stage.** `fingerprint.content_hash_for_dir` digests each source file's
**repo-relative path** followed by its content hash, so moving
`crates/core/src/rng.rs` to `crates/core/src/support/rng.rs` mints a new
identity for a byte-identical program. There is no way to avoid this and no
reason to want to: a hash that ignored paths would collide two different trees.

What follows from it:

1. **Each stage is a new lineage step** in
   `data/bot_versions/morpheus-rs.json` if it is registered. Nine steps for a
   refactor that changes no behaviour would be nine rows of noise. **Register
   once, at Stage 9**, with a note that steps 1–8 were pure moves — and run
   `run_parity.sh` before the registry step, as the rewrite plan requires of
   any morpheus-rs step.
2. **Existing rating data does not transfer.** Everything measured — M6's
   retracted round, M7's `n8-s32-b4-d8` x86 qualification, the parity numbers —
   is attached to `morpheus-rs@5456f5532cc2` / `@0bbe9a3170c3` /
   `@ef5a20484a38`. A round run under `--strict-versions` after the refactor
   compares a *different* content hash. Since the program is provably the same
   (the parity corpus and the mutation tally are the proof), the honest
   handling is to say so in the registry note rather than to re-measure — but
   **M7's owed replicated strength contrast should be run before or after the
   refactor, not across it.** Straddling it would put the two arms on two
   hashes and hand a future reader a question nobody needs.
3. **The submitted artifact is unaffected and stays reproducible only from git
   history.** `morpheus-rs@ef5a20484a38` is what generals.bot accepted on
   2026-08-10 and what is playing rated games. It is pinned by its registry
   entry (`closure_ref refs/bot-versions/ef5a20484a38`, with every file's blob
   and sha256) and by `SUBMISSION.json` inside the uploaded zip. After the
   refactor, `package_submission.py` on the working tree produces a
   *differently named* archive holding the same program. If that submission
   ever has to be rebuilt, it is rebuilt from the pinned commit — exactly the
   position M8 accepted when it retired R4's static fallback. Worth stating in
   the Stage 9 commit so nobody later reads a hash mismatch as a defect.
4. **No other bot's hash moves.** The change is confined to files under
   `bots/morpheus-rs/`, and `_SHELL_REF_RE`'s comment trap is not touched —
   nothing in this plan adds a bot-relative path to any `.sh` under `bots/`.

## 6. Risks and non-goals

### This must be a pure move

Nothing here may change what the binary computes. Concretely, the following are
**out of scope** and must not ride along:

- **No reordering of floating-point operations**, no "while I'm here" tidying
  of an accumulation, no changing a `mul_add` back into `a * b + c`. M3 measured
  that exact substitution at 11.3 ms → 5.5 ms, and M5 showed a one-ulp
  difference in `matrix.rs` flipping 568 regret branches. A move that reorders
  arithmetic is not a move.
- **No visibility narrowing.** `pub` → `pub(crate)` is tempting once the groups
  exist and is a real improvement; it also changes the crate's API surface,
  which `crates/bot` and `parity` consume. Separate pass.
- **No item renames.** File and module names change; function, struct, const
  and surface names do not. Item renames would invalidate `mutation_check.py`'s
  `before` patterns wholesale, break the one-for-one correspondence with the
  Python oracle's names, and turn a mechanical diff into a reviewable one.
- **No changes to the parity surface names or the integer encodings.** They are
  the contract with `tests/parity_cases.py` and with every recorded measurement.
- **No `FILE_SURFACES` tightening**, per §5.
- **No new dependencies.** The crate has none; `support/` exists to make that
  cost visible, not to argue it away. The measured budget
  (`sha2` 565 files / 5.6% of cap, candle-core 3,888 / 39%) says a dependency
  is affordable and says nothing about whether one is needed here.
- **No threads.** Single-core is a competition constraint, not a tuning choice
  (rewrite-plan §1). Nothing in a file move should introduce `rayon`, a thread
  pool, or a `std::thread::spawn`; the startup thread-count invariant in
  `main.rs` would catch it, and it should never have to.
- **No crate split**, per §1 — including "just `parity` as its own crate",
  which is the most plausible-sounding version of it and still changes the LTO
  unit.
- **No `Cargo.toml` profile edits.** `lto = "fat"`, `codegen-units = 1`,
  `strip = "symbols"`, `panic = "unwind"` and the `mutation` profile all stay
  exactly as they are; each has a recorded reason.

### Named risks

1. **A split reorders arithmetic by accident.** The realistic vector is a
   function moved into a file with different `use` statements resolving a name
   differently, or a helper duplicated instead of moved. *Mitigation:* the
   parity corpus is bit-exact on every surface except `net` (1e-5, measured
   4.05e-6) and `matrix` (against the pairwise-dot oracle), so a reordering
   shows up as a mismatch rather than as drift. It is the reason parity runs
   after every stage rather than at the end.
2. **The optimizer notices anyway.** Module moves within a crate should be
   codegen-neutral, but M3's unexplained 1.10 → 0.28 ms depthwise swing means
   "should" has a track record here. *Mitigation:* `selfcheck`'s `warmup_ms`
   is 26 forwards and free; record it at Stage 0 and read it after Stages 3, 5
   and 7. A move that costs measurable latency is a finding worth publishing,
   not a reason to abandon the refactor — but M7's p99.9 = 140 ms against a
   150 ms limit means it must be *noticed*.
3. **The mutation map drifts silently.** *Mitigation:* it cannot — `stale` and
   `in-test` outcomes exit non-zero, and `run_parity.sh` runs the tool. The
   real risk is the *opposite*: re-pointing an entry at the wrong new file so
   it still matches something. Re-check the tally against Stage 0's number
   after each stage; it should be 140/167 throughout, and any movement is a
   defect in the re-pointing, not a discovery.
4. **Half a refactor.** Nine stages is a lot of appetite. *Mitigation:* the
   stage order puts the cheap high-fan-in moves first and the low-value
   `parity/` split last, and §4 names two explicit resting places (after 6a,
   after Stage 7). A tree with eight clean directories and one flat
   `parity.rs` is a finished state, not an abandoned one.
5. **Merge conflicts against in-flight work.** M7 still owes a replicated
   strength contrast and `bots/morpheus-rs-s32/` exists as a gitignored
   decision arm. A `git mv` of every file in the crate conflicts with anything
   touching those files. *Mitigation:* run the stages when no morpheus-rs
   measurement is open, and see §5 on not straddling the M7 contrast.
6. **Reviewer fatigue reading a 17k-line diff.** *Mitigation:* one commit per
   stage, `git mv` so renames are detected, and no content edits in the same
   commit as a move except the mandatory `use` lines. `git show --stat -M` on
   each stage should be almost entirely renames.

### Non-goals worth stating because someone will ask

- This does not shrink the submission, speed anything up, or reduce the
  dependency budget.
- It does not settle which knobs the bot should be playing (M7/M8's open item)
  or touch `deployment.json`.
- It does not close M5's 27 explained mutation survivors; the mutation tally is
  used here as an invariant, not as a target.
- It does not change the Python oracle, `bots/morpheus/`, in any way.
