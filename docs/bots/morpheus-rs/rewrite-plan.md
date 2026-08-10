# Morpheus-rs rewrite plan

Status: **confirmed 2026-08-08 — implementation complete, submission pending.
M0 through M8 are done (§14–§23), except that M0.5's and M8's exit gates both
await a generals.bot submission only the account holder can make. The Rust bot
plays its own decisions as of M5; M6's +425 Elo claim is **retracted** (§21,
§22) and the current estimate of the contrast is +138 Elo; M7 qualified the
latency on x86 and left the knob pick `unproven`; M8 built and audited the
submission archives. What is owed: a replicated strength contrast for the
knobs, and the upload.**
Revised against the declared-final morpheus state at commit `9d6f186`
(oracle `morpheus@73967d2125cc`, registry step 18 — see §14; the
`17c8ac2684ec` this plan first named was the *previous* registry head, the
tree at `d57b545`, one commit before the declared-final state). The three
post-plan commits (`b152147` kill-window planner, `d57b545` castle savings
pipeline, `9d6f186` emergency defense) changed only `tactics.py` (+644 lines),
one line of `runtime.py`, and tests/fixtures — no architecture change, details
folded in below. The rewrite is a committed decision: §9 carries tripwires
and fallbacks, not exits.

Goal: maximize playing strength inside the fixed per-turn deadline. Rust is the
means: more completed simulations and more particles inside the internal
deadline, plus predictable tail latency. The Python bot `bots/morpheus/` is
**never edited for gameplay** during this project (telemetry/probe edits are
allowed); the Rust bot lives in a new directory `bots/morpheus-rs/` with its own
bot id and version lineage, and the two are compared as siblings under the
arena's normal decision rule.

## 1. Verified baseline

All facts below were checked against the working tree on 2026-08-08.

- Entry chain: `run.sh` → `main.py` → `bots/_common/wire.py` (`run_stdio`) →
  `agent.py`. The wire protocol is line-oriented: handshake
  `player_id H W`, then per turn a 5-int scalar line + three `H`-row grids
  (type, owner, army), reply is five integers `p r c d s`.
- `run.sh` exports `OMP/MKL/OPENBLAS/VECLIB/NUMEXPR` thread pins before `exec`;
  `agent.py` additionally pins torch threads for in-process callers.
- Judge limit 150 ms/move; shipped `deployment.json` plays
  `normal_deadline_ms = 140`, `reserve_ms = 10`, `first_move_limit_ms = 8500`
  of the ~10 s grace.
- **The shipped config is explicitly not an accepted deployment.**
  `deployment.json` records *"Qualification verdict is no on this host"*, and
  `thread-pinning.md` measures the pinned bot at ~11.6% of moves over 150 ms.
  The knobs were selected for a world where search never ran. Consequence for
  this plan: the Python bot is a valid **decision oracle** (deterministic given
  a frame and RNG stream) but its knob values carry no authority; producing the
  first *qualified* configuration is a milestone of this project (M7), not an
  afterthought.
- Artifact: three TorchScript files (`model.pt`, `model_policy.pt`,
  `model_policy_wdl.pt`), manifest-confirmed 249,316 params, 12 inverted
  residual blocks `64 → 128 → 64` (pointwise expand, depthwise 3×3 with
  dilation cycle [1,2,4], pointwise project), GroupNorm(8) + ReLU6, 21×21
  board, 49 input channels, 11 output heads. ~0.2 GFLOPs per forward at
  batch 1 — small enough that framework dispatch overhead, not FLOPs,
  dominates batch-1 latency.
- Shipped `offline_p99_ms` (single-threaded M3 Pro; **to be re-measured in
  M0, quoted here only for shape**):

  | component | p99 ms |
  | --- | ---: |
  | particle_transitions (8 particles) | 99.5 |
  | leaf_batch (4) | 35.3 |
  | enemy_prior_batch | 30.1 |
  | root_inference | 11.9 |
  | belief_proposal (unused — uniform proposal deployed) | 3.8 |
  | belief_tensor | 2.8 |
  | selection | 1.8 |
  | backup | 1.8 |
  | hashing, reply | ~0.001 |

  The budget story is stark: belief transitions + root inference already
  consume ~110 of 140 ms at p99, which is why the deployed bot cannot finish
  8 simulations on every warm move. **The single biggest structural win is the
  transition kernel, not inference.**
- `numpy` in 20 modules; `torch` in 6, of which 5 execute at match time
  (`agent`, `deployment`, `evaluator`, `inference`, `network`) — `export.py`
  is offline tooling.
- 28 non-test modules, 10,303 lines; 35 test files plus `conftest.py` and
  `shaping_boards.py`. The final pre-rewrite commits grew `tactics.py` from
  1,639 to 2,283 lines (63 → 77 functions): kill-window planner (forced
  collecting march on a visible general), castle savings pipeline (sticky
  base-price site, tithe, anchor, forced build), and emergency defense
  (threat-aware garrison floor, forced response, kill race). All of it is
  pure functions of `(obs, memory, prior, recent_actions, belief)` — no new
  cross-turn state — but `constrain_nn_action` now takes the **belief** as an
  input (`runtime.py` passes it), which the parity harness must account for
  (§5). The four new test files (`test_kill_window.py`,
  `test_castle_savings.py`, `test_emergency_defense.py`,
  `test_strike_and_recovery.py`) plus the re-captured `shaping-parity.json`
  are the behavioral spec the Rust port of these subsystems is written
  against.

### Competition environment (from RULES.md + generals.bot/docs)

- One dedicated CPU core per bot, hard 2 GB memory cap, no GPU, Linux,
  x86 assumed. **Single-core is a design constraint, not a tuning choice: the
  Rust bot is single-threaded by construction** (no rayon, no thread pools).
- Sandbox toolchains: CPython 3.12.10, g++ 12.2, **rustc/cargo 1.97 stable**.
- `build.sh` runs once at intake, **no network ever** — all crates vendored in
  the zip.
- Zip ≤ 50 MB, unpacked ≤ 512 MB, **≤ 10,000 files**. The file count is the
  binding constraint on the dependency tree. *Measured at M2 (§17), because
  this line originally asserted that `cargo vendor` "blows past 10k files
  easily" and that is not what the numbers say:* `sha2` vendors to **565 files
  / 6.3 MB** (5.6% of the cap) and `candle-core` — M3's likely pick — to
  **3,888 files / 51.5 MB across 91 crates** (39%). Both fit; two more crates
  the size of candle would not. Every dependency must still earn its place, but
  against a budget that binds at the inference crate, not at a hash.
  Figures and method:
  [`morpheus-rs-dependency-budget.md`](../../research/measurements/morpheus-rs-dependency-budget.md).

These limits immediately disqualify libtorch FFI (CPU libtorch alone is
~100–200 MB of shared libraries) and make `ort`/ONNX Runtime awkward (the
crate downloads prebuilt binaries at build time — network — so the `.so`
would need manual vendoring, plus glibc-compat risk).

## 2. Architecture decision: clean-room binary, Python as oracle

**Decision: a standalone Rust binary behind the same stdio protocol, developed
clean-room with the Python bot as the frozen reference oracle. PyO3 does not
appear in the shipped path at all, and — going further than the original
default — not in the harness either: parity runs through CLI subcommands on
the Rust binary (§5), which is less machinery than a maturin/PyO3 build and
keeps the crate graph slim for the 10k-file budget.**

Why not strangler-fig (PyO3 extension inside the existing Python agent, ported
module by module):

- The end state must be a standalone binary anyway (zip, 1 core, and the whole
  point is removing interpreter overhead per node). Strangler work on the
  Python↔Rust boundary — numpy↔ndarray marshalling, GIL discipline, maturin
  packaging — is throwaway, and the marshalling tax lands exactly on the hot
  loops we are trying to fix: a transition kernel called per particle per turn
  would pay a conversion on every call.
- It would also violate the "don't edit Python morpheus" constraint, or force
  a third hybrid bot directory nobody wants to maintain.
- Estimated extra cost if chosen anyway: PyO3/maturin infrastructure, a
  vendored wheel in the submission zip, roughly 1–2 weeks of boundary work,
  and a final cutover milestone identical to the clean-room plan's M6. The one
  real benefit — in-play validation of each ported module before the whole bot
  exists — is replaced in this plan by per-module parity gates (§5), which
  validate against recorded real-game frames instead of live play.

The honest downside of clean-room: **no shippable bot exists until M6.**
Mitigation is milestone-level parity gates so correctness risk retires early,
and a walking skeleton in M1 (protocol + pass bot) so the packaging/matchup
path is proven from week one.

## 3. Network inference decision

**Decision: implement the network directly in Rust with `candle-core`
(Hugging Face's minimal pure-Rust tensor library), loading weights from a new
safetensors export. Benchmark against a 1-day `tract-onnx` spike at milestone
M3 and keep whichever is faster; define a bespoke fixed-shape kernel as a
measured upgrade path, entered only if profiling shows the framework leaving
≥30% on the table at batch 1–4.**

> **Superseded at M3 (§18).** All three were measured and the ranking
> inverted: candle costs 4.4× TorchScript at batch 1, tract 1.35×, and the
> bespoke kernel — this section's last resort — is 1.68× *faster*. The decision
> rule below is what settled it, so read this section for the rule and §18 for
> the answer. The reasoning that follows about TorchScript being
> overhead-dominated is also wrong: it runs at 34 GFLOP/s, not 17.

The candidates, weighed on the axes that matter here:

| Candidate | Batch 1–4 latency | Load time | Zip cost | Build/toolchain | Numerical drift vs TorchScript | Export change |
| --- | --- | --- | --- | --- | --- | --- |
| **candle** (hand-written graph + safetensors) | Good — `gemm` crate is well-tuned for f32; graph is 12 identical blocks we control | ~ms (mmap safetensors) | Pure Rust, moderate crate count | cargo only | Low and *controllable* — we choose op order; expect ≤1e-5 MAE | safetensors (weights only, no graph format) |
| **tract** (ONNX) | Good — designed for small nets on CPU, does whole-graph optimization, fixed-shape specialization | ~10–100 ms graph compile | Pure Rust, larger crate tree | cargo only | Medium — torch→ONNX conversion adds a translation layer where drift and op-mismatch bugs live | ONNX export added |
| **ort** (ONNX Runtime) | Likely best raw kernels | ~100 ms | +20–30 MB vendored `.so`, C++ ABI, glibc risk in sandbox | must hand-vendor binaries (crate wants network at build) | Medium (same ONNX layer) | ONNX export added |
| **burn** | No batch-1 advantage; heavier abstraction, younger | — | Large crate tree | cargo only | Low–medium | safetensors |
| **ggml bindings** | Built for quantized transformers; dilated depthwise conv support is the weak spot | — | C dep | vendored C | High | custom |
| **libtorch FFI** | Identical to today by definition | slow | **~100–200 MB — over the 50 MB zip. Disqualified.** | — | zero | none |
| **bespoke SIMD** | Best possible — fixed shapes (441-cell board, 64/128 ch) allow pre-packed skinny GEMMs and a trivial depthwise pass | ~ms | zero deps | cargo only | Low — we write it | safetensors |

Reasoning for candle-first rather than bespoke-first: the net is genuinely
tiny (~0.2 GFLOPs), so at batch 1 the current 11.9 ms root inference implies
TorchScript is running at ~17 effective GFLOPS — overhead-dominated. A
pre-packed fixed-shape implementation should land in the 3–6 ms range on one
core, and candle's `gemm` backend gets most of that without us writing
kernels. Writing the graph by hand in candle (it is ~15 layer definitions)
gives us the safetensors-only artifact contract, full control of numerics, and
a working reference implementation that a later bespoke kernel can be
validated against layer-by-layer. Bespoke-first would spend the riskiest
effort before knowing whether it is needed.

Measurement points that settle it (all at M3, on both M3 Pro and Modal x86,
single core): batch-1 and batch-4 wall time p50/p99 over 1,000 forwards with
realistic tensors; cold load + warmup time; MAE vs TorchScript outputs on the
frame corpus for every head. Decision rule: keep the fastest option whose MAE
≤ 1e-5 on every head; escalate to bespoke only if the winner is ≥1.3× slower
than the roofline estimate for the same shapes.

### Artifact contract (training keeps working unchanged)

- The training side and its TorchScript exports are untouched. `export.py`
  in `bots/morpheus/` gains one additive step (allowed per review): emit
  `model.safetensors` (full state dict, f32, canonical key names) next to the
  `.pt` files and record its sha256 in `manifest.json`. Nothing that consumes
  the `.pt` files changes.
- Until a new checkpoint ships, a one-shot converter
  (`bots/morpheus-rs/tools/convert_artifact.py`, dev-time only, uses torch)
  reads the *committed* `artifact/model.pt`, verifies `manifest.json` sha256s,
  and writes `bots/morpheus-rs/artifact/model.safetensors` plus a copied
  manifest. The Rust bot refuses to start if the manifest's
  `architecture_version` ≠ `morpheus-net-v1` or the tensor schema differs —
  same guardrails as the Python loader.
- If the tract spike wins at M3, the converter also emits ONNX; the contract
  stays "converter reads the frozen TorchScript artifact", so training still
  changes nothing.

## 4. Port order and milestones

Order follows risk and value: the biggest latency win (transitions) and the
hardest correctness surface (exact competition transition) go first; the
runtime controller and tactics — pure translation, no speedup story — go last.
Each milestone has an exit gate. **The rewrite itself is a committed decision
— there is no "keep Python" outcome.** A failed gate therefore never ends the
project; it triggers the named fallback in §9 and, if the fallback also
fails, a re-plan of that milestone. Python morpheus remains the frozen oracle
and the arena baseline throughout, nothing more.

**M0 — baselines, corpus, and CPU-feature probe (no Rust yet).**
Re-measure the Python bot: per-component p99s and per-move wall time
(p50/p99/max), completed simulations per normal move, over ≥20 instrumented
competition games on the M3 Pro, and the same suite on a single-core Modal
x86 CPU container (torch already runs there for training). Capture the parity
frame corpus (§5) in the same runs.

M0 also includes the **Modal CPU-feature probe**: Modal's SDKs are
Python/Go/JS only, so this is a Python `@app.function(cpu=1)` following the
repo's `scripts/morpheus_modal_*.py` pattern — proposed
`scripts/morpheus_rs_modal_cpu_probe.py`. It reports `/proc/cpuinfo` flags,
`lscpu`, the derived x86-64 microarchitecture level (v2/v3/v4), cache sizes,
and the CPU model, across several runs to sample host variety. Its output
decides the compile target: `target-cpu` is set to the highest level every
observed host supports (expected x86-64-v3; AVX-512 paths only if v4 shows up
consistently), with the caveat recorded that Modal is a *proxy* for "x86
server CPU" — the generals.bot sandbox cannot be probed directly (no network,
no visible logs), so the binary additionally does runtime feature detection
on any hand-written SIMD path rather than trusting the compile target alone.
Publish everything under
`docs/research/measurements/morpheus-rs-baseline.md`.
*Exit gate: baseline numbers and CPU-feature report published; compile
target chosen; corpus fixtures committed; oracle content hash recorded.*

**M0.5 — sandbox smoke test.**
Submit a trivial Rust bot (reads frames, replies pass) to generals.bot as a
zip with vendored deps and `build.sh`, to validate the offline build path,
file-count budget, and binary compatibility before any real porting.
*Exit gate: the sandbox builds and runs it. Risk R4's tripwire lives here —
failing it forces the static-binary fallback (§9) before any real porting.*

**M1 — walking skeleton + transition kernel.**
Crate `bots/morpheus-rs/` (workspace: `core` lib + `bot` bin). Stdio wire
protocol, `state.py` equivalent (flat arrays), `action.py` codec + legal
masks, `observe.py` fogged emission, and the full competition transition
(`transition.py`: builds-first, chase/reinforce/smaller-source priority,
combat, growth, deathtouch from turn 800, truncation at 1200 as a driver
check). The binary plays legal prior-free moves (pass or trivial policy) end
to end through `matchup.py --mode competition`.
*Exit gate: bit-exact transition/mask/observe parity on the full corpus
(tier 1, §5); matchup gate finishes.*

**M2 — deterministic state machinery.**
`hashing.py` (SHA-256 digests, same payload layouts), `memory.py`,
`symmetry.py`, `tensor.py` (49×21×21 builder). Hashes and integer planes
bit-exact; float planes ≤1e-6 vs Python f32.
*Exit gate: tier-1/tier-2 parity on corpus for all four modules.*

**M3 — inference.**
Safetensors converter, candle graph, warmup, the `inference.py`/
`evaluator.py`/`network.py` equivalents; tract spike; benchmark shoot-out
(§3). All 11 heads MAE ≤1e-5 vs TorchScript on corpus tensors.
*Exit gate: chosen engine beats TorchScript batch-1 and batch-4 p99 on x86,
or R1's fallback ladder (§9) has been walked to its end and the best
available engine is accepted with the shortfall documented.*

**M4 — belief filter.**
`belief.py`, `proposal.py` (uniform proposal is the deployed path; the policy
proposal ports too but stays off), `reservoir.py`, `recovery.py`
(rejuvenation replay rides the M1 transition kernel),
`particle_summary.py`. Parity with recorded RNG streams (§5): identical
surviving particle sets, weights ≤1e-9 (f64), identical recovery outcomes.
*Exit gate: tier-1/2 parity incl. recovery-triggered frames; measured
particle-transitions time at n=8 (expect ≥10× vs Python).*

**M5 — search, tactics, runtime controller.**
`matrix.py` (regret matching plus), `tree.py` (arena-allocated nodes, §6),
`search.py`, `tactics.py` (1,639 lines — the pure-translation slog: play
mask, mandatory actions, prior shaping, hard rules, general hunt),
`runtime.py` (admission control with nearest-rank p99 estimators, degradation
path, telemetry schema), `deployment.py`/`schema.py` (config parsing —
`deployment.json` format unchanged so existing tooling reads both bots).
*Exit gate: with frozen RNG and fixed simulation count, root decisions match
the oracle on ≥99% of corpus frames; every divergence enumerated and traced
to a within-tolerance tie (§5). Heuristic shaping scores ≤1e-9 in f64
(mirrors `test_heuristic_scores_parity.py`).*

**M6 — integration and the arena gates.**
Full bot behind `run.sh`. Register `morpheus-rs` in the version registry.
Run the AGENTS.md verification gate
(`PYTHON=.venv/bin/python` caveat applies to the *Python* side of the pairing),
then the strength evaluation per `docs/arena/decision-rule.md`: both bots in
the same rounds, shared ≥5-bot opponent panel including `cm_expander`, pinned
round seed, `--seat-policy alternate`, ≥200 games/arm (plan for ~1,150/arm to
afford a "proven flat" verdict), verdict from `fit.delta` quoted as
`Δ ± SE, CI₉₅, P(B>A)`.
*Exit gate at parity knobs: verdict is `no change` or better — at identical
configuration the Rust bot should play the same bot, so anything worse means
a real divergence, back to M5.*

**M7 — knob re-qualification (where the strength is won).**
This is the first *qualified* deployment either bot will have had. Procedure
in §7.
*Exit gate: a `deployment.json` whose measured p99.9 move time on the x86
reference host is < 150 ms with zero overruns over ≥20 games, and the M6
pairwise contrast re-run at new knobs reads `improvement` vs Python morpheus.*

**M8 — submission packaging.**
Zip builder, vendored crates, `build.sh`, size/file-count audit (§8).
*Exit gate: sandbox accepts and the bot plays rated games on generals.bot.*

## 5. Parity harness

**Capture.** Telemetry/probe edits to Python morpheus are allowed. Frames are
captured through `arena.instrument.runner` + `bots/morpheus/probe.py` (both
outside the content hash), extended to record per turn: the raw stdio
observation, the agent's RNG draw log (see below), belief snapshot (particle
states, weights, ess), root tensor bytes, root prior/value heads, heuristic
shaping scores, chosen action, completed simulations, and per-component
timings. Corpus: ~20 games against a varied panel, stratified to ~500–1,000
frames covering pre-contact, post-contact, recovery-triggered turns,
late-game (turn ≥ 800 deathtouch regime), and first moves.

**RNG strategy — record and replay, do not re-implement NumPy.** Reproducing
NumPy's Generator bit-for-bit in Rust is possible but fragile and worthless at
play time. Instead the capture probe records every draw the Python bot makes
(belief sampling, proposal sampling, reservoir admission, search
action-sampling) as a flat per-turn stream; the Rust bot's RNG is an injected
trait with two impls: `Replay(recorded stream)` for parity mode and
`SmallRng` for play. Draw-site *order* thereby becomes part of the ported
contract — a real constraint on the port, and the main way parity mode finds
control-flow divergences.

**Comparison tiers** (per review decision D):

1. **Bit-exact:** transition next-state, legal masks, observation emission,
   hashes/digests, memory planes, symmetry maps, action codec, wire bytes.
2. **Tolerance:** network heads ≤1e-5 MAE vs TorchScript; tensor float planes
   ≤1e-6; heuristic/shaping scores ≤1e-9 with both sides computing the
   comparison path in f64; belief weights ≤1e-9.
3. **Decision-level:** identical chosen action on ≥99% of frames with frozen
   RNG and fixed simulation count; every divergence machine-enumerated with
   the tie margin, and accepted only if traced to a within-tolerance tie
   flip. No-search frames (policy-fallback decisions) must agree exactly.
   ~~Note the final-state wrinkle: the hard-rule layer (`constrain_nn_action`)
   consumes the **belief**~~ — **wrong, corrected at M5 (§20).**
   `runtime.py` passes a belief, `constrain_nn_action` accepts it, and the body
   never reads it. The belief reaches the decision through
   `heuristic_action_scores` on the *shaping* path, so the `shaping` and
   `decide` surfaces take one and the `constrain` surface deliberately does
   not.

**Mechanics.** The Rust binary exposes parity subcommands
(`morpheus-rs parity transition|tensor|belief|decide --frames <file>`)
emitting canonical JSON; pytest tests under `bots/morpheus-rs/tests/`
(excluded from the content hash by the existing `tests/` rule) load fixtures,
invoke the binary, and compare. Rust-side unit tests run under `cargo test`
independently.

**CI and the suite budget.** The repo's 12 s ceiling guards the core suite; the
full corpus (hundreds of frames × full search) cannot and should not fit it.
Split: a **smoke slice** — ≤10 frames through tier-1 checks plus one full
`decide` — runs with the bot's pytest suite in <1 s and skips with a named
reason when the release binary is absent; the **full corpus** runs via
`bots/morpheus-rs/tools/run_parity.sh` manually and as a required step before
M5/M6 exit gates and before any registry step of `morpheus-rs`.

## 6. Data layout and allocation

Everything sized once at first move from `deployment.json`; zero heap
allocation on the per-turn path (enforced in debug builds by a counting
allocator behind a feature flag, asserted in the smoke tests).

- **Game state**: struct-of-arrays over the padded 21×21 board — fixed
  `[u8; 441]` terrain/type, `[i8; 441]` owner, `[i32; 441]` army (armies
  exceed u16 in long games), scalars packed alongside. One state = ~3 KB,
  copyable with `memcpy`. States for particles, search scratch, and replay
  live in one preallocated slab indexed by handle; "clone particle" is a slab
  copy, never an allocation.
- **Particles**: `n_particles_max` (sized for the largest config the re-tune
  grid will try, e.g. 64) slots plus equal scratch for the propose–weigh
  cycle; weights, enemy-memory blocks, and action histories in parallel
  arrays. Rejuvenation replay reuses a dedicated scratch region.
- **Tree**: nodes in a `Vec` arena addressed by `u32`; children in a
  hash map keyed by (action, obs-digest) using a fast non-crypto hasher over
  the already-computed SHA-256 digest bytes (the digests themselves stay
  SHA-256 for parity — at 0.001 ms p99 they are not worth changing). Enemy
  tables (≤8/node) inline in the node; joint stats `N/W/Q` as fixed-capacity
  `K_self×K_enemy` (16×12) flat f32 arrays allocated with the node. LRU
  eviction reuses freed indices via a free list; 4,096 nodes ≈ a few dozen MB,
  comfortably inside 2 GB.
- **Tensors**: one preallocated `[f32; 49*441]` per batch slot (max batch 8),
  written in place by the builder; no intermediate ndarray graph.
- **Inference**: weights mmap'd from safetensors; im2col/packing buffers (or
  candle's workspaces) preallocated at warmup for the fixed shapes
  {1,4,8}×49×21×21.
- **Digests/keys**: 32-byte digests stored inline (`[u8; 32]`), never boxed.

## 7. What actually gets faster — module by module

Honest expectations; every number gets replaced by M0/M3/M4 measurements.
"Translation only" means: port it because the binary needs it, expect no
meaningful latency win.

| Module (lines) | Role at match time | Expected gain | Why / notes |
| --- | --- | --- | --- |
| `transition.py` (399) | applied per particle per turn + per simulation + rejuvenation replay | **10–50×** | The 99.5 ms p99 `particle_transitions` component is interpreted numpy over tiny grids — worst case for Python, best case for Rust. This single kernel converts directly into more particles *and* more simulations. |
| `belief.py`/`recovery.py`/`proposal.py`/`reservoir.py` (1,487) | real-turn filter, rejuvenation | large, mostly inherited from the transition kernel | Filter logic itself is cheap; replay of 8-turn histories × beam 8 rides the kernel. |
| `tensor.py` (425) | ≥1 build per turn + per leaf/enemy-prior tensor | 5–20× | 2.8 ms → sub-ms; matters because it is on every inference call's critical path. |
| `search.py`/`tree.py`/`matrix.py` (1,574) | per simulation | 5–15× on selection+backup (1.8 ms each today) | Small absolute numbers but they are *per simulation*; at 30+ sims/turn the per-sim overhead is what caps throughput. Arena layout (§6) removes dict/np overhead per node. |
| network inference (`inference/evaluator/network`, 853) | root + leaf batches + enemy priors | **1.5–4×** | TorchScript is already C++; the win is dispatch overhead and fixed-shape specialization, not FLOPs. Root 11.9 ms → target 3–6 ms; leaf batch 35 ms → target 10–15 ms. The most uncertain estimate in the plan; measured at M3, with R1's fallback ladder behind it. |
| `tactics.py` (2,283) | root prior shaping, play mask, hard rules, kill-window/castle/defense planners — roughly once per turn | 2–10×, small absolute | **Pure translation, little gain, highest port effort per line.** It must be ported anyway (decisions depend on it, and no-PyO3 means it can't stay in Python), but it is scheduled last-with-search and its parity is fixture-driven (`shaping-parity` plus the four planner test files). The planners include BFS/path-field walks (`kill_plan`, `path_distance_field`) that are cheap in Rust but behavior-dense; port them test-first from their fixtures. |
| `runtime.py` (1,051) | admission control, telemetry | none | Translation only. Ported faithfully because the controller *is* the deadline behavior; its telemetry schema is kept identical so existing analysis tooling reads both bots. |
| `hashing.py` (107) | node keys | none (0.001 ms) | SHA-256 kept for parity; already negligible. |
| `observe.py`/`memory.py`/`state.py`/`symmetry.py`/`action.py` (963) | per turn / per simulation | 3–10×, small absolute | Straightforward; symmetry is play-time-irrelevant (training-side concept) but cheap to port for tensor parity checks. |
| `wire`/`agent`/`main`/`schema`/`deployment` (567) | I/O, config | none | Translation only. |
| `export.py`, `probe.py` | not in match path | not ported | `export.py` stays Python (training side); `morpheus-rs` gets its own arena-owned probe later. |

Net effect on the turn budget, using the shipped p99s as the shape: belief
update drops from ~100 ms to ~5–10 ms and root inference from ~12 to ~4–6 ms,
leaving ~120 ms of search headroom where today there is ~25 ms. At a 10–15 ms
leaf batch that is 6–10 batches ≈ **24–40 completed simulations vs ~5 today,
with room to raise the particle count** — before any knob tuning.

## 8. Knob re-qualification (M7)

The rewrite is precisely what unlocks re-deriving the deployed knobs, because
the current ones were fit to component costs that no longer exist. Procedure,
not guesses:

1. Rust telemetry publishes the same per-component schema; collect
   `offline_p99_ms` on the M3 Pro and on a single-core Modal x86 container
   (the deployment authority — the competition host is x86).
2. Grid over `n_particles ∈ {8,16,32,64}`, `target_simulations ∈ {16,32,64}`,
   `pending_leaf_batch ∈ {2,4,8}`, `search_depth ∈ {2,4,8,16}`, with
   `normal_deadline_ms`/`reserve_ms` derived from measured reply cost and
   scheduler jitter (reserve = measured p99.9 of serialize+flush + margin),
   not asserted.
3. Feasibility gate per config: belief + root + one leaf batch fits at
   measured p99; zero >150 ms moves over ≥20 full games at p99.9.
4. Among feasible configs, pick by the pairwise strength contrast (small
   grid of A/B rounds per `decision-rule.md`), not by simulation count alone.
5. Record the result in `morpheus-rs/deployment.json` with its
   `offline_p99_ms`, qualification host, and — unlike today — a *yes* verdict.

First-move budget: safetensors load + candle warmup on {1,4,8} + belief init
is expected well under 1 s vs Python's ~8.5 s; measured explicitly at M6 (the
grace window could then fund a deeper first-move search — noted as a
follow-up, not in this plan's scope).

## 9. Risks, ordered, each with a tripwire and a fallback

The rewrite is a committed decision, so no tripwire ends the project. Each
risk carries a measurable tripwire and the fallback it forces; a fallback
that also fails forces a re-plan of that milestone, never a retreat to
Python. Python morpheus stays the frozen oracle and the arena baseline —
nothing more.

1. **Rust inference is not faster (R1).** TorchScript's CPU kernels are
   mature; candle/tract could lose at batch 4 on x86. *Tripwire:* best
   pure-Rust option ≥1.3× TorchScript p99 at batch 1 **and** 4 on x86.
   *Fallback ladder:* candle → tract → bespoke fixed-shape kernels →
   vendored `ort` (accepting the ~20–30 MB `.so` and C++ ABI cost inside the
   zip budget). If the end of the ladder still trails TorchScript, ship the
   best Rust engine anyway: the throughput thesis (§7) rests on transitions
   and per-simulation overhead, not on an inference win. The shortfall is
   documented in the M3 report and revisited after M7 shows where the
   deadline budget actually binds.
2. **Decision parity unreachable (R2).** Float-order divergence cascades
   through prior shaping and regret matching could push corpus agreement
   below target. *Mitigation:* f64 on comparison-sensitive heuristic paths
   (Python already does this in numpy defaults), frozen-RNG harness,
   divergence enumeration by tie margin. *Tripwire:* <95% root-decision
   agreement on **no-search** frames after tier-1/2 all pass — that means an
   unfound logic bug, not float noise. *Fallback:* bisect by stage — the
   parity subcommands localize which module first diverges per frame; that
   module's port is redone line-by-line against its Python source before M5
   may exit. The parity targets are not relaxed to make the gate pass.
3. **Strength regression at the gate (R3).** *Mitigation:* M6 runs at parity
   knobs where "same decisions" is the expectation, isolating porting bugs
   from tuning. *Tripwire:* verdict `regression` per `decision-rule.md` at
   parity knobs after R2's enumeration is clean, or worse than `no change`
   after the M7 re-tune. *Fallback:* at parity knobs a regression is by
   definition an unfound divergence — return to R2's bisection; after the
   re-tune it is a qualification error — return to M7 with the failing
   games' telemetry. Until the contrast reads `improvement`, Python keeps
   playing rated games while morpheus-rs iterates: a schedule consequence,
   not an exit.
4. **Sandbox build failure (R4).** Vendored-crate build breaks offline, file
   count exceeded, CPU-feature mismatch (build host vs match host flags).
   *Mitigation:* M0.5 smoke submission before any porting; `target-cpu`
   chosen from the M0 Modal CPU probe, with runtime feature detection on any
   hand-written SIMD path. *Tripwire:* the sandbox rejects the
   vendored-source zip. *Fallback:* ship a self-contained static binary
   (`x86_64-unknown-linux-musl`, conservative `target-cpu`) with `build.sh`
   reduced to a no-op — the generals.bot docs explicitly allow a
   self-contained binary. The M8 packaging script produces both variants
   from day one, so the fallback stays tested rather than theoretical.
5. **Effort overrun on the translation slog (R5).** `tactics.py` +
   `runtime.py` + `search.py` are ~4,200 lines of behavior-dense code with no
   speed story to motivate them, and the final pre-rewrite commits grew
   tactics by 40%. *Mitigation:* they are gated behind M1–M4 wins, so the
   sunk cost when reaching them is already justified by measured kernels; the
   fixture-driven parity pattern (`shaping-parity.json` plus the four planner
   test files) ports test-first. *Tripwire:* M5's exit gate not met after the
   allotted effort window (set at the post-M4 review, §13). *Fallback:*
   re-scope M5 into per-subsystem slices (play mask → shaping → hard rules →
   planners), each with its own parity gate, and re-estimate — the slog gets
   more checkpoints, not a smaller destination.
6. **Memory/size ceilings (R6).** 2 GB runtime and 512 MB unpacked are
   comfortable (the bot targets ~100 MB resident); the 10k vendored-file
   count is the real one. *Mitigation:* dependency budget reviewed at every
   `cargo add`; `cargo vendor` file count checked in the M8 packaging script
   and in the M0.5 smoke test. *Fallback:* R4's static-binary variant
   sidesteps the vendored-file count entirely.

## 10. Build, packaging, and repo integration

- **Layout:** `bots/morpheus-rs/` containing `run.sh`, `Cargo.toml`,
  `Cargo.lock`, `rust-toolchain.toml` (pin ≤1.97 to match the sandbox),
  `src/`, `artifact/` (safetensors + manifest), `tools/`, `tests/`
  (pytest parity, excluded from hash), `target/` (build output, excluded —
  see below).
- **run.sh:** keeps the five thread-pin exports (harmless, and they guard any
  transitively linked BLAS) plus `RAYON_NUM_THREADS=1` defensively; performs
  a staleness check (hash of `src/**`+`Cargo.*` vs a stamp in `target/`) and
  runs `cargo build --release` only when stale; `exec`s
  `target/release/morpheus-rs`. First-run build cost is paid outside any
  match, at the caller's leisure; the matchup gate documentation gains a note
  that a cold checkout needs one build.
- **Content hash:** `fingerprint.py` hashes every file under the bot dir
  except `_SKIP_DIRS`. One minimal arena edit (approved): add `"target"` and
  `"vendor"` to `_SKIP_DIRS`. Result: `.rs` sources, `Cargo.lock`, `run.sh`,
  and the committed artifact are the rating identity; build output and
  vendored copies of crates.io code are not (the *lock file* pins them, so
  identity is preserved without hashing 10k vendored files). The Python
  bot's hash is untouched.
- **Thread pinning invariant:** the Rust bot is single-threaded by
  construction; a startup assertion verifies the process thread count after
  warmup and refuses to play otherwise (the Rust analog of
  `test_play_and_calibration_pin_the_same_thread_count`).
- **Submission zip (M8):** `bots/morpheus-rs/tools/package_submission.py`
  produces `run.sh` (no build, just exec), `build.sh`
  (`cargo build --release --offline`), sources, `.cargo/config.toml` pointing
  at the vendored registry, `vendor/`, and `artifact/`; audits zip size,
  unpacked size, and file count against the published limits.
- **Registry:** `morpheus-rs` registers as a new bot id with its own lineage
  in `data/bot_versions/morpheus-rs.json`. Per review decision H it stays a
  sibling of `morpheus` permanently; no replacement step is planned.

## 11. Success criteria — how each is measured

1. **Correctness:** the tiered parity results (§5) on the full corpus,
   published with the M5/M6 gates: tier-1 zero mismatches, tier-2 max errors
   per surface, tier-3 agreement rate plus the enumerated divergence list
   with tie margins. The existing `test_hash_backup_parity.py` /
   `test_heuristic_scores_parity.py` fixture pattern is extended across the
   language boundary, with the smoke slice in CI.
2. **Latency:** M0 re-measured Python baseline vs M6 Rust measurement, same
   hardware (M3 Pro) and same games where seeds allow: per-move wall p50/p99,
   completed simulations per normal move, per-component p99 table. Repeated
   on Modal x86 single-core as the deployment-authority numbers.
3. **Strength:** the AGENTS.md matchup gate, then the pairwise contrast per
   `docs/arena/decision-rule.md` (same rounds, shared panel with anchor,
   alternated seats, ≥200 games/arm minimum and ~1,150/arm planned), verdict
   quoted as `Δ ± SE, CI₉₅, P(B>A)` — never a rank. Run twice: at parity
   knobs (expect `no change`) and after M7 re-tune (require `improvement`).
4. **First-move budget:** measured load+warmup+init time in the Rust
   telemetry over the M6 games; must fit `first_move_limit_ms` with the same
   reserve discipline as today (expected to pass by a wide margin).

## 12. Where things live

- This plan: `docs/bots/morpheus-rs/rewrite-plan.md` (this file). After
  acceptance, implementation knowledge splits into small topic files under
  `docs/bots/morpheus-rs/` per docs-keeper rules (`inference.md`,
  `parity.md`, `packaging.md`, …), and `docs/index.md` gains the entries.
- Measurements: `docs/research/measurements/morpheus-rs-*.md`.
- Bot code: `bots/morpheus-rs/` as in §10. Training-side and Python-bot
  files change only where §3's artifact contract says (one additive export
  step in `export.py`) and where the capture probe (§5) needs telemetry —
  both inside the edits approved during plan review.

## 13. Open items

The post-commit revision pass is done (2026-08-08): parity surfaces
re-verified against `9d6f186`, oracle frozen at `morpheus@73967d2125cc`,
line counts and the tactics/belief coupling updated in §1, §5, §7, §9. The
rewrite was confirmed as a committed decision — §9 was reworded from kill
criteria to tripwire + fallback. `deployment.json` and the artifact did not
change. Remaining open items:

- Decide the effort window that arms R5's tripwire (proposed: review after
  M4 with measured kernels in hand).

## 14. M0 — done

Delivered 2026-08-08. Numbers and their caveats live in the reports, not here.

- **Oracle content hash: `morpheus@73967d2125cc`**, registry step 18. This
  plan originally named `17c8ac2684ec`, which is step 17 — the tree at
  `d57b545`, *before* the `9d6f186` emergency-defense commit the same
  paragraph calls the declared-final state. The frozen oracle is the
  declared-final tree, and it is now a registered lineage step rather than an
  unregistered working tree.
- **Baselines:**
  [M3 Pro, 20 games](../../research/measurements/morpheus-rs-baseline.md) and
  [one x86 core, 6 games](../../research/measurements/morpheus-rs-baseline-modal.md).
  Both carry an uncaptured control arm, so the cost of measuring is bounded
  rather than assumed (M3 Pro: 1.01× at p99).

  The headline correction to §1: **the shipped `offline_p99_ms` table
  understates live cost across the board**, and the gap is worst exactly where
  the rewrite bets. On the M3 Pro, belief update + root inference is
  **166.7 ms at p99** against a 140 ms internal deadline — §1 estimated ~110 —
  so at p99 the deadline is spent before search is admitted at all. On one x86
  core it is worse: `particle_transitions` alone measures **357.5 ms** at p99
  (3.6× shipped), simulations per normal move fall from 12 to 8, and the
  belief update is deferred on 86.9% of turns against 81.2% on the laptop.

  Two component ratios deserve their own note, because they are not in §7's
  expectations at all: `selection` measures 9.6× its shipped p99 on the M3 Pro
  and 13.5× on x86, and `backup` 2.9×/4.7×. §7 has both as "small absolute
  numbers, 5–15× gain"; they are neither small nor previously measured under
  live play. M5's port should treat them as first-class, not as the tail of
  the search work.

  The plan's arithmetic direction survives — transitions dominate, inference
  does not — but every knob in `deployment.json` was fitted against numbers
  that are wrong by 1.4× to 13×. That is M7's problem, and M7 now has a
  measured floor to re-derive from.
- **CPU probe:**
  [`morpheus-rs-cpu-probe.md`](../../research/measurements/morpheus-rs-cpu-probe.md).
  **Compile target: `x86-64-v3`** — the floor across sampled hosts. Modal
  placed containers launched together on one fleet generation, and separate
  runs landed on different ones, including an AMD generation without AVX-512.
  So `target-cpu` stays at v3 and §9's runtime feature detection requirement
  stands on evidence rather than caution.
- **Corpus:** captured through a new generic capture hook in
  `arena.instrument` plus `bots/morpheus-rs/tools/capture_morpheus.py`. Format,
  strata, and the recorded-RNG contract:
  [`parity-corpus.md`](parity-corpus.md). Committed smoke slice:
  `bots/morpheus-rs/tests/fixtures/parity-smoke.jsonl.gz`; the full corpus is
  derived data under `data/morpheus/morpheus-rs/`.
- **Not settled by M0:** the x86 arm is six games on one Modal fleet, not the
  twenty this section asks for, and the container reports 17 visible CPUs
  against its one-core reservation — so it is a *shape* result (Linux, x86,
  contended, no AVX-512 guaranteed) rather than a deployment qualification.
  M7 needs the full suite on the reference host before any knob is accepted.

## 15. M0.5 — rehearsed, not yet submitted

Delivered 2026-08-08. Layout, hash rules, and the traps below:
[`packaging.md`](packaging.md). Rehearsal results:
[`morpheus-rs-sandbox-smoke.md`](../../research/measurements/morpheus-rs-sandbox-smoke.md).

The walking skeleton exists: a zero-dependency Cargo workspace
(`crates/core` + `crates/bot`) speaking the stdio protocol and replying pass.
It finishes the AGENTS.md gate against `cm_expander` under `--mode competition`
(truncation at 1200, a normal end), and both submission variants build and play
inside a one-core Linux x86 container with `block_network=True` and the
sandbox's own rustc 1.97.1 — vendored in 3.3 s at 10 files / 8.7 KB, static in
0.03 s at 3 files / 226 KB, against limits of 10,000 files and 50 MB.

**The exit gate is not met, and cannot be met here.** It reads "the sandbox
builds and runs it", and only the account holder can submit to generals.bot.
What the rehearsal can falsify has been falsified: offline build, file count,
toolchain version, and — the one the packaging host genuinely could not answer
— that the musl binary an arm64 Mac cross-builds actually executes on x86-64
Linux. What remains is the sandbox's own image agreeing, which is exactly what
R4's tripwire is for.

*Corrected during M2:* the "offline build" claim above originally rested on
zips with **no dependencies**, where `--offline` cannot fail because there is
nothing to resolve. It never exercised `.cargo/config.toml` source
replacement, a transitive graph, or a build script running at intake. A third
`vendor-probe` variant now carries all three (`tools/vendor_probe.py`) and
rides the same offline container — 571 files, builds in 2.5 s. The claim is
now worth what it appeared to be worth.

Four things nearly broke it, each now a rule in `packaging.md`: shipping
`rust-toolchain.toml` would make an offline sandbox try to *download* the
pinned patch release; a `build.sh` beside `run.sh` breaks the repo's own gate
through an unguarded `relative_to` in `matchup.py::build_agent`; cross-linking
musl from macOS needs the toolchain's `rust-lld`; and a *comment* naming
`bots/morpheus/run.sh` in a shell script put the Python bot inside the Rust
bot's content hash, because `_SHELL_REF_RE` cannot tell prose from a `source`
line.

§10's approved `_SKIP_DIRS` edit landed as `target`, `vendor`, **and `tools`** —
the third for the reason `probe.py` is excluded: the capture module and the
packager never play, so editing them must not re-identify the bot. No existing
bot's hash moved.


## 16. M1 — done

Delivered 2026-08-08. The deterministic board layer is ported and proved:
`state`, `transition`, `action` (codec, legal mask, live build cost),
`observe`, and the `VisibleMemory` container. Harness and its own validation:
[`parity-harness.md`](parity-harness.md).

**Exit gate met.** Tier-1 parity is bit-exact over the full corpus — 1,328
heavy frames from 20 games, **495,867 cases** across five surfaces
(`transition` 127,692, `order` 344,101, `observe` 21,282, `mask` 1,396, `cost`
1,396), zero mismatches. The matchup gate finishes under `--mode competition`
against `cm_expander` (loss at turn 215, a normal end). The bot plays the first
legal move from the ported mask and nothing else: M1 asks for "legal
prior-free moves", and a hand-written heuristic here would be code nobody
intends to keep.

**A green parity run was not enough, and that is the milestone's real finding.**
Mutation testing — break one behaviour in the real source, check the harness
notices — caught two behaviours that no recorded case could distinguish:

- Deleting the 50-tick army growth passed every recorded transition case; no
  sampled state sat at `time % 50 == 49`.
- Deleting the NumPy negative-index wrap in `_determine_move_order` passed even
  after 1,528 cases aimed at move ordering. The wrap reads row `h-1`, and the
  preset pads smaller boards to 21×21 with mountains, so on a padded board that
  row is border and can never be owned. **The corpus cannot tell the two
  implementations apart there** — a limit of §5's replay-only design, not of
  this port.

The response was `synthetic_states()`, positions built to reach what replay
cannot, plus `tools/mutation_check.py` as a standing check: 15 of 17 mutations
caught, the two survivors documented as equivalent mutants. §5 should be read
with this correction — **replay proves agreement, mutation proves the proof** —
and every later milestone's parity claim needs both halves.

Two smaller notes for later milestones:

- The parity stream is integers, not JSON (§5 said JSON). Rust has no std JSON,
  and the crate stays zero-dependency to protect the 10,000-file budget.
  Rationale in `parity-harness.md`; the tier-1 guarantee is unaffected.
- `transition` currently clones a ~3 KB state per call. §6's slab allocation is
  a search and belief concern (M4/M5), but the kernel is called per particle
  *and* per simulation, so M4 should measure before assuming the clone is free.

## 17. M2 — done

Delivered 2026-08-08. `hashing`, `memory`'s update rule, `symmetry`, and the
49-plane `tensor` builder are ported, with SHA-256 written out by hand to keep
the crate zero-dependency. Harness detail: [`parity-harness.md`](parity-harness.md).

**Exit gate met, and tier 2 exceeded.** Parity is exact over the full corpus —
1,328 heavy frames, **500,072 cases** across nine surfaces (the five from M1
plus `memory` 1,399, `hash` 1,399, `tensor` 1,399, `symmetry` 8), zero
mismatches. The matchup gate still finishes under `--mode competition`.

The tensor's float planes match **bit-for-bit**, not merely inside §5's 1e-6.
That is deliberate: the widths are mirrored site by site, because the Python
mixes `float64` and `float32` on purpose and the network was trained on the
rounding it actually does. Computing everything in double would be more
accurate and would diverge.

**§5's 1e-6 is too loose to be the gate, and M2 proves it.** A mutation
computing `army_value` in single precision stayed inside 1e-6 on every case and
survived. The harness therefore enforces bit-exactness and reports the 1e-6
figure separately, so a future platform's genuine drift is a deliberate
decision with a number attached rather than a check that silently never fires.
Later milestones with float surfaces (M3's network heads at 1e-5, M4's belief
weights at 1e-9) should assume their budgets are similarly unable to see
precision bugs, and enforce the tightest agreement actually achieved.

Mutation coverage grew to 32; **28 caught**, four equivalent mutants
documented. Two of M2's new mutations found real holes rather than confirming
the port: no case had a **remembered castle observed as plain fog** — an
emitted board cannot produce one, since a castle in fog encodes as type 5 — and
no previous action carried a **half-split**, leaving the tensor's move-kind
plane constant at 1.0. Both are now crafted cases. This is the second milestone
running where the mutation pass, not the parity pass, was what found the gap.

Two notes for M3 and M4:

- `hashing` narrows `remembered_owner` and `remembered_castle_owner` to one
  signed byte on the way into the digest, because the Python arrays are `int8`
  while this crate stores `i32`. The mutation that widened them is caught, but
  the trap will recur wherever a NumPy dtype is narrower than its Rust
  counterpart — check the dtype, not the semantic type, whenever a new payload
  is hashed.
- `particle_summary.py` is *not* ported. `tensor` takes a `BeliefSummary` as an
  input and the harness feeds it from Python, which keeps the M2/M4 boundary
  clean. M4 ports the aggregation and joins the two halves.

### M2 addendum — two corrections found by asking "why?"

Neither came from a failing gate; both came from a question about a decision
that had already been made and written down.

**The dependency-budget claim in §1 was wrong.** It asserted that `cargo
vendor` "blows past 10k files easily", and that assertion had been reused to
justify hand-writing SHA-256. Measured: `sha2` is **565 files, 5.6% of the
cap**; candle-core, M3's likely pick, is **3,888 files, 39%**. Both fit. §1 now
carries the numbers and `tools/dependency_budget.py` re-derives them, so the
next dependency argument starts from a measurement rather than a memory.

The hand-written SHA-256 stays, on reasons that survive the correction:
hashing costs 0.001 ms p99, so a library's speed advantage buys nothing
measurable; the digests are dictionary keys, not a security boundary; the
implementation is verified against the NIST vectors *and* 6,995 digest
comparisons against Python's `hashlib`; and `sha2` drags `libc` in for CPU
detection — 404 of those 565 files — inside a static musl binary.

**M0.5's offline-build proof was weaker than it read.** Both shipping zips have
no dependencies, so their `--offline` build could not have failed. §15 is
amended above and the gap is closed by a probe rather than by a permanent
dependency the shipped bot does not need.

The pattern worth keeping: both of these were *documented* conclusions that no
test covered. A gate proves what it exercises; a justification proves nothing
at all until someone measures it.

## 18. M3 — done, and it reversed §3's ranking

Delivered 2026-08-09. The network runs in Rust, from a safetensors conversion
of the frozen artifact, faster than the TorchScript it replaces. Engine
detail: [`inference.md`](inference.md). Figures:
[`morpheus-rs-inference-bench.md`](../../research/measurements/morpheus-rs-inference-bench.md).

**Exit gate met.** The shipped engine beats TorchScript at batch 1 and batch 4
on both hosts — 1.68×/1.44× on the M3 Pro, 1.64×/4.15× on one x86 core — and
all eleven heads agree with the oracle inside §5's 1e-5 MAE over the full
corpus. The x86 batch wins are large because TorchScript's batched call goes
superlinear there (4× the work for 11.8× the time) while a loop of single
forwards stays flat; batch 1 is nearly a tie, and the container reports 17
visible CPUs against a one-core reservation, so some of that penalty may be
contention. A shape, not a qualification — M7 owes the reference host a full
suite.

**§3 chose the wrong engine, by its own decision rule.** The plan ranked candle
first, tract second, and a bespoke fixed-shape kernel last — to be "entered
only if profiling shows the framework leaving ≥30% on the table". Measured on
the M3 Pro at `policy_wdl` p99, against TorchScript's 6.09 ms (b1) and 20.61 ms
(b4):

| engine | batch 1 | batch 4 |
| --- | ---: | ---: |
| candle 0.9 | 4.57× slower | 6.20× slower |
| tract-onnx 0.21 | 1.40× slower | 2.34× slower |
| **bespoke (shipped)** | **1.68× faster** | **1.44× faster** |

Both library options trip R1's tripwire (≥1.3× at batch 1 **and** 4), so §9's
ladder was walked candle → tract → bespoke and stops at the third rung; the
vendored-`ort` rung, with its 20–30 MB `.so`, never has to be argued about.

candle's loss has one cause, and it is worth recording because it generalizes:
it implements grouped convolution by looping over groups, so the depthwise
layer becomes 128 separate convolutions costing **1.14 ms** per call where the
same layer costs 0.02 ms here. Twelve blocks of that is half of candle's forward for
6% of the network's arithmetic. It is a library tuned for transformers meeting
a MobileNet-shaped net, not a general statement about frameworks — its
pointwise GEMMs are fine.

**§3's premise was also wrong, in the direction that matters less.** It
inferred from the shipped 11.9 ms root inference that TorchScript was
"overhead-dominated at ~17 effective GFLOPS". Measured directly,
single-threaded TorchScript does 212 MFLOP in 6.09 ms — **35 GFLOP/s**, a
competent GEMM library on a fast core, not overhead. The bespoke engine reaches
60 GFLOP/s, so §3's 3–6 ms target was hit (3.6 ms) for a reason it did not
give. §7's throughput thesis never rested on inference either way.

**The largest single lever was a language detail, not an algorithm.** Rust
compiles floating-point with contraction off, so `acc += a * b` never becomes a
fused multiply-add. Writing `mul_add` explicitly moved the forward from 11.3 ms
to 5.5 ms — from losing to TorchScript by 1.8× to beating it. Eight accumulator
lanes in GroupNorm's statistics took another 0.9 ms.

The honest failure of the milestone is a measurement that will not sit still.
The depthwise stage read 1.10 ms for most of M3 and reads 0.28 ms now, from
edits elsewhere in the crate; three candidate causes were tested and all three
refuted, leaving whole-crate LTO as the only remaining explanation and no
proof of it. Per-stage numbers here are build-sensitive and must be
re-measured, not quoted. The whole-forward figure was stable to 0.01 ms across
every run, which is why it is the one the gate uses.

**The x86 run found a bug that would have shipped.** Its first attempt
measured 277 ms per forward against TorchScript's 5.69 ms — 49× slower.
`f32::mul_add` promises a single rounding, so a target without a hardware FMA
calls libm's `fmaf()` to emulate it. The binary compiles, passes parity, and
returns correct answers fifty times too slowly.

`.cargo/config.toml` sets `target-cpu=x86-64-v3` to prevent exactly that, but
cargo discovers that file by walking up from the **working directory**, not
from `--manifest-path`. The benchmark had the bug, and so did
`tools/submission/build.sh` — the script the judge runs at intake, which
resolved its own directory into `$DIR`, passed `--manifest-path`, and never
`cd`-ed. Run from any other directory, the submitted bot would have spent
277 ms per inference against a 150 ms deadline: correct moves, all of them too
late, with nothing in a match log to say why.

Both launchers now `cd` first, `gemm::HAS_HARDWARE_FMA` records what the binary
compiled with, `bench` prints it, the x86 script refuses to publish a report
without it, and `Session::load` warns on stderr. The packager's static-musl
path was already correct. §1's `target-cpu` line should be read as a
correctness requirement, not a tuning preference.

**Parity: the first surface with a real tolerance.** `net` (all eleven heads
through all three entry points) and `prior` (`legal_normalized_policy` plus
`backup_value`) add 2,798 cases to the corpus run, for 502,870 across eleven
surfaces. `net` cannot be bit-exact — two engines, different summation orders,
fused multiply-adds — so §5's 1e-5 is doing real work for the first time:
measured worst MAE 4.05e-6, 2.5× under budget, against the *thousandfold* slack
M2 found beneath the tensor's 1e-6. Every float surface now prints its worst
observed |Δ| so the headroom stays a number.

`prior` re-taught M2's lesson unprompted. The first version softmaxed in f64 —
more accurate — and disagreed with the oracle in the eighth decimal, because
`torch.softmax` runs at the tensor's dtype and every decision the oracle has
ever made was made on that f32 rounding. Mirroring the width fixed it.

**Addendum: the depthwise halo, and half a measurement.** The x86 report put
the depthwise layer at 1.52 ms — a third of the forward for 6% of its
arithmetic — against 0.28 ms on the M3 Pro. Laying its plane out with a
four-cell halo, so a tap becomes one contiguous 601-element axpy instead of 21
rows of 21, took it to 0.66 ms on x86 and left arm64 unchanged at 0.29 ms. The
whole batch-1 forward went 5.07 ms → 4.09 ms there.

The rewrite is worth having; the reason it was nearly skipped is worth more.
Three formulations of that loop had been compared *on arm64*, found within 1%
of each other, and written up as evidence that the layer's cost was structural
and not worth chasing. All three were fine on arm64. None of that transferred.
An arm64-only profile is half a measurement, and M4 and M5 should treat it as
one.

Two notes for M4 and M5:

- The bot still plays M1's first-legal-move. M3 ships the engine, not a
  decision: `NetworkEvaluator`'s prior shaping needs `tactics.py` (M5) and its
  tensor needs `particle_summary.py` (M4). Nothing in the *playing* path loads
  the artifact yet, so the first move that does will be the first real test of
  the load-and-warmup budget (3.9 ms + 21.4 ms measured standalone, against
  8.5 s of grace).
- The x86 arm is one Modal container, and M0 already found separate runs
  landing on different fleet generations. It is a shape result, not a
  qualification. M7 still owes the reference host a full suite.

## 19. M4 — done, and the mutation pass found more than the parity pass

Delivered 2026-08-09. The belief layer is ported: `belief`, `proposal`,
`recovery`, `reservoir`, `particle_summary`, plus the injected RNG §5 specified
and nothing had needed until now. Design and the two NumPy behaviours it rests
on: [`belief.md`](belief.md). Figures:
[`morpheus-rs-belief-bench.md`](../../research/measurements/morpheus-rs-belief-bench.md).

**Exit gate met, both halves.** Parity is exact over the full corpus — 1,328
heavy frames, **515,710 cases across twenty-one surfaces**, zero mismatches,
including the recovery-triggered strata the gate names. The measured belief
update at n = 8 beats the Python by **30.6× at p50 / 23.8× at p99** on the
filter alone and **46.6× / 39.6×** on the path that includes recovery, against
a bar of ≥10×. The AGENTS.md matchup gate still finishes under
`--mode competition`.

**The M0 baseline's 142.6 ms was recovery, and nobody could have known that.**
`runtime.py` charges `filter_step` and the `recover_belief` it may trigger to
one component, so the baseline reported a single p99 for a cost distribution
that is bimodal by a factor of a hundred: the filter alone measures 1.1 ms at
p99, the same block with recovery 127 ms. §7's "large, mostly inherited from
the transition kernel" was right about the mechanism — rejuvenation replays an
eight-deep history across a beam of eight, and every step is M1's kernel — but
the *budget* implication is new. M7 should reserve for the two paths
separately; a knob fitted to the mixture is fitted to how often recovery
happened to fire on the capture host.

**§5's record-and-replay design works, and creates one new way to be wrong.**
`Replay` verifies every argument of every draw before returning the recorded
result, which makes draw-site order a checked contract exactly as §5 intended.
But it returns the *oracle's* index whatever the port computed — so a surface
that compared only sampled actions would pass a port whose every probability
was wrong. `propose` therefore emits each particle's distribution alongside its
action. Anyone adding a randomized surface should assume the same trap.

**Two NumPy behaviours had to be reproduced, and both are host-conditional.**
`np.sum` is pairwise, not sequential, and it feeds an ESS that decides whether
a resample runs — and a resample consumes a draw, so a last-bit difference
desynchronizes the replay stream rather than rounding an answer. `np.argsort`
is an unstable introsort, and under the deployed *uniform* proposal every legal
action ties, so the order recovery walks its candidates in is decided entirely
by the sort's internals. Both are transcribed; both have their own parity
surface. The second one is worse than it looks: NumPy ≥ 2.0 dispatches
`argsort` to `x86-simd-sort` under AVX-512-SKX, and M0's CPU probe found Modal
hosts both with and without it — so on some x86 hosts **the Python bot itself**
would rank those candidates differently. That is a property of the oracle, not
of the port, and the surface exists so it fails with a name.

**The milestone's real finding is the mutation pass.** Thirty-five new
mutations; the first run caught twenty-one and left **fourteen unexplained
survivors** — by far the worst any milestone has had, against a parity run that
was green throughout. Every one was a hole, and four of them shared a cause
worth stating plainly: **the corpus is not a sample of inputs, it is a sample
of the inputs the deployed configuration produces.** Eight particles at
weight 1/8 means normalizing is the identity, 1/8 is exact in f32, and eight
equal values sum the same in any order — so `ess`'s division,
`summarize_belief`'s rescale, an f32 accumulator in the belief planes, and the
pairwise sum itself were all invisible at once. The other three were an entry
point with no surface at all (`initialize_belief`), an answer encoding that
emitted a history's *length* where the bug reorders its contents, and a
comparison that was written and then not made. Final: **71 of 86 caught**, 15
survivors, all with recorded explanations. Detail:
[`parity-harness.md`](parity-harness.md).

Two smaller notes:

- One mutation went **stale** — M3's depthwise halo rewrite had moved the line
  it matched, so it had quietly stopped testing anything. Failing the run on a
  misaimed mutation is what surfaced it. That rule has now paid for itself
  twice.
- `tools/mutation_check.py` gained a scoping map and a `mutation` cargo
  profile. Eighty-six mutations against fat LTO and twenty surfaces each is
  over an hour; scoped and without the LTO link it is about fifteen minutes.
  A survivor still re-runs the *full* surface set before it is recorded, so an
  over-narrow map costs a slow run rather than a false coverage hole.

Two notes for M5:

- The bot still plays M1's first-legal-move. Nothing calls the belief in the
  *playing* path yet: the decision needs `search.py` and `tactics.py`.
  `particle_summary` closes the join M2 left open, so the tensor path is now
  complete on the Rust side alone.
- `ParticleReservoir::replace_from_belief` is the one place in the belief layer
  where play-time answers legitimately differ. The Python builds
  `np.random.default_rng(0)` inside the method; reproducing *which* particles
  that seed picks would mean reimplementing PCG64, which §5 declined. The
  inputs are equal-weight, so both draw from the same distribution and differ
  only in the sample; the harness injects the oracle's recorded draws and checks
  it exactly. M5 owns the caller and should keep the generator dedicated.

## 20. M5 — done, and the bot plays

Delivered 2026-08-09. `matrix`, `tree`, `search`, `tactics`, `runtime`,
`deployment` and the network evaluator are ported: about 5,200 lines of Python,
the largest and least glamorous slice of the rewrite. **The Rust bot now makes
its own decisions** — the belief, the network, the shaping blend, the hard
rules and the search all run in the playing path, and M1's first-legal-move is
gone. Design notes:
[`search-and-tactics.md`](search-and-tactics.md). Harness:
[`parity-harness.md`](parity-harness.md).

**Exit gate met, and the decision half of it exactly.** Parity runs over the
full corpus — 1,328 heavy frames, **533,552 cases across thirty surfaces**,
zero mismatches. The gate asks for root decisions matching the
oracle on ≥99% of frames; `decide` reads **1,322 of 1,322**, and the closest
call separated the committed action from the runner-up by 3.42e-05 of shaped
prior, 52× the network prior's worst disagreement. The gate's other half —
heuristic shaping scores inside 1e-9 in f64 — is met by a wide margin: they
match to the **last bit**, so the harness enforces bit-exactness and keeps
1e-9 as the floor. The AGENTS.md matchup gate finishes under `--mode
competition`: a win at turn 825 by capturing the enemy general, with two
castles built.

**§5 was wrong about the belief, and the correction is small but load-bearing.**
The plan records a "final-state wrinkle" saying `constrain_nn_action` consumes
the belief, so decision parity needs the captured snapshot as an input.
`runtime.py` passes one, the function accepts one, and the body never reads it.
The belief does reach the decision — through `heuristic_action_scores` on the
shaping path — but not through the hard rules. §5 is amended above. The
parameter stays on the Rust signature so the discrepancy stays visible.

**The milestone's real finding is that the oracle's search is not
reproducible, and regret matching is why it matters.** M1–M4 all matched to the
last bit. `matrix.rs` cannot, because the Python writes `q_eff @ sigma_enemy`
and NumPy sends `@` on f64 to BLAS, whose reduction order is the vendor's —
Accelerate here, OpenBLAS on the container. Measured over 3,000 random simplex
vectors at the widths the search uses, BLAS agrees with neither a sequential
sum (mean 0.5 ulp) nor NumPy's own pairwise reduction (mean 0.4 ulp). This is
the third host-conditional behaviour the harness has had to name, after
`npsum` and `argsort`.

On its own it is a ulp. But `regret_matching_strategy` branches on
`sum(max(regret, 0)) <= 0`, and on the **first backup of every fresh enemy
table** the true regret is exactly zero — all joint entries still sit at
first-play urgency, so `u_enemy` equals `v`. Whether the accumulated float
lands on `0.0` or on `2.8e-17` decides between falling back to the prior and
spreading over the positive entries: one ulp, a qualitatively different mixed
strategy, deterministically, on every new table on every turn.

That is proved rather than argued. Re-running the *oracle* with its `@`
replaced by NumPy's own pairwise reduction — the single substitution the port
makes — reproduces the Rust answer bit-for-bit on all 1,362 `search` cases. The
surface therefore carries two oracles: the port must match the pairwise one
exactly, and what the shipped one moves is tallied — **2,083 statistic vectors
over 1,362 cases, 568 regret-branch flips**, decided on magnitudes up to
8.88e-16.

**Consequence for M6, and it changes the exit gate's reading.** §4's M6 gate
says "at identical configuration the Rust bot should play the same bot, so
anything worse means a real divergence". That holds for the no-search decision
and is now measured at 100%. It does **not** hold once a simulation completes,
and not because of a porting bug: the Python bot does not play the same bot as
itself across BLAS vendors. M6 should read a `regression` verdict at parity
knobs as evidence to investigate, not as proof of R2's "unfound logic bug", and
the pairwise-dot oracle is the tool that separates the two.

**The mutation pass is where M5 falls short, and it says so.** Eighty-one new
mutations, three passes, **118 of 167 caught**. The first pass returned 63
survivors with a single shape: every rule gated on a *game phase* — the
garrison window, the castle window, a live threat, a reachable kill — and every
behaviour needing a deep or contended tree. Neither is reachable from seven
recorded frames and a 5×5 synthetic board built for M1's transition
boundaries. Eleven purpose-built tactical positions, richer search
configurations and a `runtime` surface closed fifteen and took the count to
118; three more are now recorded as equivalent by construction.

Forty-six remain, each diagnosed and none mysterious: rules that need *two*
gates in range at once, rules that need an exact numeric coincidence, an
oscillation exemption shadowed by the next exemption, and tree contention a
24-node cap over four batches still does not produce. Detail and the specific
positions each would need: [`parity-harness.md`](parity-harness.md).

So the milestone's claim is deliberately split. **The parity result meets the
standard the earlier milestones set; the mutation coverage does not.** M1's
lesson — replay proves agreement, mutation proves the proof — is exactly why
that distinction is written down rather than averaged away. The parity numbers
say the two implementations agree on half a million cases; the mutation number
says that for forty-six behaviours in the tactical and search layers, nothing
in the case set could have told them apart. Closing that is the first thing M6
should do, before it reads anything into a strength contrast.

Two smaller notes:

- Tier 3 had to be reformulated to be answerable. A full-search decision
  depends on cross-turn state — the tree, the rolling history digest, the
  estimator windows — that no single frame carries, and the corpus is a record
  of what happened rather than a script that reproduces it. The *no-search*
  decision has no such dependency, and M0 measured belief plus root at 166.7 ms
  against a 140 ms deadline, so it is what the bot plays on a large fraction of
  turns. `search` covers the tree separately, from a frame, with a scripted
  evaluator so the comparison cannot fail on the last bit of a softmax.
- `deployment.json` for `morpheus-rs` is the Python's file copied field for
  field, so M6 compares two bots at one configuration. Its `offline_p99_ms`
  values are the *Python's* measured costs and are wrong for this binary by
  roughly the ratios M4 measured; they are there to make admission behave
  identically, not because they describe this bot. M7 re-derives every field.

## 21. M6 — the coverage debt, and a gate that passed for the wrong reason

Delivered 2026-08-09. The bot is integrated behind `run.sh`, registered as
`morpheus-rs@5456f5532cc2` (lineage step 1), measures itself, and has played
its arena round. Harness detail: [`parity-harness.md`](parity-harness.md).
Telemetry: [`telemetry.md`](telemetry.md). Figures:
[latency](../../research/measurements/morpheus-rs-m6-latency.md) and
[strength](../../research/measurements/morpheus-rs-m6-strength.md).

**Exit gate: the matchup half stands, the strength half is RETRACTED.** The
AGENTS.md gate finishes under `--mode competition` (win at turn 736, two
castles). The pairwise contrast — seven bots, 21 pairs, 192 games each, round
seed 7, seats alternated, `--strict-versions` — read
`morpheus@73967d2125cc` → `morpheus-rs@5456f5532cc2` at **+425.06 ± 22.77 Elo,
P(B > A) = 1.0000** over 1,152 games per arm.

**That number does not replicate** (found at M7, 2026-08-10). Four later
measurements of the same two programs, one of them replaying this round's own
seeds under its own job count, agree at **0.504 ± 0.042** against the round's
0.867 — a 7.4-sigma disagreement, cause unidentified, with map seeds,
tournament parallelism, external CPU load and the programs themselves each
ruled out by experiment. The current best estimate of the contrast is **+138
Elo**. Full account:
[morpheus-rs-m6-replication.md](../../research/measurements/morpheus-rs-m6-replication.md).

The rewrite's deployment case is untouched, because it was never this number:
on one x86 core the Python bot is late on **13.8%** of moves and reaches
`RULES.md`'s 50-fault forfeit around move 360 of a typical game, while this
binary is late on **0 of 21,000** (§22).

**§4's reasoning for that gate was wrong, and the error is worth more than the
verdict.** It said: "at identical configuration the Rust bot should play the
same bot, so anything worse means a real divergence." The knobs *are*
identical — verified field by field, only a note, a runtime name and a
qualification host differ — but **identical configuration is not identical
behaviour when the configuration is a deadline.** Every knob in
`deployment.json` is a time budget or a bound on work attempted inside one. The
Python completes its belief update *and* root inference on 18.4% of turns; this
binary does it on 100%, at 16 simulations per move against 12, with 0.02% of
moves over the judge's limit against 8.76%.

That reasoning is still right, and it is *also* the reason a strength round can
go wrong: a bot whose strength is a function of spare host compute will be
measured differently by two hosts in two states, and nothing in the round
manifest records which state it was. M6 published the difference between two
bots; the replication says a large part of what it published was the difference
between two afternoons.

The consequence for how M6's gate should have been read: a `regression` at
parity knobs would still have meant "investigate" (M5's note stands), but `no
change` was never the right prior. Faithfulness is what the parity harness
proves — 533,726 cases, `decide` 1,322 / 1,322 — and the strength round cannot
speak to it either way.

**M5's instruction was to close the mutation gap before reading anything into a
strength contrast, and doing so found three harness bugs rather than three
missing positions.** The pass reads **140 of 167 caught**, up from 118, with all
27 survivors explained; no new mutations were written, so the whole movement is
M5's twenty-nine open survivors becoming twenty-two caught and seven
equivalent-with-a-measurement. Half wanted positions built the way M5's eleven
were, and fifteen more are in `tactical_positions()`. The other half wanted
something else:

- **M5's four "disagreeing" particles were one board.** `_replace()` on a
  NamedTuple copies the tuple and shares every array in it, so all four held the
  same `armies`, hashed to one enemy view, and no node ever had two enemy
  tables. Four eviction and LRU mutations survived for that reason alone, under
  a comment asserting the opposite.
- **A root-only comparison cannot see a child.** A leaf value is applied
  unchanged to every edge on the path, and `Replay` returns the oracle's sampled
  index whatever this side computed — so a divergence inside a child changes
  neither the root's statistics nor the tree's shape. `search` now emits every
  node.
- **A constant leaf value hides every weighting.** `ScriptedEvaluator` returns
  one number for every leaf, so every `q` entry is equal and an average of equal
  numbers does not depend on its weights. The enemy-hash cache key, the
  reservoir's weights and the marginal aggregation were unobservable for that
  and nothing else.

Two rules cannot be reached by playing at all: the retention score's touch term
only ever decides a tie in `last_used`, which is a per-node backup counter that
advances on every backup — 680 backups, never a tie. `evict` states the table
set instead, as `runtime` states its samples. Thirty-one surfaces now.

**The bot measures itself.** `arena.instrument.runner` constructs an `Agent`
in-process and samples `probe.py`; a subprocess binary offers neither, so §7's
promise of "its own arena-owned probe later" comes due here — M6's latency and
first-move claims are unmeasurable without one. `telemetry.rs` writes one JSONL
line per turn on `probe.py`'s key set, armed by `MORPHEUS_RS_TRACE`, buffered to
the end for the reason the runner buffers. §10's thread-count invariant is
checked after warmup through `/proc/self/status`, on the platform that has one.

**First-move budget (§11.4), measured:** 3.7 ms load + 109.1 ms warmup + 0.1 ms
init = **114.8 ms at p50**, against `first_move_limit_ms` of 8,500 and the
Python's ~8.5 s. The follow-up §8 noted — that the grace window could fund a
deeper first-move search — is now a 8.4-second question rather than a
hypothetical one.

Two notes for M7:

- **The simulation ceiling is now the binding constraint, not the deadline.**
  `target_simulations = 16` was fitted to a bot that could not reach it; this
  one hits it on the median move with 30 ms of budget unspent. The re-tune grid
  in §8 should treat 16 as its floor, not a midpoint.
- **A component table needs its call counts.** The trace carries calls for three
  components, and only those three could be normalized per call — which
  mattered: `enemy_prior_batch` reads 0.9× per turn and 1.6× per call. M7 fits
  knobs from these numbers and should emit all ten.

## 22. M7 — the knobs are qualified, the pick is not, and M6 is retracted

Delivered 2026-08-10, **incomplete by design**: §8's steps 1–3 are done and
measured, step 4 came back `unproven`, and step 5 has therefore not been taken.
No knob shipped. Figures: [knobs](../../research/measurements/morpheus-rs-m7-knobs.md),
[x86 qualification](../../research/measurements/morpheus-rs-m7-x86-guard0.md),
[the M6 retraction](../../research/measurements/morpheus-rs-m6-replication.md).

**The cost model is one number.** A network forward costs ~4.9 ms on the M3 Pro
and ~5.1 ms on x86, and a turn affords about twenty-six of them. That explains
the whole grid: `target_simulations` 16 → 32 buys four real simulations and 64
buys none (the deadline binds at ~20); `search_depth` 2 → 8 is nearly free
because a simulation's cost is its leaf forward, not its tree walk; particles
trade against search at about four simulations per doubling; and
`pending_leaf_batch = 8` has the best median and the worst tail, which is what
a zero admission guard predicts.

**§8 asks for a reserve that does not exist.** `reserve_ms` is parsed, stored
and never read — in the Python oracle as much as in the port, so the port is a
faithful transcription of a dead knob. The turn deadline is
`turn_start + normal_deadline_ms` and nothing subtracts a reserve. What
protects the tail is `admission_guard_ms`, which ships at 0.0; M6's two moves
over the limit were forecast misses (`leaf_batch` at 72 and 110 ms against a
forecast near 20), admitted because the guard is zero. Priced: **15 ms of guard
costs four simulations and buys twenty milliseconds of tail.**

**The latency half of the exit gate is met, on the host that has authority.**
Twenty games per configuration on one x86 core, ~10,000 normal moves each:
`n8-s32-b4-d8` reads p99.9 **140 ms**, max **141 ms**, **zero moves over 150**,
at twenty simulations against the parity config's sixteen. x86 is *faster* at
inference and ~1.7× *slower* at the belief kernel than the laptop, which is
why the shipped table has to come from there and not from here.

**The strength half is `unproven` and nothing was shipped.** `parity → s32`
fits at **+18.15 ± 23.42, CI₉₅ [−27.8, +64.1], P = 0.781**. Reading a pick out
of that would be inventing one.

**And the round that produced it found something worse.** The same round put
`morpheus` and `morpheus-rs` far closer than M6 had, which turned into the
[M6 retraction](../../research/measurements/morpheus-rs-m6-replication.md):
M6's +425 does not replicate, the cause is unidentified, and seeds,
parallelism, external load and the programs are each ruled out by experiment.
The lesson is recorded in
[`decision-rule.md`](../../arena/decision-rule.md#a-round-is-not-evidence-until-it-replicates)
rather than here, because it is not a morpheus fact: **a round is not evidence
until it replicates, and host state is an experimental variable this repo does
not record.**

What M7 still owes:

- A replicated strength contrast for the knobs, on a quiet host, run twice.
  Until then `bots/morpheus-rs/deployment.json` keeps the parity knobs and its
  *no* qualification verdict, which is now only half true: the latency is
  qualified on x86, the configuration it qualifies is not the one measured for
  strength, and saying so is more useful than a verdict field that cannot
  express it.
- `bots/morpheus-rs-s32/` is the decision arm and is gitignored derived data
  (the `bots/morpheus-*/` rule); it stays until the contrast is settled, then
  goes, with its registry entry left as provenance.

## 23. M8 — the submission is built, and a green smoke test meant nothing

Delivered 2026-08-10. Two archives, audited, reproducible, and — for the first
time — put in front of a competition match as the *submitted* bytes rather than
the repo launcher. Rules: [`packaging.md`](packaging.md). Figures:
[M8](../../research/measurements/morpheus-rs-m8-submission.md) and
[the offline x86 smoke](../../research/measurements/morpheus-rs-sandbox-smoke.md).

**The exit gate is not met and cannot be met here** — the same wall M0.5 hit.
It reads "the sandbox accepts and the bot plays rated games on generals.bot",
and only the account holder can submit. Everything upstream is done: the
vendored zip is **43 files / 1.13 MB** and the static fallback **7 files /
1.59 MB**, against 10,000 files and 50 MB; both build offline on the sandbox's
own rustc 1.97.1 in a one-core `block_network` container (17.2 s and 0.2 s);
both are byte-reproducible; and the vendored bundle wins the AGENTS.md gate
against `cm_expander` at turn 592 with a castle built. The submission checklist
is at the end of the M8 report.

**M8's finding is that none of the packaging evidence collected before it was
worth what it read.** Delete `artifact/` from a bundle and the smoke test still
passes, byte-identically: two well-formed actions, exit zero, green. That is
not a defective test, it is the shape of the bot. `Seat::new` returning `Err`
degrades the seat to passing every turn rather than exiting, because the judge
forfeits a game on an early exit but charges one fault out of fifty for a bad
reply — so the *correct* behaviour during a game makes a missing artifact, a
`deployment.json` that fell back to the Part 07 placeholders, and a build with
no hardware FMA all look like a bot that answers the protocol perfectly.

§4's one-line description of this milestone — "zip builder, vendored crates,
`build.sh`, size/file-count audit" — is a list of things that can all be true
of an archive containing a brick. `morpheus-rs selfcheck` is what M8 adds: it
resolves the config explicitly, loads the weights against their manifest
digest, builds the real seat with its warmup and thread-count invariant,
reports `HAS_HARDWARE_FMA`, and decides one hand-built frame where a skip is
the only wrong answer. Both `build.sh` variants run it and abort intake on
failure — a rejected submission costs a resubmission, a degraded one costs
every rated game it plays. It caught a real broken archive within the hour: an
aborted packaging run had left a partial four-file static zip that the Modal
smoke then shipped to x86, and only the selfcheck failed it.

**The repo's own gate cannot be run on the submission layout, and that is a
constraint on verification rather than an inconvenience.**
`matchup.py::build_agent` executes any `build.sh` beside a `run.sh` and formats
its log line with `build.relative_to(REPO_ROOT)`, unguarded, against the
*submodule's* root — so every submission-shaped directory crashes it before the
first move. M0.5 recorded this as a reason to keep `build.sh` in `tools/`; M8
is where it stopped being about file placement. The gate is run by building at
intake and then deleting `build.sh`, which is exactly the judge's sequence and
leaves `build_agent` — which the judge does not have — as the only untested
part.

**R4's fallback plays a different speed, which nothing had measured.** §9 asks
for the static variant to be built and smoked every packaging run "so the
fallback stays tested rather than theoretical", and it has been. Tested is not
qualified: on one x86 core the two variants' 26 warmup forwards agree within 2%
— same target, same FMA, same kernels — while one decision costs **1.75× to
2.2×** more under musl and the artifact load ~1.9×, the allocator difference
and nothing else. So the fallback does not inherit M7's `p99.9 = 140 ms, zero
over 150`. Triggering R4 means re-deriving knobs for it, or adding an
allocator, which would be this crate's first dependency and needs the
measurement §17 taught the project to demand.

Two notes for whoever submits:

- Upload the **vendored** zip. `SUBMISSION.json` inside it records the content
  hash of the program, which is the row to match in
  `data/bot_versions/morpheus-rs.json` when a rated result comes back.
- `deployment.json` still carries M7's half-verdict: the latency is qualified
  on x86, the configuration it qualifies is not the one measured for strength,
  and M7's replicated contrast is still owed. M8 packages a bot; it does not
  settle which knobs it should be playing.
