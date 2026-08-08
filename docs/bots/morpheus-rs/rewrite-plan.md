# Morpheus-rs rewrite plan

Status: **confirmed 2026-08-08 — implementation under way. M0, M0.5, M1, and M2
are done (§14–§17); M0.5 awaits a submission only the account holder can make.**
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
   Note the final-state wrinkle: the hard-rule layer (`constrain_nn_action`)
   consumes the **belief**, so decision parity on any frame requires the
   captured belief snapshot as an input, not just the observation — the
   capture format in this section already includes it, but the `decide`
   parity subcommand must accept it explicitly rather than re-deriving it.

**Mechanics.** The Rust binary exposes parity subcommands
(`morpheus-rs parity transition|tensor|belief|decide --frames <file>`)
emitting canonical JSON; pytest tests under `bots/morpheus-rs/tests/`
(excluded from the content hash by the existing `tests/` rule) load fixtures,
invoke the binary, and compare. Rust-side unit tests run under `cargo test`
independently.

**CI and the 7 s budget.** The repo's 7 s ceiling guards the core suite; the
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
