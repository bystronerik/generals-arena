# Joe-rs port plan

Status: **implemented, 2026-08-14 — J0–J4 done, J5 not attempted.** Landed
knowledge lives in the sibling topic files ([export.md](export.md),
[parity.md](parity.md), [xla-semantics.md](xla-semantics.md),
[latency.md](latency.md)); this plan stays as the design record.

Goal: a minimal Rust sibling bot `bots/joe-rs/` that plays the greedy policy
of the frozen joe network (`bots/joe/artifact/ema.eqx`, manifest at
`bots/joe/artifact/manifest.json`: `history_transformer`, depth 5, embed 384,
8 heads, patch 3, history 7, 8,556,250 params). No search, no belief
machinery, no sampling — one forward pass and an argmax per turn, exactly
what `bots/joe/agent.py` does today.

`bots/joe/` gameplay code is **never edited**. The one allowed touch on the
joe side is an additive weight-export step (§3). The two bots are siblings
with separate ids and lineages, compared under the arena's normal rules.

Constraints inherited from the reviewed
[`morpheus-rs/rewrite-plan.md`](../morpheus-rs/rewrite-plan.md) (§1–§3, §10),
restated because they bind here too:

- **No PyO3** anywhere — shipped path or harness. Parity runs through CLI
  subcommands on the Rust binary; conversion tooling is plain Python.
- **candle + safetensors first** for inference, with the caveat that
  morpheus-rs M3 measured candle 4.4–6.2× *slower* than TorchScript on its
  conv net and shipped a bespoke kernel instead. Joe's op mix is different
  (§5), so candle gets its own measurement, not a presumption — and the same
  fallback (§9, R1).
- **Sandbox limits**: Linux x86, one dedicated core, 2 GB memory, no GPU,
  rustc/cargo 1.97, `build.sh` runs offline, zip ≤ 50 MB, unpacked ≤ 512 MB,
  **≤ 10,000 files** — all crates vendored. Single-threaded by construction
  (no rayon).
- Compile target `x86-64-v3` per the morpheus-rs M0 CPU probe; runtime
  feature detection on any hand-written SIMD.
- `fingerprint._SKIP_DIRS` already excludes `target`, `vendor`, and `tools`
  arena-wide (morpheus-rs §15), so joe-rs needs **no arena edits**: sources,
  `Cargo.lock`, `run.sh`, and the committed artifact are the rating identity.

Two morpheus-rs lessons carried in as rules, not suggestions:

- **Replay proves agreement, mutation proves the proof** (M1): every parity
  gate ships with a mutation pass that breaks the real source and confirms
  the harness notices.
- **Loose float tolerances cannot see precision bugs** (M2): mirror float
  widths site by site, enforce the tightest agreement actually achieved, and
  report the nominal tolerance separately.

## 1. Verified baseline (checked against the working tree, 2026-08-13)

- Entry chain: `run.sh` → `main.py` → `bots/_common/wire.py::run_stdio` →
  `agent.py::Agent`. Wire protocol: handshake line `player_id H W`; per turn
  a scalar line `turn my_land my_army opp_land opp_army` followed by three
  `H`-row space-separated integer grids (type, owner, army); reply is five
  integers `p r c d s` plus newline, flushed.
- Per-turn path (`agent.py::step`): wire frame → 14-channel raw tensor
  (`frame_to_raw`) → build-cost grid → 39-channel augmentation with
  persistent `AugmentedObsState` → move + build masks → one float32 forward
  → greedy argmax over 10×21×21 masked logits → decode to
  `[pass, r, c, dir, half]`. A pass reply is clamped to `1 0 0 0 0` because
  the pass channel's argmax cell can sit in the pad region.
- Joe is **deterministic at play time**: greedy decode, no RNG draws. This
  removes the entire record-and-replay RNG machinery morpheus-rs needed —
  the biggest simplification this port gets.
- Artifact: `ema.eqx` is `eqx.tree_serialise_leaves` output — 100 array
  leaves written in pytree order as concatenated `.npy` records, no names.
  Cross-check: 8,556,250 params × 4 bytes + 100 leaves × 128 B header =
  34,237,800 bytes = the manifest's `weights_size` exactly.
- Latency baseline
  ([joe-phase2-cpu-latency.md](../../research/measurements/joe-phase2-cpu-latency.md)):
  the full Python per-move path on one x86 core runs p50 9–11 ms,
  p99 ≈ 19 ms against the 150 ms move limit. Forward-pass compute is
  ~0.4 G MACs (~0.8 GFLOPs). There is **no latency emergency**: the port's
  budget goal is "comfortably inside the limit", not "beat JAX".
- First move: Python spends ~4 s of the ~10 s grace on JIT compile. Rust has
  no JIT; load + warmup should be tens of milliseconds.

## 2. Layout

```
bots/joe-rs/
  run.sh                 # thread pins + stamp-checked cargo build + exec (morpheus-rs pattern)
  Cargo.toml Cargo.lock  # single binary crate `joe-rs` — no workspace; v1 needs no lib split
  rust-toolchain.toml    # pin ≤ 1.97; never ships in a submission zip (morpheus-rs §15 trap)
  src/
    main.rs              # stdio loop
    wire.rs              # frame parse / reply
    obs.rs               # joe_obs.py port: raw tensor, build cost, masks, augment, normalize
    net.rs               # forward pass (candle), weight load + schema check
    action.rs            # decode/encode, greedy pick
    parity.rs            # `joe-rs parity <surface> --fixtures <file>` subcommands
  artifact/
    model.safetensors    # converted weights (committed iff ema.eqx policy allows; see §3)
    manifest.json        # copy of joe's + safetensors sha + tensor-schema version
  tools/
    convert_artifact.py  # .eqx -> safetensors (dev-time Python; outside content hash)
    capture_fixtures.py  # runs Python joe over games, dumps parity corpus
    mutation_check.py    # break-the-source harness check
  tests/                 # pytest parity drivers + committed smoke fixtures (outside hash)
```

`run.sh` reuses the morpheus-rs launcher pattern verbatim in spirit: the five
thread-pin exports plus `RAYON_NUM_THREADS=1`, a source-stamp check so
`cargo build --release` runs only when sources moved, then
`exec target/release/joe-rs`. (And per the recorded trap: no literal path to
any other bot's files, even in comments — the shell-reference scan would drag
it into the closure.)

## 3. Weight export: `ema.eqx` → safetensors

`ema.eqx` has no tensor names, but its leaf order is fully determined by the
`HistoryTransformer` pytree structure. The converter therefore never parses
the `.eqx` framing itself:

1. `tools/convert_artifact.py` (Python, dev-time, jax + equinox + safetensors)
   verifies `manifest.json`'s `weights_sha256` against the file, rebuilds the
   template exactly as `agent.py` does, and calls
   `eqx.tree_deserialise_leaves` — the same code path the deployed bot
   trusts.
2. It flattens `eqx.filter(net, eqx.is_array)` with
   `jax.tree_util.tree_flatten_with_path` and derives dotted names from the
   paths. Expected schema (float32 throughout; equinox `Linear.weight` is
   `(out, in)`, `y = Wx + b`; LayerNorm eps = 1e-5):

   | tensor | shape |
   | --- | --- |
   | `embedder.weight` / `.bias` | (384, 351) / (384) |
   | `value_token` | (1, 384) |
   | `pos_encoding` | (52, 384) |
   | `transformer_layers.{0..4}.norm1.weight` / `.bias` | (384) |
   | `transformer_layers.{i}.attn.{q,k,v,out}_proj.weight` / `.bias` | (384, 384) / (384) |
   | `transformer_layers.{i}.norm2.weight` / `.bias` | (384) |
   | `transformer_layers.{i}.ff_linear1.weight` / `.bias` | (1152, 384) / (1152) |
   | `transformer_layers.{i}.ff_linear2.weight` / `.bias` | (384, 1152) / (384) |
   | `norm_out.weight` / `.bias` | (384) |
   | `policy_head.weight` / `.bias` | (90, 384) / (90) |
   | `value_head.weight` / `.bias` | (128, 384) / (128) |
   | `temporal_encoder.{army,land}_l{1,2}.weight` / `.bias` | l1 (512, 512), l2 (384, 512) |
   | `temporal_type_embed` | (2, 384) |
   | `bin_centers` | (128) |

   That inventory sums to exactly 8,556,250 parameters in 100 leaves — the
   manifest's `n_params` — so the converter asserts both numbers and refuses
   to write on any mismatch. `bin_centers` is a serialized leaf, so it is
   **exported, not recomputed** in Rust; a `linspace` reimplementation is a
   parity risk with zero upside.
3. Output: `bots/joe-rs/artifact/model.safetensors` plus a manifest that
   copies joe's (provenance intact), adds `safetensors_sha256`, the source
   `weights_sha256`, and a `tensor_schema: joe-net-v1` tag. The Rust loader
   refuses to start on a schema-version or shape mismatch — the same
   guardrail the Python loader gets from `tree_deserialise_leaves`.

Where it lives: the converter is the v1 path and lives in
`bots/joe-rs/tools/` (morpheus-rs precedent — outside the content hash,
reads the *committed* joe artifact, touches nothing in `bots/joe/`). As a
follow-up, `scripts/joe_export_bot.py` may gain one additive flag that emits
the safetensors next to `ema.eqx` on future exports; that is allowed but not
required for v1 and changes nothing that consumes `ema.eqx`.

Like joe, a re-conversion rewrites a file inside the bot dir and therefore
forks joe-rs's rating identity — that is the intended behavior, not a bug.

## 4. Observation encoding: porting `joe_obs.py`

Scope of the port (`obs.rs`), in call order per turn:

1. `frame_to_raw` — wire grids → the engine's 14-channel `(14, H, W)` f32
   tensor. Integer comparisons on type/owner codes; bit-exact by
   construction.
2. `build_cost_from_raw` — 35 base + `max(0, 14 − 2·manhattan)` surcharge
   from own structures within radius 6. Pure integer kernel; bit-exact.
3. `compute_valid_move_mask` — `(H, W, 4)` in the engine's UP/DOWN/LEFT/RIGHT
   direction order; boolean logic, bit-exact.
4. `compute_build_mask_from_raw` — own plain cell with `armies ≥ cost`;
   bit-exact.
5. `augment_obs` — the stateful part: pad `H×W` (true boards are 18–21) to
   21×21 with the seen-pad-mountain accumulation rule, mountain vs
   structures-in-fog split, scalar-channel broadcast into padding; the two
   7-deep army-delta history stacks; `seen`/`enemy_seen` via 3×3 max-pool
   (SAME); accumulated castles/generals/mountains; last-seen enemy army
   value and its `log1p(t)/5` decay channel; the 512-wide opponent
   army/land ring buffers; coordinate channels; the 25-channel stack.
   `AugmentedObsState` becomes a plain Rust struct of fixed `[f32; 21*21]`
   (and `[f32; 512]`) arrays carried across turns.
6. `normalize_observations` (divisions by 50) and `prepare_action_mask`
   (−1e9 penalties, half-move channels sharing the move mask, pass always
   allowed, build channel from the build mask).
7. `decode_action` / `encode_action` — flat index ↔ `[pf, r, c, d, half]`
   over the 10-channel head (0–3 full, 4–7 half, 8 pass, 9 build).

**Bit-for-bit verification.** Everything above is f32 adds, subtracts,
multiplies, and divides-by-50 in a fixed order, plus one transcendental
(`log1p` on channel 21). The port mirrors op order and float width site by
site (the M2 rule), so the target is:

- **Bit-exact** on all integer-derived channels, masks, costs, the action
  codec, and every pure-arithmetic float channel.
- Channel 21 (`log1p(t)/5`) is the one site where XLA's `log1p` and Rust's
  may differ in ULPs. The harness reports per-channel max ULP distance; if
  channel 21 is not bit-exact, the achieved bound is pinned as the enforced
  gate for that channel alone, with the number and reason documented —
  never a blanket 1e-6.
- Because `AugmentedObsState` accumulates for up to 1,200 turns, parity is
  checked **sequence-level**, not just per frame: replay a whole recorded
  game's frames through the Rust state machine and compare every turn's
  augmented tensor. Single-frame fixtures with recorded input state exist
  too, for localization when the sequence check fails.

The fixture generator (`tools/capture_fixtures.py`) drives the *actual*
`bots/joe/agent.py` step function (imported, not reimplemented) over
recorded competition games and dumps per turn: raw wire text, the 14-channel
raw, build cost, both masks, the input and output obs-state, the augmented
tensor, the temporal stack, the flat logits, value, and the chosen action.

## 5. Forward pass in candle (`net.rs`)

The graph, single sample, fixed shapes throughout (pad_to 21, patch 3 →
7×7 = 49 patches; 52 tokens × 384):

1. Normalize obs (§4.6), patchify `(39, 21, 21)` → reshape
   `(39, 7, 3, 7, 3)` → permute `(1, 3, 0, 2, 4)` → `(49, 351)`.
2. Embed: linear 351 → 384 per patch.
3. Temporal encoder: two independent MLPs 512 → 512 → 384 (SiLU between) on
   `history / 50`; add `temporal_type_embed`.
4. Sequence `[value_token, temporal_army, temporal_land, patch_0..48]` +
   `pos_encoding` (52, 384).
5. Five pre-norm blocks: LN → MHSA (8 heads, head_dim 48, scale √48,
   softmax in f32 — deployment is f32 everywhere, so no bf16 casts exist to
   port) → residual → LN → 384 → 1152 SiLU → 384 → residual.
6. `norm_out`, then:
   - **Policy head**: linear 384 → 90 per patch token → `(49, 90)` →
     unpatchify `(7, 7, 10, 3, 3)` → permute → `(10, 21, 21)` → add the
     −1e9 mask → flatten to 4,410 logits → argmax (first-max tie-break, to
     match `jnp.argmax`).
   - **Value head, implemented in v1**: linear 384 → 128 bin logits,
     softmax, dot with the exported `bin_centers` (linspace −1..1) →
     scalar in [−1, 1]. Cost is one extra 384×128 GEMM — noise. v1 play
     ignores it, but it is computed, parity-tested, and logged to stderr as
     free eval telemetry, so the parity surface covers the whole network.

Candle op coverage is not the risk it was for morpheus: the graph needs
matmul, bias add, LayerNorm, softmax, SiLU, reshape/permute — all present in
`candle-core` + `candle-nn`, no convolutions, no exotic ops. The risk is
**dispatch overhead at batch 1**, which is exactly where candle lost the
morpheus shoot-out. Joe's shapes are friendlier (52×384 GEMMs instead of
tiny dilated depthwise convs), so candle gets measured before being judged
(§8, J3). Weights load by mmap from safetensors; all activation buffers are
preallocated at startup for the fixed shapes; zero per-turn heap allocation
is the target, checked the morpheus way (counting allocator in debug).

Latency budget: ~0.8 GFLOPs/forward on one x86-64-v3 core. Target: full
per-move path (parse → obs → forward → reply) **p99 ≤ 50 ms** on one x86
core — 3× the Python baseline is acceptable for v1; the limit is 150 ms.
Tripwire and fallback in §9 (R1).

## 6. Parity testing

**Corpus.** ~10 competition games of Python joe against a varied panel
(mixed opponents, mixed seeds, at least one game reaching the turn-800+
regime and one truncation), captured by `tools/capture_fixtures.py` (§4).
Stratified to ~300–500 frames for per-frame surfaces plus 3 full game
sequences for the state-accumulation check. Committed smoke slice
(≤ 10 frames + one short sequence) under `bots/joe-rs/tests/fixtures/`; the
full corpus is derived data under `data/joe/joe-rs-parity/` (gitignored,
consistent with the `data/joe/` rule).

**Tiers** (morpheus-rs §5 shape, joe-sized):

1. **Bit-exact**: wire parse, 14-channel raw, build cost, move/build masks,
   action codec, and augmented-obs channels per §4 (channel 21 possibly
   pinned at its achieved ULP bound).
2. **Tolerance**: flat policy logits and the 128 value-bin logits vs the
   recorded JAX outputs. Nominal gate ≤ 1e-4 max abs error (different GEMM
   summation orders make bit-exactness unreachable); the *achieved* max
   error over the corpus is recorded and becomes the enforced bound.
   Value scalar ≤ 1e-4.
3. **Decision-level**: greedy action equal on ≥ 99.5% of corpus frames;
   every divergence machine-enumerated with its logit tie margin and
   accepted only if the top-2 gap is inside tier-2's achieved error bound.
   Sequence fixtures must additionally agree turn-by-turn under the Rust
   bot's own accumulated state.

**Mechanics.** The binary exposes
`joe-rs parity {raw,cost,mask,obs,forward,decide,sequence} --fixtures <f>`
emitting a compact binary/integer stream (morpheus-rs learned std-Rust has
no JSON and the crate budget is precious — but joe-rs already carries candle,
so if `serde_json` rides in transitively for free it may be used; decided at
J1 by looking at the actual lockfile, not by preference). Pytest drivers in
`bots/joe-rs/tests/` compare against fixtures; they carry a `joe` marker and
skip with a named reason when the release binary is absent, keeping the
default suite inside the repo's 15 s ceiling.

**Mutation pass.** `tools/mutation_check.py` breaks the Rust source one
behavior at a time — history-stack roll direction, seen-accumulation OR,
pad-mountain rule, a division-by-50 site, `q_proj`/`k_proj` swap, softmax
scale, argmax tie-break — and asserts the harness fails each time. Known
corpus blind spots get crafted synthetic fixtures up front: a board smaller
than 21 with an owned cell adjacent to the pad border (exercises the
seen-pad-mountain rule), a half-move decode, a build decode, and a pass
whose argmax cell lies in padding.

## 7. Verification gate and repo integration

Per AGENTS.md, before joe-rs is called ready:

```bash
PYTHON=.venv/bin/python .venv/bin/python competition-module/competition/matchup.py \
  bots/joe/run.sh \
  bots/joe-rs/run.sh \
  --mode competition --seed 0
```

(`PYTHON=.venv/bin/python` is load-bearing for the Python seat — without it
the joe side dies silently.) The match must reach a normal end. The game is
stored under `data/games/` before any rating refit; `joe-rs` registers as a
new bot id with its own lineage in `data/bot_versions/joe-rs.json`.

After the gate, the optional-but-cheap strength sanity check: an
`evaluate-bot-change`-style A/B of joe vs joe-rs per
`docs/arena/decision-rule.md`. At true parity the verdict should read
`no change`; anything else means an unfound divergence and reopens J4.

Docs: after this plan is accepted, implementation knowledge splits into
small topic files under `docs/bots/joe-rs/` (`export.md`, `parity.md`, …)
and `docs/index.md` gains the entries — docs-keeper rules, same as
morpheus-rs did it.

## 8. Milestones

Ordered so correctness risk retires before any speed question is asked, and
so a shippable skeleton exists from day one (morpheus-rs M1 lesson).

- **J0 — conversion and schema.** `convert_artifact.py` writes
  `model.safetensors` + manifest from the committed joe artifact.
  *Done when:* sha256 of the source verified; 100 tensors with exactly the
  §3 names/shapes/dtypes; param sum = 8,556,250; safetensors sha recorded;
  a round-trip check in Python (load safetensors, compare every leaf
  bit-exact against `tree_deserialise_leaves`) passes.
- **J1 — walking skeleton.** Crate + `run.sh` (stamp build) + stdio wire +
  pass-every-turn agent; parity subcommand plumbing exists with the `raw`
  surface wired.
  *Done when:* the §7 matchup gate finishes with the skeleton (a pass bot
  loses or truncates — both are normal ends); fixture capture tool produces
  the corpus; the smoke fixtures are committed.
- **J2 — observation port.** `obs.rs` complete per §4.
  *Done when:* tier-1 parity is green over the full corpus including the
  3 full-game sequences; the crafted synthetic fixtures pass; the mutation
  pass catches every planted obs mutation (survivors documented as
  equivalent); channel-21 policy decided and written down.
- **J3 — forward pass.** `net.rs` in candle, both heads, loader schema
  checks, warmup.
  *Done when:* tier-2 parity green over the corpus with the achieved error
  bound pinned; full per-move path benched on one x86 core (Modal 1-core
  container, mirroring the Phase 2 method) at p99 ≤ 50 ms — or R1's
  fallback has been invoked and its outcome benched instead.
- **J4 — integration.** Greedy decode replaces the pass stub; pass-reply
  clamp; stderr value telemetry; end-to-end determinism check (same seed →
  same game, twice).
  *Done when:* tier-3 decision parity green (≥ 99.5%, divergences
  enumerated); sequence fixtures agree turn-by-turn; the §7 gate finishes
  with the real bot; `joe-rs` registered; the A/B sanity check run and
  quoted.
- **J5 — submission packaging (optional, only if joe-rs is to be
  submitted).** Extract the morpheus-rs packager into shared logic both Rust
  bots drive: vendored crates, offline `build.sh`, no `rust-toolchain.toml`
  in the zip, `build.sh` kept out of `matchup.py::build_agent`'s trap,
  size/file audit. Planned in detail — split, milestones, tests, tripwires —
  in [packaging.md](packaging.md), which also corrects R4 below: the measured
  binding limit is the 50 MB zip cap (80.5% used), not the file count (39.6%).
  *Done when:* the offline one-core container build passes, counts quoted
  against the 10,000-file / 50 MB limits.

## 9. Risks

1. **R1 — candle batch-1 overhead.** Morpheus measured candle 4.4× slower
   than TorchScript at batch 1; joe's GEMM-shaped graph should fare better,
   but that is a guess until J3. *Tripwire:* full-move p99 > 75 ms on one
   x86 core. *Fallback:* bespoke fixed-shape path — the whole net is ~7
   distinct GEMM shapes plus LayerNorm/softmax/SiLU loops, and morpheus-rs
   proved the bespoke route both feasible and fastest. The safetensors
   artifact contract is identical either way, so the fallback changes no
   tooling. Even at the tripwire, 75 ms is inside the 150 ms limit — this
   risk costs effort, not the project.
2. **R2 — silent leaf misalignment in export.** An `.eqx` file has no
   names; a converter bug that permutes two same-shaped tensors (e.g.
   `q_proj` vs `k_proj`) loads cleanly and plays plausibly-badly.
   *Mitigation:* the converter never parses the file — it goes through
   `tree_deserialise_leaves` into the exact deployment template; J0's
   bit-exact round-trip plus J3's logit parity make a permutation
   detectable; the mutation pass plants exactly this bug to prove it.
3. **R3 — transcendental / accumulation drift.** `log1p` ULPs (channel 21)
   and 1,200 turns of f32 state accumulation could push late-game frames
   past tolerance or flip near-tie argmaxes. *Mitigation:* sequence-level
   fixtures, per-channel ULP reporting, tie-margin enumeration at tier 3.
   *Tripwire:* a tier-3 divergence whose tie margin exceeds the tier-2
   bound — that is a logic bug, not float noise; bisect by parity surface,
   fix, and the targets are not relaxed.
4. **R4 — crate/file budget.** candle-core measured 3,888 files (39% of the
   10k cap) in the morpheus audit; safetensors/memmap2 are small.
   *Mitigation:* `dependency_budget.py` re-run at every `cargo add`;
   binding only if J5 happens, and the bespoke fallback (R1) drops candle
   entirely if it ever binds.
5. **R5 — scope creep.** The Python bot is 9–11 ms p50; there is no
   performance story forcing search, batching, or bf16 into v1. Anything
   beyond greedy decode is a new plan, not a J-milestone extension.

Memory is a non-risk: ~35 MB of weights + fixed activation buffers against
2 GB.
