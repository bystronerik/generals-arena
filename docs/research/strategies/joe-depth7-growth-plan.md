# Joe depth-7 growth plan

Status: draft 2026-08-18. Grow the trained depth-5 M net to depth 7 by
function-preserving layer insertion, then continue PPO on vast for ~20k
iterations from the base run's step-50000 checkpoint. This reopens decision 8
of [averagejoe-competition-plan.md](averagejoe-competition-plan.md) (M as the
terminal tier); the deployment latency gate in section 7 decides whether the
grown net can ship at all.

## 1. Goal and non-goals

- Produce `joe-M7`: the base 50k checkpoint with 2 inserted transformer
  blocks (depth 5 → 7, everything else unchanged: embed 384, 8 heads, ff ×3,
  patch 3), adapted by ~20k further PPO iterations.
- The surgery is exactly function-preserving: at step 0 the grown net plays
  bit-identically to the base net. This is verified, not assumed (section 6).
- Non-goals: width growth, ff growth, any change to the observation, action
  space, or PPO recipe. Scratch-vs-grown comparison is out of scope for now;
  the arena contrast in section 8 is grown-M7 vs base-M.

## 2. Growth operation

The blocks are pre-norm residual (`x = x + attn(norm(x))`,
`x = x + ffn(norm(x))`), so a block whose `attn.out_proj` and `ff_linear2`
weights **and biases** are zero is an exact identity. Note the zeroed
sublayer outputs are exact zeros even under bf16 (`x + 0 == x` bitwise), so
the parity check can run with the production `use_bf16` setting.
Measured 2026-08-18: the identity holds bitwise only per-op — jitting the
two nets compiles different-depth programs whose fusion shifts bf16
rounding in the *shared* layers by an ulp. `assert_forward_parity`
therefore runs un-jitted (vmapped eager), which executes the identical
primitive sequence for the shared layers in both nets. Same lesson as
joe-rs: do not chase XLA bit-exactness across different compiled programs.

- Insertion: `[L0, L1, NEW, L2, L3, NEW, L4]` — interleaved, a judgment
  call; identity-at-init makes the placement low-stakes.
- New blocks come from a fresh depth-7 `build_network` template (new PRNG
  key), then `eqx.tree_at` zeroes `attn.out_proj.{weight,bias}` and
  `ff_linear2.{weight,bias}`. Their `norm1/norm2/q/k/v/ff_linear1` keep
  fresh init — they receive gradients as soon as the zeroed layers move.
- Everything outside `transformer_layers` (embedder, value token, positional
  encoding, temporal encoder + type embed, `norm_out`, both heads) is copied
  from the base net unchanged. Token count does not change, so
  `pos_encoding` needs no surgery.
- **Both lineages are grown**: the full training net from the checkpoint
  tuple and the EMA net (the deployment policy). Param count goes ~8M → ~11M
  (+2 blocks x ~1.5M).

## 3. Continuation semantics: a new run, schedules keyed from 0

The grown run is a **new run** (`joe-M7-vast-<stamp>`) starting at global
step 0 with `num_iters: 20000` — not a resumed run at step 50000 via
`iteration_offset`. Reason, from the code:

- `main.make_optimizer` folds `iteration_offset` into the LR schedule, but
  `main.run` **zeroes `iteration_offset` whenever a `state.json` exists**.
  The first boot of an offset-based run would train at the offset LR, and
  every preemption resume after that at the unoffset LR — the schedule would
  silently jump on the first interruption. vast runs are interruptible by
  design, so this is a real path, not a corner case.
- Keying everything from 0 makes the run self-consistent across resumes:
  the fresh Adam state's step count, the LR schedule, and the entropy
  schedule all agree, on the first boot and on every resume.

The continuation schedules are set in the config, not inherited:

| Knob | Base M at it=50k | M7 setting | Why |
| --- | --- | --- | --- |
| LR | at the 5e-6 floor | `lr_power_law_max: 2e-5` (same law, floor 5e-6) | ~10k iters flat at 2e-5 to adapt the new blocks, then decay to ~8.6e-6 at 20k. The floor is too slow to recruit new capacity; a fresh 1e-4 risks destroying the policy. `target_kl: 0.02` stays as the guard. |
| Entropy | ~0.0058 | `ent_coef_start: 0.006`, power 0.2, min 0.001 | Continues the base run's value instead of restarting at 0.05, which would deliberately noise a converged policy. |
| Curriculum | final stage | single stage `{min: 17, max: null}` | The run must not re-climb the ladder; a one-stage curriculum makes saved stage index 0 valid and the win-gate moot. |
| Seed | 44 | 46 | Fresh rollout randomness. |

Everything else in `configs/M7.yaml` is copied from `M.yaml` verbatim,
`depth: 7` excepted.

## 4. Seeding mechanism: a fabricated step-0 checkpoint set

No trainer code changes. The surgery tool writes a complete v2 checkpoint
set for the new run and uploads it with `R2Store.upload_checkpoint`; the
existing `vast_boot` resume path does the rest:

- `<run>_0.eqx` — `(grown_network, fresh_opt_state)` tuple. The optimizer
  state is built by the same `make_optimizer(cfg_M7)` + `optimizer.init`
  path `main.run` uses, so the deserialization template matches leaf-for-leaf.
- `<run>_ema_0.eqx` — the grown EMA net.
- `state.json` — schema 2, `global_step: 0`, `curriculum_stage: 0`,
  `run_name` equal to the final launch run name (main.run enforces this),
  `engine_sha` from `detect_engine_sha()`.

On boot, `resolve_latest` finds the seeded set, `download_checkpoint`
restores it, and `main.run` resumes at step 0 with the grown weights and a
step-0 Adam state. `init_checkpoint` / `ema_checkpoint` stay empty.

**Ordering is the safety-critical part**: the seed must be in R2 before the
instance boots. If boot finds no `latest.json` it silently trains a
random-init depth-7 net from scratch. Hence: pick the run name up front,
seed, verify, and only then launch (section 6).

## 5. Deliverables

| File | Content |
| --- | --- |
| `training/joe/grow.py` | `grow_depth(base_net, cfg_new, key)` returning the spliced net; `assert_forward_parity(old, new, n_samples)` doing exact-equality forward checks on masked random inputs |
| `scripts/joe_grow_depth.py` | CLI: `--base-run`, `--run-name`, `--config`; fetches the base set via `resolve_latest` + `download_checkpoint`, grows both lineages, runs the parity check, writes the step-0 set locally, uploads it, verifies `resolve_latest` now returns it. Refuses to run when the target run already has a `latest.json`. |
| `training/joe/configs/M7.yaml` | Section 3 settings; tier name `M7` so `joe_vast_train.py launch --tier M7` resolves it |
| `training/joe/tests/test_grow.py` | Tiny-dims (embed 32, depth 2→3) tests: exact forward parity, splice order, zeroed leaves, EMA path — must stay well inside the 15 s suite budget |
| This doc | The spec |

## 6. Procedure

1. **Implement now** (before the base checkpoint lands): grow.py, CLI,
   config, tests. Gate: `pytest -q` green inside budget.
2. **Preflight** once the base run reaches 50000: `resolve_latest(<base>)`
   shows the step-50000 set; record base run name and checkpoint SHAs in the
   run log section below.
3. **Surgery + local verification**: run the CLI without `--upload`.
   Checks: exact logit/value equality on ≥64 random masked inputs for both
   lineages; param count ≈ 11M; then a local seeded smoke — copy the step-0
   set into a scratch ckpt_dir and run `training/joe/main.py` with the M7
   config and tiny rollout overrides (`--num_envs 8 --num_steps 8
   --minibatch_size 32 --num_iters 2`, small pool) to prove the resume path
   loads the seed and completes an iteration end-to-end.
4. **Seed R2, then launch**: upload; confirm `resolve_latest` returns the
   step-0 set; `python scripts/joe_vast_train.py launch --tier M7
   --run-name <run>`. Per the vast/modal discipline, read the boot log a few
   minutes in and confirm the line `Restoring checkpoint set at global step
   0` — its absence means the scratch-run failure mode and the instance must
   be destroyed before the first checkpoint upload.
5. **Early monitoring**: eval win-rate vs random must start at the base
   run's level (function preservation makes step-0 eval ≈ base eval; a dip
   means the surgery or seed is wrong). Watch KL and the clip fraction over
   the first ~200 iters for LR-too-hot symptoms; fallback is relaunching
   with `lr_power_law_max: 1e-5`.
6. **Interruption drill** (free with vast anyway): after the first
   preemption or a manual `resume`, confirm the logged LR continues the
   schedule rather than jumping — this is the section 3 property.

## 7. Export and the deployment latency gate

Training success does not imply the net can ship. Two gates, in order:

1. **joe-rs depth handling**: confirm the artifact format and the joe-rs
   loader derive layer count from the artifact rather than hardcoding 5.
   If hardcoded, the joe-rs change rides the usual fan-out: exported f16
   artifact (quantize runs first in every export), joe-rs weights,
   unclejoe's copy, the parity corpus, the committed fixture.
2. **CPU latency over a real game**: depth 7 is ~+40% forward cost and
   joe's forward is ~98% of its move. Benchmark the exported bot with the
   existing latency scripts before rating anything; if the 140 ms turn
   budget breaks, the grown net is training-only. Morpheus-side impact
   (fewer leaf evaluations) is a separate measurement.

## 8. Decision measurement

Export the grown bot as a new version and run base-M vs grown-M7 **in one
measurement round** on the fixed grid; quote the pairwise rating contrast
with its interval per the decision rule. No cross-round comparison, no rank
talk. The morpheus contrast, if wanted, is its own round because depth
changes both raw strength and simulation count in opposite directions.

## 9. Risks

- **Forgot-to-seed / seed-after-launch** → silent scratch run. Mitigated by
  CLI ordering, the `latest.json` refusal guard, and the boot-log check.
- **LR too hot for a converged policy** → KL guard trips constantly or Elo
  drops. Fallback cap 1e-5; the seeded set makes relaunching cheap.
- **Adaptation underperforms scratch** — known property of grown models;
  accepted. The arena contrast against base-M is the only claim we make.
- **Latency gate fails** → grown net is unshippable regardless of Elo.
  Known before any arena spend, because the gate precedes the round.

## 10. Run log

- Base run: `joe-M-vast-20260813-0213`, global step 50000, curriculum
  stage 4, last eval wr 0.990, engine `9e3b9d13cca5`.
- Base checkpoint SHAs: full `a704e8d8d747d35bc139f05a497425ac45f126b2d7
  692972f4fc7da9c3815e3a` (102,713,664 B), EMA `900e0e4322ec58eb8b59a6fc
  a6f3fa518c48225ad683cfc3aa4c1f91222aa1cf` (34,237,800 B).
- New run: `joe-M7-vast-20260818-1741`. Surgery 2026-08-18 via
  `scripts/joe_grow_depth.py`.
- Parity: exact logit/value equality on 64 masked inputs per lineage
  (train + EMA), production bf16. Parameters 8,556,250 → 11,514,586.
- Local seeded smoke: resume line `global step 0, curriculum stage 0`;
  step-0 eval 508W/0L/4D (99%) greedy vs random — equal to the base
  run's last eval, confirming function preservation end-to-end. 2
  iterations completed at LR 2.0e-5.
- R2 seeded and verified (`resolve_latest` returns the step-0 set)
  before launch. Launch: adopted instance 47615917 (H100 SXM),
  2026-08-18 ~17:50 CEST.
- Gate 1 (joe-rs depth handling), 2026-08-18: **hardcoded, as section 7
  anticipated.** Four sites: `DEPTH` in `src/nn/net.rs` of both joe-rs and
  unclejoe, and `EXPECTED_LEAVES`/`EXPECTED_PARAMS` in `quantize_artifact.py`
  and `convert_artifact.py` (plus a `range(5)` in the converter's name
  schema). All bumped to depth 7 / 132 leaves / 11,514,586 params. They stay
  pinned rather than derived from the manifest: the pins are what refused the
  depth-7 artifact at every stage instead of mis-loading it, so a tier change
  is a deliberate edit. Step-1500 fan-out done: joe-rs and unclejoe both on
  safetensors b82e8e5aa6e4. Parity 11/11, `mutation_check` 9/9 including both
  net.rs mutants on the edited file.
- Gate 2 (CPU latency), 2026-08-18: **passes.** dev arm64 p99 32.14 ms over
  1,838 turns, against a 150 ms move budget and J3's 50 ms target; 1.34x the
  depth-5 p99, matching the predicted ~+40%. Details in
  [joe-rs/latency.md](../../bots/joe-rs/latency.md). Modal x86 confirmation
  still owed before a rated round.
- First boot crashed at deserialization: the adopted instance's ckpt_dir
  still held the base run's `config.yaml`, and `vast_boot` preferred the
  local file, so the depth-7 seed hit a depth-5 template. The crash came
  before any M7 checkpoint upload; the R2 seed stayed intact. Fixed the
  same day: `vast_boot.fetch_config` now always downloads the run's
  config from R2. Relaunched with `launch --force --instance-id
  47615917` at ~18:15 CEST.
