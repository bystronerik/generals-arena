# Joe vast.ai training plan (interruptible full run)

Phased plan to add a vast.ai training path for `training/joe`, alongside the
existing Modal entry. This plan feeds
[Phase 4 of the Joe plan](averagejoe-competition-plan.md): the one full M
training run. Status: **Phases 0-2 done and verified (2026-08-13). Phase 3
not started**.

Division of labor:

- **Modal** (`scripts/joe_modal_train.py`) stays as-is: fast prototyping,
  smoke runs, benchmarks. It is not removed or restructured.
- **vast.ai** runs the full-scale training run only, chosen for price.
  Interruptible (spot) instances: the instance can be stopped at any moment
  and the replacement can be a different machine. Preemption-safe resume is
  a first-class requirement.
- **Cloudflare R2** (S3-compatible API) holds all durable state:
  checkpoints, run state, logs, eval artifacts. vast.ai has no persistent
  volumes; instance disks die with the instance (or sit unreachable on an
  occupied host).

Both entry points call the same `training.joe.main.run()` /
`training.joe.train.ppo.train()` loop. The vast path adds a storage backend
and a launcher; it does not fork the training loop.

---

## 1. The two resume bugs (fixed in Phase 0, 2026-08-12)

Both bugs break resume on every provider. They are latent on Modal because
Modal runs rarely restart; on interruptible instances a restart is the
normal case, so they block this plan entirely.

### Bug 1 — checkpoints named by local iteration, not global step

`train()` loops `for it in range(cfg.num_iters)` and saves as
`{run_name}_{it + 1}.eqx` (`training/joe/train/ppo.py:412`, `:642-648`).
`cfg.iteration_offset` exists but only shifts the entropy/LR schedules
(`sched_it = it + iter_offset`, `ppo.py:517`); it never reaches the
checkpoint name or `state.json`. A resumed run therefore starts again at
`it = 0` and rewrites `{run_name}_500.eqx`, `_1000.eqx`, … — it overwrites
or misorders every checkpoint the earlier process wrote. The eval cadence,
pool-refresh cadence (`ppo.py:449`), and logger step have the same local-`it`
skew.

**Fix.** One global step, used everywhere:

- `train()` takes a start step and loops
  `for it in range(start_step, cfg.num_iters)`. `num_iters` becomes the
  total target for the run, which matches its current meaning (50k) on a
  fresh start.
- Checkpoint names, `state.json`, schedules, eval cadence, pool-refresh
  cadence, and `logger.log()` all use the global `it`. The separate
  `sched_it` disappears.
- `state.json` gains `global_step` (schema below) and is written
  atomically: write `state.json.tmp`, then `os.replace`. Today a kill
  mid-write can corrupt it (`ppo.py:256`).
- Resume reads `global_step` back from `state.json`. `cfg.iteration_offset`
  becomes a legacy manual override (used only when no state file exists)
  or is removed once nothing references it.

### Bug 2 — curriculum stage resets to 0 on every restart

`train()` hardcodes `current_stage_idx = 0` and `last_eval_wr = 0.0`
(`ppo.py:294-295`). Neither is checkpointed. A resumed run that had reached
stage 2 (distance 6-13) restarts at stage 0 (distance 2-6): it trains on the
wrong map distribution and must re-earn every win-rate gate. The smoke run's
"resume reproduced the pre-kill eval level" held only because the kill
happened while still in early stages.

**Fix.** Persist the full advancement state in `state.json`:

- `curriculum_stage` (the index) and `last_eval_wr` (the only counter that
  gates advancement — the gate is `last_eval_wr >= next.win_rate_threshold`
  at eval iterations, `ppo.py:425-427`).
- On resume, restore both, build `stage = stages[curriculum_stage]`, and
  construct the stage envs from that stage before the first iteration.
- Guard: if the config's curriculum list changed shape since the
  checkpoint (fewer stages than the saved index), fail loudly instead of
  clamping.

### Non-goal: bit-exact resume

Env states, the map pool, and the RNG key are not checkpointed, and the
smoke run already validated statistical resume. On resume the key derives
as `jrandom.fold_in(PRNGKey(cfg.seed), global_step)` so a restarted run
does not replay the same rollout randomness. Bit-exact replay stays out of
scope.

### `state.json` schema (v2)

```json
{
  "schema": 2,
  "run_name": "joe-M-...",
  "global_step": 1234,
  "curriculum_stage": 2,
  "last_eval_wr": 0.61,
  "engine_sha": "...",
  "time": 1765500000.0,
  "files": {
    "full": "joe-M-..._1234.eqx",
    "ema": "joe-M-..._ema_1234.eqx"
  }
}
```

`files` names the exact checkpoint files the state refers to, so a reader
never has to glob and guess which files form a consistent set.

### Tests (before any vast.ai spend)

New file `training/joe/tests/test_resume.py`, `joe`-marked like
`test_train_loop.py` (opt-in; run with `pytest -m joe`, expect ~1 min of
JAX compilation on the tiny env):

- **Kill-and-resume:** run `train()` for 2 iterations with the tiny env
  factory and `save_every=2`; record the byte hash of
  `{run}_2.eqx`; run `train()` again from the same `ckpt_dir` with
  `num_iters=4`. Assert: (a) the loop continued at global step 2 and wrote
  `{run}_4.eqx`; (b) `state.json` shows `global_step == 4`; (c) the hash
  of `{run}_2.eqx` is unchanged — no clobbering.
- **Curriculum restore:** first run uses a 2-stage curriculum with
  `win_rate_threshold=0.0` on stage 1 so the first eval advances to
  stage 1; assert `state.json` records stage 1; resume and assert the run
  starts at stage 1 (log line / state), not stage 0, and `last_eval_wr`
  survived.
- Cheap unmarked unit tests for the state round-trip (write/read/atomic
  replace, stage-index-out-of-range guard) in the default suite — they
  must not add measurable time to the 15 s budget.

---

## 2. R2 storage layer

### Bucket layout

One bucket (working name `joe-training`), one prefix per run:

```
joe/<run_name>/
  manifest.json                    # written once at run start
  config.yaml                      # written once
  checkpoints/<run>_<step>.eqx     # immutable, never overwritten
  checkpoints/<run>_ema_<step>.eqx # immutable
  state/state-<step>.json          # immutable copy per checkpoint
  state/latest.json                # tiny pointer object, written LAST
  logs/metrics.jsonl               # periodic upload (whole-file put)
  logs/train-<boot_id>.log         # one log per instance boot
  lease/heartbeat.json             # instance id + timestamp, ~60 s cadence
  code/<git_sha>.tar.gz            # code delivery (see §3)
```

### Atomicity under preemption

S3-compatible PUTs are atomic per object: a key either holds the previous
complete object or the new complete object, never a partial write. The
corruption risk is therefore not a torn object but a *pointer to an
incomplete set*. The upload protocol removes that risk by ordering:

1. Save locally (tmp + `os.replace`, per §1).
2. Upload the step-named blobs (`.eqx` files, `state-<step>.json`). These
   keys are new — nothing existing is touched.
3. Verify each upload (compare size and SHA-256 stored in object metadata
   against the local file).
4. Only then PUT `state/latest.json`, which references the exact keys and
   checksums from step 2.

A preemption anywhere before step 4 leaves `latest.json` pointing at the
previous complete checkpoint set. A resuming instance reads `latest.json`,
downloads only the referenced keys, and verifies the checksums before
loading. Step-named objects are never overwritten or deleted during a run;
old checkpoints are pruned manually after the run ends.

### Upload cadence vs training overhead

Sizes at tier M (8.56 M params, float32): full checkpoint = network +
2 Adam moments + EMA ≈ **140 MB**; EMA-only ≈ **35 MB**. At the measured
8.9 s/iter, the current `save_every=500` means a full save every ~75 min —
that is also the maximum work lost to a preemption. For the vast run:

- Lower `save_every` / `ckpt_every` to **100-200 iterations** (~15-30 min
  exposure). Uploading 140 MB every 15 min is negligible bandwidth, and R2
  charges nothing for egress and fractions of a cent for the PUTs.
- Uploads run in a background thread fed by the existing `on_checkpoint`
  hook — the same seam Modal uses for `VOLUME.commit`
  (`scripts/joe_modal_train.py:78`), so the training loop never blocks on
  the network. The hook waits for any in-flight upload of the *previous*
  checkpoint before starting the next, so ordering (step 4 above) holds.
- `metrics.jsonl` re-uploads on the same cadence (it is small); the log
  file uploads on a timer.

### Credentials

- Create an **R2 API token scoped to the single bucket** with object
  read/write only. No account-level keys.
- Locally the token lives in an untracked env file (`.env`, gitignored) or
  the shell environment; it is never committed and never written into any
  script, template, or onstart text.
- It reaches the instance as vast.ai **account-level encrypted env vars**
  (`vastai create env-var R2_ACCESS_KEY_ID …`), which the platform injects
  into the container environment — not via `--env` on the command line and
  not baked into the image, so it appears in no shell history, template
  body, or image layer.
- Threat model note: vast.ai hosts are third-party machines; a malicious
  host can read anything the instance can. The scoped token bounds the
  blast radius to this one bucket, and the token is rotated after the run.

### New module

`training/joe/store.py` — a small `R2Store` (boto3 against the R2
endpoint): `upload_checkpoint(files, state)` implementing the ordered
protocol, `resolve_latest(run_name)`, `download_checkpoint(...)` with
checksum verification, `put_heartbeat` / `read_heartbeat`. No training
imports; unit-testable against a tmpdir-backed fake S3 client so the tests
stay in the cheap suite.

---

## 3. vast.ai bootstrap

### CLI mechanics this plan relies on

From the [vast.ai docs](https://docs.vast.ai) (CLI `vastai`, re-verify at
implementation):

- `vastai search offers '<filters>' -o 'dlperf_usd-'` with filters like
  `gpu_name in [H100_SXM, H100_NVL, H100_PCIE] num_gpus=1 verified=true
  rentable=true` (any H100 variant — §7; verify the exact name syntax
  against the CLI at implementation).
- `vastai create instance <offer_id> --image <img> --disk <GB>
  --onstart <file> --ssh --direct`, interruptible type with a bid price.
- Preemption semantics: when outbid (or the host rents on-demand), the
  instance is **stopped, not destroyed** — processes die, the disk
  persists and keeps billing. It may resume on the same machine when the
  bid wins again, or may sit unreachable indefinitely. A replacement
  instance on a different machine is therefore the normal recovery path,
  and stopped instances must be destroyed once replaced.
- The onstart script runs on **every boot**, including a same-machine
  resume after an outbid. Idempotent onstart = the entire resume story.

### Image and code delivery

- **Image:** a stock CUDA 12 base (vast's recommended PyTorch/CUDA images
  work; JAX needs only the NVIDIA driver + pip wheels), with the pinned
  stack from the Modal image (`numpy==2.4.6`, `jax[cuda12]==0.11.0`,
  `equinox`, `optax`, `pyyaml`, plus `boto3`) installed by onstart. Baking
  a custom registry image is a later optimization, taken only if measured
  boot time exceeds ~10 min.
- **Code delivery: tarball via R2, not git.** The launcher packs the repo
  checkout (including the `competition-module` submodule working tree —
  `git archive` skips submodules, so tar the checkout with the same
  excludes as the Modal `add_local_dir` set) as `code/<git_sha>.tar.gz`
  and uploads it. Onstart downloads and unpacks it. This keeps git
  credentials off the instance entirely and pins the exact code the run
  uses. The engine SHA is read locally at launch and pinned into the run
  manifest, exactly as the Modal entry does.
- **JIT cache:** optionally sync `~/.cache/jax` (compilation cache) to
  `joe/<run>/jax-cache/` in R2 at checkpoint time and restore it in
  onstart, mirroring the Volume-backed cache on Modal. Pool-generation
  kernels cost ~40 s cold; nice to have, not required.

### Onstart flow (idempotent)

`scripts/joe_vast_onstart.sh`, parameterized only by `RUN_NAME` and the
injected R2 env vars:

1. Install the pinned pip stack (no-op if already present on a
   same-machine resume).
2. Download and unpack `code/<git_sha>.tar.gz` (skip if present).
3. Read `joe/<run_name>/state/latest.json` from R2.
   - Present → download the referenced checkpoint set, verify checksums,
     start training resumed at `global_step` / `curriculum_stage`.
   - Absent → fresh start from the config.
4. Start the heartbeat writer, run training with the R2 uploader as
   `on_checkpoint`, and stream stdout to the boot log.

### Launcher

`scripts/joe_vast_train.py` — local CLI wrapping `vastai`:

- `launch --tier M --run-name … [--bid …] [--num-gpus N]`: pack + upload
  code, pick an offer (H100, default 1 GPU — §7), create the
  interruptible instance with the onstart script.
- `status`: `vastai show instances` + the R2 heartbeat + last metrics
  line, in one view.
- `resume`: destroy the stopped instance (if any) and launch a
  replacement for the same `run_name` — destroy-before-create is the
  single-writer guard (§4).
- `destroy`: tear down and stop storage billing.

No module-scope repo imports are needed (it runs only locally), but it
follows the same "never pipe through tail/head" logging discipline as the
Modal jobs.

---

## 4. Supervised interruptible run with auto-resume

### How a new instance finds the run

The run's identity is `run_name`; the R2 prefix is the rendezvous point.
Any instance booted with `RUN_NAME=<x>` and the R2 token converges to the
same run via `latest.json` — no state on any particular machine matters.

### Single-writer rule

Two instances uploading checkpoints for one run would interleave steps.
Two guards, both cheap:

- **Destroy-before-create:** the `resume` path (human or watchdog) always
  destroys the old instance before launching a replacement.
- **Heartbeat lease:** the trainer refuses to start if
  `lease/heartbeat.json` is fresher than ~5 min and carries a different
  instance id. This catches the race where a stopped instance un-stops on
  its old host after a replacement already took over.

### Who restarts preempted instances

**Decided (2026-08-12): a local watchdog.** `scripts/joe_vast_train.py
watch` polls `vastai show instances` and the heartbeat every few minutes;
when the instance is stopped/dead and the heartbeat is stale, it runs the
`resume` path (destroy → search offers → create). It runs on the laptop
(or any always-on box) under launchd/cron. The run tolerates watchdog
downtime — it just pauses, and checkpoint exposure stays at ~15-30 min of
work. Manual `resume` remains the fallback; the vast autoscaler is not
used.

### Monitoring

`status` (§3) plus periodic download of `logs/metrics.jsonl` for plots.
The existing stage-transition dip-and-recover shape is expected and
documented in the [Joe plan](averagejoe-competition-plan.md) — don't page
on it.

---

## 5. Phases

### Phase 0 — resume correctness (local, no GPU spend) — DONE 2026-08-12

- **Build:** the two fixes from §1 (`training/joe/train/ppo.py`,
  `training/joe/main.py`, `training/joe/config.py`), the v2 `state.json`
  schema, atomic local writes, and `training/joe/tests/test_resume.py`.
  The state read/write/guard logic landed as its own stdlib-only module,
  `training/joe/state.py`, so the round-trip tests need no JAX import and
  stay in the cheap default suite (`test_resume_state.py`). Resume
  detection lives in `main.run()`: a v2 `state.json` in the checkpoint dir
  wins over `--init-checkpoint`; a v1 file (pre-Phase-0 run dirs) falls
  back to the legacy manual path. Modal auto-resumes by re-launching with
  the same `--run-name`.
- **Verify (runnable):** `pytest -m joe training/joe/tests/test_resume.py`
  green — continuation at the same global step, same curriculum stage, no
  checkpoint clobbering (byte-hash check). Default suite still green under
  the 15 s budget. One Modal smoke resume (`--init-checkpoint` path) still
  works, since Modal shares the loop.
- **Result (2026-08-12):** all three resume tests plus the updated
  `test_train_loop.py` green in 65 s (`pytest -m joe`); default suite
  712 passed with the seven new state tests adding no measurable time.
  The suite as a whole measured ~30 s warm on this machine — over the
  12 s budget before this change too; tracked separately, not caused
  here. (Superseded 2026-08-12: that overage was found and fixed. The
  suite now measures 12.4-14.1 s warm against a 15 s budget. See
  AGENTS.md, "Test suite budget".) Modal resume verified the same
  day with two short tier-M jobs on
  the run `joe-M-resume-test-20260812` (H100, ~5 min GPU total): job 1
  ran global steps 1-10 with full saves at 5 and 10; job 2, re-launched
  with the same `--run-name` and `num_iters=20`, printed the resume
  banner (global step 10, stage 0, last eval wr 2% — job 1's final
  eval), continued at `Iter 11/20`, and saved at 15 and 20. The Volume
  kept all four step-named checkpoint pairs intact and `state.json`
  ended at schema 2, `global_step` 20. The test run dir
  `/vol/joe/joe-M-resume-test-20260812` (~0.6 GB) can be pruned.
- **Gate:** no vast.ai work starts before this phase is green.

### Phase 1 — R2 store + checkpoint/run-state schema — DONE 2026-08-13

- **Build:** `training/joe/store.py` (§2), the background uploader wired
  to `on_checkpoint`, R2 bucket + scoped token, `.gitignore` entry for
  `data/joe/` (local checkpoint dirs are derived data and stay out of
  git; `main.py` already defaults to `data/joe/<run>`).
- **Verify (runnable):** store unit tests against the fake S3 client in
  the cheap suite; one real laptop round-trip against R2 (upload a tiny
  run dir, kill the uploader mid-checkpoint on purpose, confirm
  `latest.json` still resolves to the previous complete set, resume-load
  verifies checksums).
- **Cost:** R2 ≈ pennies (≤10 GB storage, no egress fees).
- **Result (2026-08-12):** `store.py` landed with `R2Store` (ordered
  upload, latest pointer, SHA-256 verification on both directions,
  heartbeat) and `CheckpointUploader` (the background `on_checkpoint`
  hook: joins the previous upload, re-reads `state.json`, mirrors
  manifest/config/metrics, retries a failed upload on the next
  checkpoint). Eleven fake-client tests in
  `training/joe/tests/test_r2_store.py` (renamed from the planned
  `test_store.py` — the basename collides with the arena's
  `tests/test_store.py` under pytest); default suite 687 passed in
  13.8 s. `boto3` added to `requirements-dev.txt`, imported lazily so
  the cheap suite never loads it. The live round-trip
  (`scripts/joe_r2_roundtrip.py` against the real `joe-training` bucket
  with the scoped token in `.env`) passed 2026-08-13: kill-mid-checkpoint
  left `latest.json` on the previous complete set, resume-load verified
  checksums, the retry moved `latest.json` forward, and the script
  cleaned up its throwaway prefix.

### Phase 2 — vast.ai bootstrap — VERIFIED 2026-08-13

- **Build:** `scripts/joe_vast_onstart.sh`, `scripts/joe_vast_train.py`
  (§3), account env vars for the R2 token (`sync-env`), code-tarball packing
  (`training/joe/pack.py`). Instance boot is `training.joe.vast_boot`
  (lease check, heartbeat, `CheckpointUploader`, `main.run`). The vast.ai
  CLI is the pip package `vastai`; the launcher finds `.venv/bin/vastai`
  next to the venv interpreter when it is not on `PATH`.
- **Verify (runnable):** a paid micro-smoke on one cheap interruptible
  GPU (RTX 4090 class, S-config-sized overrides, a few iterations):
  (a) fresh boot trains and lands checkpoints + `latest.json` in R2;
  (b) `vastai destroy instance` mid-run, `resume` launches a different
  machine, and the run continues at the same global step and stage —
  the cloud twin of the Phase 0 test.
- **Cost:** ≈ $1-5.
- **Result (2026-08-13):** cheap suite still green. Paid micro-smoke
  `joe-S-vast-smoke-20260813` on interruptible RTX 4090 completed both
  gates. Gate (a): instance `47570031` wrote `logs/boot.json` (Python
  3.12.13) and `state/latest.json` at `global_step=6` / stage 0, with
  checkpoints at 2/4/6 plus `_final`. Gate (b): `num_iters` raised to 10
  in R2, then `resume` created instance `47570422` on a different
  machine. Logs showed restore at global step 6 / stage 0 / last eval wr
  2%, then `Iter 7/10`. `latest.json` moved to `global_step=10`; step
  2/4/6 blobs stayed in the bucket. Onstart creates a Python 3.12 venv
  via `uv` because the stock PyTorch image is 3.11 and `jax==0.11.0`
  needs 3.12. Both instances were destroyed; the R2 prefix is kept.

### Phase 3 — supervised full run (feeds Joe plan Phase 4)

- **Build:** the watchdog (`watch` subcommand + launchd/cron entry),
  `save_every`/`ckpt_every` lowered to the §2 cadence in the run config.
- **Verify (runnable):** one deliberate kill drill in the first day of
  the real run (destroy the instance, watch the watchdog recover, check
  step/stage continuity in the logs). The run's success criteria are the
  Joe plan Phase 4 ones: curriculum reaches its final stage, in-training
  eval ≥ ~90% vs random at full distance, then the **competition gate**
  (`matchup.py --mode competition` with `PYTHON=.venv/bin/python`) on the
  exported bot. Games go to `data/games/` before any rating refit, as
  always.
- **Cost:** interruptible 1×H100 has historically run at roughly half of
  Modal's ≈ $4/h — ballpark **$120-300** for the 3-5 day M run vs the
  $300-500 Modal estimate, plus ~$0.x/day instance disk. Re-check offer
  prices at launch; the bid level is an open question below.

---

## 6. Files to add or change

| File | Change |
| --- | --- |
| `training/joe/train/ppo.py` | Global step in loop/names/cadences; persist + restore curriculum state; atomic `state.json` |
| `training/joe/main.py` | Resume detection from `ckpt_dir/state.json`; pass start state into `train()` |
| `training/joe/config.py` | Deprecate `iteration_offset` (legacy manual override only) |
| `training/joe/state.py` | **New** — v2 `state.json` read/write/guard, stdlib-only (Phase 0) |
| `training/joe/store.py` | **New** — R2Store: ordered upload, latest pointer, checksums, heartbeat |
| `training/joe/tests/test_resume.py` | **New** — kill-and-resume, curriculum restore, no-clobber (joe-marked) |
| `training/joe/tests/test_resume_state.py` | **New** — state round-trip, atomic replace, stage guard (cheap suite) |
| `training/joe/tests/test_r2_store.py` | **New** — store logic against a fake S3 client (cheap suite) |
| `scripts/joe_r2_roundtrip.py` | **New** — Phase 1 live verification: kill-mid-checkpoint drill against the real bucket |
| `scripts/joe_vast_train.py` | **New** — launch / status / resume / destroy / sync-env (`watch` is Phase 3) |
| `scripts/joe_vast_onstart.sh` | **New** — idempotent boot: deps, code, resolve `latest.json`, train |
| `training/joe/pack.py` | **New** — checkout tarball (working tree, not `git archive`) |
| `training/joe/launch.py` | **New** — offer query, smoke overrides, vastai binary lookup |
| `training/joe/vast_boot.py` | **New** — instance-side lease, heartbeat, R2 resume, `run()` |
| `training/joe/tests/test_pack.py` | **New** — tarball include/exclude round-trip (cheap suite) |
| `training/joe/tests/test_launch.py` | **New** — offer query, smoke overrides, binary lookup (cheap suite) |
| `docs/engine/joe-vast-train.md` | **New** — how to launch; not strategy |
| `scripts/joe_modal_train.py` | Untouched (prototyping path) |
| `.gitignore` | Add `data/joe/*` (+ `.gitkeep` carve-out) |
| `AGENTS.md` | File-placement rows for `data/joe/` and the R2 bucket (at implementation, not before) |

Both entries keep calling `run()` → `train()`; the only difference is the
`on_checkpoint` hook (Volume commit vs R2 upload) and where the checkpoint
dir lives.

---

## 7. Decisions and open questions

Decided (2026-08-12):

1. **GPU type: H100, any variant.** Best price/performance for this run;
   the Phase 1 throughput numbers and the no-OOM-at-80-GB smoke both
   measured it (8.9 s/iter at tier M). SXM, NVL, and PCIe variants count
   as interchangeable — the throughput differences between them are too
   small to matter here, so the launcher matches any H100 and takes the
   cheapest offer. No per-variant benchmarking. 24 GB cards stay out —
   they very likely OOM at `num_envs=2048`.
2. **Single GPU by default; keep multi-GPU working.** The full run uses
   1×H100. The loop already pmaps and `save_checkpoint` stores the
   unreplicated leaves, so a checkpoint moves between 1- and N-GPU
   machines. The launcher takes `--num-gpus` (default 1) and passes
   `num_gpus=N` to the offer filter; nothing else changes. Constraint to
   preserve: `minibatch_size` divides the per-device sample count for any
   supported N. Multi-GPU interruptible offers are scarcer and one
   preemption costs the whole node — use only if wall-clock (3-5 days)
   becomes the problem.
3. **Restarts: the local watchdog** (§4). Manual `resume` is the
   fallback; the vast autoscaler is not used.

Open:

1. **Bid level.** Bid amount trades price against preemption frequency;
   with 15-30 min checkpoint exposure, frequent preemption is cheap but
   not free (boot + JIT warm-up per restart). Pick after watching offer
   prices for a day or two.
2. **Same-machine stopped instances.** After a replacement launches, the
   old stopped instance still bills disk until destroyed —
   destroy-before-create handles it, but a `watch` sweep for orphaned
   stopped instances is cheap insurance.
