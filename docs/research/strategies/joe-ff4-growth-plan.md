# Joe ff×4 growth plan

Status: draft 2026-08-19. Graft an ff×4 FFN (1152 → 1536 hidden units per
block) onto the depth-7 lineage by function-preserving column insertion,
then continue PPO on vast for ~20k iterations as a new run. Companion to
[joe-depth7-growth-plan.md](joe-depth7-growth-plan.md); the machinery
(fabricated step-0 seed, schedules keyed from 0, boot-log discipline) is
identical and is not re-argued here — only the deltas are.

## 1. Seed choice: the M7 lineage, not the M base

The seed is the **newest complete checkpoint set of
`joe-M7-vast-20260818-1741`** at the moment that run is stopped — not the
base `joe-M-vast-20260813-0213` step-50000 set. Decided 2026-08-19:

- Future cost is identical either way (~20k iters at depth-7/ff×4 cost);
  the seed only determines what the run starts from.
- The M7 set is the M base plus its depth-adaptation iterations at a sane
  LR — strictly more training for the same money already spent. Discarding
  it buys only single-surgery provenance, which is not worth paid H100 time.
- Growth surgeries compose across runs: an FF graft onto a partially
  adapted depth-7 net is exactly as function-preserving as one onto the
  base.

## 2. Growth operation

Per block, for all 7 blocks (this differs from the depth graft: FF growth
edits **every existing layer** instead of adding self-contained new ones):

- `ff_linear1`: 384 fresh-init rows appended (weight rows + bias entries)
  — the new hidden units. Fresh init from a depth-7/ff×4 `build_network`
  template, new PRNG key.
- `ff_linear2`: 384 **zeroed** columns appended; bias unchanged. The new
  hidden units contribute exactly zero to the block output, so the net is
  function-preserving at init. Gradients reach the zeroed columns
  immediately (their input activations are nonzero), which recruits the
  new units; no duplicate-neuron symmetry exists because the `ff_linear1`
  rows are random, not copied.
- Everything else — attention, norms, embedder, heads, temporal encoder,
  `pos_encoding` — is copied unchanged. Token count and embed width do not
  change.
- **Both lineages are grafted**: training net from the checkpoint tuple
  and the EMA net. Params 11,514,586 → 13,581,658 (+7 × 2×384×384 + 7×384
  bias). Leaf count stays 132.

**Parity caveat, sharper than the depth case.** The depth graft left
shared-layer GEMM shapes untouched, so un-jitted parity was bitwise. Here
the FFN GEMMs change shape (1152 → 1536 reduction/output dims), and a
different shape may tile the *nonzero* partial sums differently, moving
results by an ulp even eager. The check therefore: attempt bitwise first;
on failure, accept max |Δlogit| and |Δvalue| at f32-epsilon scale (≤1e-5)
**plus** argmax-action equality on every one of ≥64 masked inputs, per
lineage. Do not chase XLA bit-exactness across different-shape programs —
same lesson as the depth port and joe-rs.

## 3. Continuation config: `configs/M7F4.yaml`

Copied from `M7.yaml` with:

| Knob | M7F4 setting | Why |
| --- | --- | --- |
| `ff_factor` | 4 | The graft |
| `num_iters` | 20000 | Same adaptation budget; the still-young depth blocks and the new FF units adapt together |
| LR | cap 2e-5, floor 5e-6, same power law from 0 | The M7 run is stopped inside its flat 2e-5 phase, so continuation = the same cap; ~10k flat then decay |
| `ent_coef_start` | the M7 run's ent value at its stop step (~0.0014 at step ~2000), floor 0.001 | Plain continuation, as M7 continued the base run's 0.0058 |
| `seed` | 47 | Fresh rollout randomness |

Run name `joe-M7F4-vast-<stamp>`; single-stage curriculum, step-0 seed
set, fresh optimizer — all exactly as the depth plan sections 3–4.

## 4. Deliverables

| File | Content |
| --- | --- |
| `training/joe/grow.py` | Add `grow_ff(base_net, cfg_new, key)`; extend the parity helper with the tolerance + argmax fallback of section 2 |
| `scripts/joe_grow_depth.py` | Generalize (or sibling `joe_grow_ff.py`): same fetch → graft → parity → seed → verify pipeline, `--op ff` |
| `training/joe/configs/M7F4.yaml` | Section 3; tier name `M7F4` for `launch --tier M7F4` |
| `training/joe/tests/test_grow.py` | Tiny-dims FF-graft cases: preservation within tolerance + argmax equality, zeroed-column placement, fresh-row independence, EMA path; inside the 15 s budget |
| This doc | The spec |

**As built, 2026-08-19 (procedure step 1).** `scripts/joe_grow_depth.py` is
generalized in place, not duplicated: `--op {depth,ff}` selects the graft,
the default target config follows the op (`M7.yaml` / `M7F4.yaml`), and
`--expect-params 13581658` makes the step-3 parameter check a refusal
rather than a reading. `assert_forward_parity` takes `atol=None` (bitwise,
depth) or a float (bitwise-first, then tolerance + argmax, ff) and returns
the run-log numbers. `M7F4.yaml` carries `ent_coef_start: 0.0013` — the
step-~2000 value of the M7 schedule `0.006 / (step + 1) ** 0.2`; recompute
it from the confirmed stop step of procedure step 2 before the surgery.

Pre-surgery local check at production dims (fresh-init nets, depth 7,
ff×3 → ff×4): parameters 11,514,586 → 13,581,658 and 132 leaves exactly as
section 2 predicts, and parity came out **bitwise** on this host despite
the shape change. The tolerance path stays as the documented fallback —
the real surgery runs on other hardware, where the tiling may differ.

## 5. Procedure

1. **Implement + tests** while the M7 run keeps training (each extra
   depth-only iteration is kept in the seed, so there is no deadline
   pressure).
2. **Stop the M7 trainer, keep the instance.** Instance 47615917 is
   **reused** for the M7F4 run, not destroyed. Stop the trainer process
   yourself — nothing else does: `kill -INT` the `vast_boot` pid (exits
   through its `finally`: uploader join, heartbeat stop), confirm with
   `nvidia-smi --query-compute-apps`, and do it just after a checkpoint —
   everything since the last confirmed set is lost. Record the final
   confirmed step; `resolve_latest` names the seed set.
3. **Surgery + local verification**: graft both lineages; parity per
   section 2; param count 13,581,658; local seeded smoke (resume line
   `global step 0`, step-0 greedy eval equal to the M7 run's last eval,
   2 tiny iterations).
4. **Seed R2, then adopt-launch onto the same instance**:
   `python scripts/joe_vast_train.py launch --tier M7F4 --run-name
   joe-M7F4-vast-<stamp> --instance-id 47615917`. The launch verb packs
   fresh code, uploads the M7F4 config and `launch.json`, re-labels the
   instance, and runs the adopt onstart — and its branch has **no destroy
   call at all** (`resume` is the verb that destroys; never use it here).
   `--force` is not needed: the new run has no `launch.json` yet. Read the
   **tail** of `/workspace/joe-adopt-onstart.log` (it appends across
   boots; grepping from the top matches stale content) and confirm
   `Restoring checkpoint set at global step 0` plus a fresh
   `repo/.joe-code-sha` matching the new tarball.
5. **Protect the instance from old-run tooling.** The M7 run's R2
   `instance.json` still records 47615917, and `_instance_ids_for_run`
   matches on that record even after the label changes — so
   `resume --run-name joe-M7-…` (destroy-before-adopt) or
   `destroy --run-name joe-M7-…` would destroy the shared instance.
   **Never run either for the M7 run while the instance lives.** Optional
   hardening right after adoption: overwrite the M7 `instance.json` with
   an empty id (`store.put_instance("joe-M7-vast-20260818-1741", "")`) so
   the record no longer names the instance; the M7 checkpoints and state
   stay untouched either way.
6. **Early monitoring**: step-0 eval at seed level; KL/clip-fraction watch
   over the first ~200 iters; fallback relaunch at cap 1e-5.
7. **Interruption drill** on first preemption: LR continues the schedule.

## 6. Export gates

1. **Pinned shape sites** (the depth port confirmed they refuse rather
   than mis-load): `FF_DIM 1152 → 1536` and the manifest check
   `ff_factor 3 → 4` in `bots/joe-rs/src/nn/net.rs` **and**
   `bots/unclejoe/src/nn/net.rs`; `EXPECTED_PARAMS → 13,581,658` in
   `bots/joe/tools/quantize_artifact.py` and
   `bots/joe-rs/tools/convert_artifact.py` (+ the converter's FF dims in
   `expected_schema`; `EXPECTED_LEAVES` stays 132); the net.rs header
   docs and `docs/bots/joe-rs/export.md`. Then the normal export
   sequence, parity corpus, unclejoe copy, committed fixture —
   the full fan-out.
2. **Latency: already measured, gate passed.** 2026-08-19, Modal x86, one
   core, synthetic ff×4 artifact through the patched loader, the
   1,838-turn recorded wire log: p50 38.0 / p90 39.8 / **p99 40.95 / max
   46.1 ms** — inside the 150 ms limit and the 50 ms J3 target (tripwire
   75). Local arm64 cross-check 37.2 ms p99 vs 32.1 ms for ff×3. Re-run
   on the real exported artifact is cheap and confirms nothing changed,
   but the shape itself is cleared. Morpheus leaf-budget impact (+20%
   forward → fewer simulations) is a strength question for the arena, not
   a budget gate.

## 7. Decision measurement

Primary contrast: exported M7F4 bot vs the exported depth-7 bot, both arms
in one round, pairwise contrast with interval per the decision rule — this
isolates the FF graft given depth 7. A secondary same-round arm against
base-M measures the combined growth if wanted. The morpheus contrast is
its own round.

## 8. Risks

- **Parity is tolerance-based, not bitwise** (section 2) — a real wiring
  bug still fails loudly: argmax disagreement or Δ far above epsilon, and
  the step-0 greedy eval equals the seed's eval only if the graft is
  right.
- **Graft touches all 7 layers** — a per-layer splice bug hits every
  block; the tiny-dims tests pin column placement per layer.
- **Two trainers on one GPU** if the M7 trainer is not stopped before the
  adopt: the onstart never kills an existing process, and the lease guard
  does not fire on a same-instance adopt — both trainers would run and
  both would write `latest.json`. Procedure step 2 orders the manual stop
  first.
- **Accidental destroy via old-run tooling** — the M7 `instance.json`
  still names 47615917 (procedure step 5). No `resume`/`destroy` for the
  M7 run while the instance lives.
- **Adaptation budget shared** between young depth blocks and new FF
  units; 20k may land short of a scratch depth-7/ff×4 net. Accepted; the
  arena contrast is the only claim.

## 9. Run log

- Export gate 1 (pinned shape sites), 2026-08-19: **done.** `FF_DIM 1152 ->
  1536` and the manifest cross-check `ff_factor 3 -> 4` in both crates;
  `EXPECTED_PARAMS -> 13,581,658` in `quantize_artifact.py` and
  `convert_artifact.py`; the converter's FF dims now read a module-level
  `FF_DIM = 1536` instead of literals; `EXPECTED_LEAVES` stays 132 as section
  2 predicted. Added the `M7F4` row to `test_tier_parameter_counts`, which
  covered S/M/M7 only. Full fan-out done: joe-rs and unclejoe both on
  safetensors ba3f7da75793. Parity 11/11 over a rebuilt 15-member corpus,
  `mutation_check` 9/9 including both net.rs mutants on the edited file, tier-2
  pins unmoved (worst 43% of pin).
- Export gate 2 (latency) confirmed on the real artifact, 2026-08-19: dev
  arm64 p99 **37.87 ms** over 2,400 turns, within 2% of the 37.2 ms this host
  was predicted to show. Details in
  [joe-rs/latency.md](../../bots/joe-rs/latency.md).
- Note for section 7: joe-rs now carries M7F4, so the depth-7 arm the primary
  contrast needs is no longer live. It is reconstructible — the M7 f32 original
  is `data/joe/ema.f32.26f4b06154de.eqx` and f16 rounding is deterministic.

- **M7 stop (procedure step 2), 2026-08-19 ~02:03 CEST.** `kill -INT` on
  the `vast_boot` pid; the process left through its `finally`
  (`vast_boot finished` in the log), `nvidia-smi --query-compute-apps`
  came back empty, and the instance stayed up. Final confirmed step
  **2000**, curriculum stage 0, last eval wr 0.994140625, engine
  `9e3b9d13cca5`. The local log had reached iter 2092, so 92 unconfirmed
  iterations were discarded — the cost the procedure accepts for stopping
  just after a checkpoint. Waiting for the step-2500 checkpoint instead
  would have bought 500 depth-only iterations for ~83 minutes of H100
  time.
- **Seed set SHAs**: full `4bb4b4e54fb0e0b32f959d00b9bbf975d415695aae51f4
  8236b9cf73d38a5e4e` (138,225,984 B), EMA `a8ce1b89d134de2a1c3ab0a38d6cc
  2f09fb9adf046c11679ace26646581cd364` (46,075,240 B).
- **Entropy recomputed** from the confirmed stop step, as section 4 asks:
  `0.006 / (2000 + 1) ** 0.2 = 0.0013119`, and the trainer logged
  0.0013120 at step 2000. `M7F4.yaml` already carried
  `ent_coef_start: 0.0013`, so no edit was needed.
- **New run**: `joe-M7F4-vast-20260819-0207`. Surgery 2026-08-19 via
  `scripts/joe_grow_depth.py --op ff --expect-params 13581658`.
- **Parity: bitwise**, both lineages, 64 masked inputs each — max abs
  diff 0 for logits, value, and value_aux; argmax agreement 1.0000. The
  tolerance path of section 2 was not exercised. This does **not** retire
  it: the surgery runs on the same local host as the pre-surgery check of
  section 4, so this is the same tiling, not a second one. Parameters
  11,514,586 → 13,581,658, the count `--expect-params` demands.
- **Local seeded smoke**: resume line `global step 0, curriculum stage 0`
  and `Parameters: 13,581,658`; step-0 eval 509W/0L/3D (99%) greedy vs
  random over 512 games — equal to the M7 run's last eval, so function
  preservation holds end-to-end; 2 iterations completed at LR 2.0e-5.
- R2 seeded and verified (`resolve_latest` returns the step-0 set) before
  the launch.
- **Launch**: adopted instance 47615917 (H100 SXM), 2026-08-19 ~02:07
  CEST, `launch --tier M7F4 --instance-id 47615917`. Code tarball
  `4b3e3e54d687…` packed from a clean tree; the instance's
  `repo/.joe-code-sha` `cd16281075…` equals `launch.json`'s
  `code_sha256`. Boot log tail: `Lease acquired`, `Restoring checkpoint
  set at global step 0, curriculum stage 0`, `Parameters: 13,581,658`,
  `Devices (1): [CudaDevice(id=0)]`. The fetched `config.yaml` shows
  `depth: 7`, `ff_factor: 4`, `seed: 47`, `num_iters: 20000`,
  `ent_coef_start: 0.0013`, `lr_power_law_max: 2.0e-05`.
- **Hardening (procedure step 5)**: the M7 run's `instance.json` now
  records an empty instance id, so `_instance_ids_for_run` no longer
  names 47615917 for that run; the relabel closed the label match at the
  same time. The M7 checkpoints and state are untouched.
- **Step-0 eval on the instance**: 508W/0L/4D (99%) greedy vs random over
  512 games — the seed's level, no dip. It is one game away from the M7
  number because the eval key and the map pool differ per run; the local
  smoke is the exact-equality check.
- **First launch OOM (procedure step 6), 2026-08-19.** The run completed
  20 iterations at a steady 12.86 s and then died inside `it=20`, the
  first `reset_pool_every` boundary. The BFC allocator refused a
  12.94 GiB request. JAX surfaced the error at the next blocking read
  (`float(metrics["approx_kl"][0])` in `train/ppo.py`), so the traceback
  line is not the cause. The allocator warning came about 40 minutes
  after the iteration-20 line — the run stalled before it failed. No
  checkpoint existed yet (`ckpt_every` 500), so R2 still held only the
  verified step-0 seed and the relaunch gave up only those 20 iterations.
- **Cause**: two allocations stack at the refresh. `train/ppo.py` bound
  the new 200k-map pool before it dropped the old `pool` and `pool_rep`,
  and JAX preallocates 75% of the card — the M7 trainer measured
  61,452 MiB of an 81,559 MiB H100, so ~20 GiB never entered the arena.
  ff×3 fits with the transient double pool; ff×4 does not. Every joe run
  carried the double pool; only the wider net made it fatal.
- **Fix, both numerically inert.** `scripts/joe_vast_onstart.sh` exports
  `XLA_PYTHON_CLIENT_MEM_FRACTION=0.92`, set after the
  `/etc/environment` copy so it stays scoped to the trainer;
  `train/ppo.py` releases the old pool and its replica before it builds
  the new ones. No PPO, environment, or network knob moves, so the
  section 7 contrast against the depth-7 bot holds. `pytest -q`: 525
  passed in 13.7 s.
- **Relaunch**: `launch --tier M7F4 --instance-id 47615917 --force` onto
  the same instance. Code `4b3e3e54d687…-dirty-88d9b61b7456`; the
  instance's `.joe-code-sha 88d9b61b7456…` and
  `XLA_PYTHON_CLIENT_MEM_FRACTION=0.92` confirmed in the trainer's own
  environment. The arena is now 75,168 MiB, up from 61,452 MiB — more
  headroom than the 12.94 GiB that failed.
- **The fix is inert, measured, not assumed**: the relaunch reproduced
  the step-0 eval (508W/0L/4D) and iterations 6, 7, and 8 to every logged
  digit — loss 1.6982 / 1.6825 / 1.6792, KL 0.0158 / 0.0167 / 0.0152,
  GNorm 92.88 / 55.50 / 61.96 — the values the first launch logged before
  it died. Dropping a Python reference moves no number.
- **Procedure step 6 verdict, iterations 0–210 after the relaunch.** The
  run cleared every pool refresh (20, 40, …, 200) with no allocator
  warning, no stall, and a flat 12.86 s/iteration — versus M7's 12.05 s,
  so the ff×4 graft costs ~7% wall clock per iteration, not the ~20% the
  forward-cost estimate of section 6 implied. The OOM fix holds.
- **KL and clip are flat**: KL min 0.0085, mean 0.0155, max 0.0270; clip
  fraction 0.07–0.12, mean 0.10. Both sit in M7's band (M7 ran KL
  0.0115–0.0229). No LR-too-hot signature appeared, so the cap-1e-5
  fallback of procedure step 6 was not used.
- **`target_kl: 0.02` is inert in this config**, and this is a lineage
  property, not an M7F4 one: `num_epochs` is 1, so the `target_kl` break
  in `train/ppo.py` leaves a loop that ends after one pass anyway. 26 of
  210 iterations logged KL above 0.02 with nothing to catch them. The LR
  cap is the only real guard on this recipe.
- **Greedy eval vs random (512 games)** at iterations 0/49/99/149/199:
  508 / 511 / 508 / 510 / 501 wins, and **zero losses at every point**.
  The seed level holds. Note the eval saturates near 100% and so detects
  a collapse, not a small regression; the section 7 arena contrast stays
  the only real claim.
- **Self-play behaviour oscillates**, and this is the one M7F4-specific
  signal. Castles, draw rate, and episode length rise together and then
  relax on a ~30–40 iteration cycle: castles 0.52–7.58 (mean 2.12), draws
  1–35% (mean 8.3), eplen 377–790. M7's depth graft never left 0.4–3.2
  castles or 12% draws through 2000 steps. The 501/512 eval at iteration
  199 fell in a castle-heavy phase and its shortfall is 11 draws, not one
  loss — the eval tracks the phase rather than a regression. Watch
  whether the amplitude grows.
- **Do not read the per-iteration W/L/D split as strength.** Both seats
  run the same net (`collect_rollout` takes a single `net`), so the split
  is symmetric by construction, and episode length grows against a fixed
  256-step rollout window — inside an excursion the completed-episode
  count falls from ~1250 to ~760 and the percentages are computed over a
  shrinking, early-ending subset. Read the eval instead.
- gnorm runs 42–335 (mean 129) against M7's 60–120. It is the pre-clip
  norm and `max_grad_norm` is 0.267, so both runs are scaled down hard.
