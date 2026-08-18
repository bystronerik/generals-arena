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

Fill in during execution: M7 stop step + seed set SHAs, new run name,
surgery date, parity numbers (max Δlogit, Δvalue, argmax agreement), launch
instance id, step-0 eval.
