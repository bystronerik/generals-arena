# Joe distill-then-RL plan (joe-MD)

Status: draft 2026-08-19. Train the full-size architecture (depth 7,
ff ×4) from random init by **distilling the mature joe-M policy into it**,
then continue PPO from the distilled checkpoint as a new run. This is the
alternative to waiting out the growth path's recruitment grind.

Evidence base:
[joe-m7f4-update-lever-bench.md](../measurements/joe-m7f4-update-lever-bench.md)
— the grown M7F4 run is signal-limited: the surgery-added capacity (blocks
2/5 at ~10 % of mature scale, ff ×4 units at ~5 % after 4k steps) recruits
at a pace no global update lever changes (a second PPO epoch adds √2
diffusion; a 1.5× LR cap adds drift without recruitment). Distillation
attacks the diagnosed limit directly: a dense, batch-to-batch **coherent**
gradient (match the teacher everywhere) reaches every parameter from step
one, and the student learns the teacher's function in its own basis — no
zero-init-protected units, no recruitment regime.

Naive alternatives rejected: transplanting trained parts between
independently trained runs fails on basis mismatch (independently trained
nets have permutation-scrambled internal representations; spliced parts
receive activations they never trained on). A full from-scratch RL run at
the target size costs the most and discards the 50k-iteration joe-M
investment.

## 1. Teacher and student

- **Teacher: the joe-M final EMA** (`joe-M-vast-20260813-0213`, depth 5,
  ff ×3, 8.56M params) — the same artifact M7F4 uses as its frozen eval
  reference. Not the M7F4 EMA: it is at parity with joe-M (ref eval
  44–50 % throughout), its half-recruited units add nothing, and a
  single-provenance teacher is cleaner.
- **Student: depth 7, embed 384, n_head 8, ff ×4, patch 3** (13,581,658
  params — the M7F4 target shape), fresh random init, new seed. Same
  observation, mask, and action head as every joe net.
- The teacher plays with **sampled** actions (softmax, as in training
  rollouts), not greedy: coverage of the state distribution it actually
  generates. Both seats are the teacher, exactly like self-play.

## 2. Phase B — distillation

Online, no corpus storage: each iteration runs the existing self-play
rollout machinery with the **teacher** as the acting policy, then updates
the student on every state of the batch. Storing observations at corpus
scale (~34 KB/sample) is a non-starter; online distillation needs only
the existing iteration buffer.

Losses, per state (student and teacher see identical masks):

- **Policy**: cross-entropy −Σ p_T log p_S over the masked 4410-way
  action softmax (teacher probabilities renormalized on the mask).
- **Value**: cross-entropy −Σ q_T log q_S over the 128 HL-Gauss bin
  softmax — distribution matching, not the scalar. Environment returns
  are not used in this phase; Phase C reintroduces them.
- Total = policy + 0.5 × value (`vf_coef` unchanged). No entropy bonus,
  no advantages, no GAE, no `adv_top_frac` filter, no PPO ratio.

Training setup: Adam, constant LR 1e-4 (the fresh-run RL cap; supervised
training tolerates it), grad clip 0.267, batch 2048 envs × 256 steps × 2
seats, minibatch 2048, one pass per batch. Fresh data is unlimited and
the teacher forward is the marginal cost; if teacher compute ever
dominates, a second supervised pass per batch is a legitimate cost lever
(unlike PPO, supervised reuse does not go stale). EMA (0.999) is kept —
deployment convention and the eval subject.

Gates, on the `eval_every` cadence (50 iters):

| Gate | Instrument | Target |
| --- | --- | --- |
| (a) agreement | student argmax = teacher argmax on fresh states | ≥ 90 %, logged always |
| (b) head-to-head | student EMA greedy vs teacher greedy, 512 games (existing `evaluate_vs_ref`, ref = teacher) | ≥ 48 % on 3 consecutive evals |
| (c) sanity | greedy vs random | ≥ 98 % once (a) > 50 % |

**Exit**: gate (b) satisfied → the newest full checkpoint set seeds
Phase C. **Abort/reassess**: no upward agreement slope in the first ~200
iters, or 5×10⁹ samples consumed without gate (b).

Cost model: per sample ≈ teacher forward (0.65× student) + student
forward/backward (3×) ≈ 2× the RL pipeline → ~40k samples/s on the H100,
~6.5 h per 10⁹ samples. Literature-typical distillation needs ~1–10 % of
the teacher's RL sample budget (joe-M: ~5×10¹⁰), so 0.5–1.5 days
($25–75 interruptible). The first GPU-hour shows the agreement slope —
the cheap kill-point.

## 3. Phase C — RL continuation

A new run seeded at step 0 from the distilled set, exactly like the
growth continuations (fabricated step-0 seed, schedules keyed from 0):

- M7F4's continuation settings verbatim where they apply: 1 epoch,
  `adv_top_frac` 0.25, LR power law capped at 2e-5, entropy continuation
  (`ent_coef_start` 0.0013), `num_iters` 20000, fresh seed.
- The frozen-reference eval keeps the teacher as reference — it now reads
  as a true improvement gauge: sustained > 50 % means the student
  exceeded the teacher.
- Strength verdicts come from arena export contrasts at the r-checkpoint
  cadence, same-round pairwise per
  [decision-rule](../../arena/decision-rule.md), never from the in-run
  eval alone.

Budget: 1.5–3 days ($75–150). Total plan: ~$100–225 GPU.

## 4. Code deltas

| Piece | Change |
| --- | --- |
| `training/joe/train/distill.py` | new: teacher rollout + supervised update loop, agreement gate; reuses `_observe_both`/augment/pool machinery |
| `training/joe/config.py` | `train_mode: ppo\|distill` (default ppo), `teacher_checkpoint`, `teacher_config` |
| `training/joe/main.py` | mode dispatch; teacher built from its own config (depth-5 teacher, depth-7 student coexist) |
| `training/joe/vast_boot.py` | fetch teacher files from the run prefix — same mechanism as the `eval_ref_*` fetch; set both to the same files |
| `training/joe/configs/MD.yaml` | Phase B config |
| `training/joe/configs/MD-RL.yaml` | Phase C config |
| tests | distill-loss math on synthetic distributions, mask renormalization, mode dispatch, gate accounting — CPU only, inside the 15 s budget |

## 5. Procedure

1. Code + tests; local CPU micro-smoke of the distill loop on a tiny env.
2. Commit; 4090 `--smoke` with `train_mode: distill` — agreement must
   climb from the masked-random baseline within the smoke iterations.
3. Seed R2 prefix `joe-MD-...`: teacher files under `ref/`, `config.yaml`
   via the launcher.
4. Launch H100. Check the boot log and the first agreement/gate lines
   within the first hour before leaving it (modal-jobs discipline).
5. Run to gate (b); stop; record the measurement doc
   (`joe-md-distill.md`: agreement curve, samples used, h2h trajectory).
6. Seed Phase C from the distilled set (grow-tool copy path); launch with
   `MD-RL.yaml`.
7. Arena exports at the r-cadence; leaderboard per the evaluator flow.

## 6. Hedge and open questions

- **Hedge**: M7F4 keeps training at the plan's 2e-5 through Phase B (its
  revert relaunch is pending at step 4000). Retire it only on an arena
  contrast, not on in-run evals.
- **Open — DAgger-style mixing**: if h2h stalls below gate (b) while
  agreement is high, mix ε of student actions into the rollout policy
  late in Phase B (distribution-shift correction). Not in scope for the
  first attempt.
- **Open — Phase C entropy**: 0.0013 assumes the distilled policy lands
  near the teacher's entropy (~0.32). Verify on the distilled checkpoint
  before Phase C; a materially sharper student may want a brief higher
  floor.
