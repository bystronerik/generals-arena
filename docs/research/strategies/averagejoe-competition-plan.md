# Average Joe reimplementation plan (competition mode)

Phased plan to replace the current Morpheus NN training workflow
(`training/morpheus/` + `scripts/morpheus_*`) with a reimplementation of the
Average Joe pipeline: pure self-play PPO in a JAX-vectorized env, trained end
to end on GPU, under **competition rules**.

Status: **plan only — nothing implemented**. Phase 0 decisions below block
implementation start.

Sources of truth, in order:

1. Paper: *Superhuman AI for Generals.io Using Self-Play Reinforcement
   Learning* (arXiv:2606.23348, Straka/Lisý/Schmid). Read in full 2026-08-11.
2. Released code: `~/Work/learning/AverageJoe` (train/, networks/, configs/).
3. This repo's `RULES.md` + `GeneralsEnv(mode="competition")` for everything
   the competition changes.

---

## 1. Why this is a smaller lift than it looks

`competition-module` **is** `strakam/generals-bots` — the exact JAX env the
paper released, with competition rules layered on as modifiers
(`build_castles`, `deathtouch`, `strip_neutral_castles`) and pinned by
`GeneralsEnv(mode="competition")`. The AverageJoe training code imports
`generals.core.env`, `generals.core.game.get_observation`,
`generals.core.action.compute_valid_move_mask` — the same modules, same
stateless pool API (`reset()` → pool, `step(state, actions, pool)`), that our
submodule carries.

So there is **no domain-transfer problem and no second engine**: we train
directly in the competition ruleset, in the paper's own env family, and the
"which env do we train in" question has a one-word answer: this one. The port
is (a) adapting AverageJoe's rollout/PPO/network code to our fork's small API
drift, (b) extending the action space for castle-building, (c) building a
CPU deployment path that survives the 150 ms / 1-core / 2 GB match limits.

## 2. The recipe, component by component

Confirmed against both paper and code. "L" = the released `configs/L.yaml`
(the paper-scale config); S/M are the smaller tiers we scale from.

| Component | Paper | Released code | We adopt |
| --- | --- | --- | --- |
| Algorithm | PPO, clipped surrogate, entropy bonus | `train/ppo.py`; `target_kl=0.02` early-stop | same |
| Opponent | pure self-play, one shared policy both seats | `rollout_selfplay.py` — both seats in one 2N batch | same — **no league, no opponent pool** |
| Reward | sparse terminal ±1, zero for draws | `rewards.win_lose_reward`, hardcoded in `ppo.py` | same — **no shaping** |
| Network | ViT: 3×3 patches → tokens; +2 temporal tokens (opponent army/land 512-step windows via MLPs); +1 value token; policy head H×W×9, value head over HL-Gauss bins | `networks/transformer.py` `HistoryTransformer`, `networks/common.py` (38-ch augmented obs, `pos_encoding`, bf16) | same, with a **10th per-cell action for build** (§4) |
| Value loss | categorical HL-Gauss, 128 bins, range [−1,1], σ=0.04 | `make_value_loss_fn` (`value_loss: ce`) | same (σ=0.04 — the `config.py` default 0.75 is a trap) |
| Advantage | GAE, γ=1.0, λ=0.9; advantages normalized cross-device | `compute_gae` with truncation-aware carry zeroing + train-mask | same |
| Filtering | top 25% of transitions per batch by predicted advantage | `top_k(|adv|)` — **absolute value** (see §3) | code's |adv| (decided, D3) |
| EMA | τ=0.999 on policy params; **EMA is the deployment policy** | updated per-iter on CPU, saved as `_ema.eqx` | same |
| Curriculum | spawn-distance cap raised from 4 to full | staged: distance windows gated on eval win-rate vs random ≥ 0.6 (`CurriculumStage`) | code's staged version |
| Schedules | LR clip(0.5·t^-1.1, 5e-6, 1e-4); entropy 0.05·t^-0.2; paper says shape barely matters | `power_law` for both | same |
| Batch (L) | 512 envs × 512 steps × 2 seats/iter; minibatch 1024; **1 epoch/iter**; grad-clip 0.267 | matches L.yaml | same, tier-scaled |
| Map pool | — | 200k pre-generated states, regenerated every 10–20 iters (`reset_pool_every`) | same |
| Eval | win-rate vs random (curriculum gate) + round-robin Elo vs frozen reference checkpoints | `train/evaluations.py`, `evals/ref_eval.py` | same in-training; arena eval only on exported bots |

Tier template (from released configs, net sizes only — board is always ours):

| Tier | depth / embed / ff / patch | ≈ params | Notes |
| --- | --- | --- | --- |
| S | 4 / 352 / ×2 / 2→**3** | ~5M | S.yaml uses `adv_top_frac: 0.5`, patch 2 |
| M | 5 / 384 / ×3 / 3 | ~8M | `gae_lambda: 0.7` in M.yaml (L uses 0.9) |
| L | 7 / 448 / ×3 / 3 (paper) | 15.35M | see disagreement D1 |

## 3. Paper vs released code — disagreements (not silently resolved)

- **D1 — torso depth.** Paper Table II: depth 7, 15.35M params (the math
  checks out: ~2.0M/layer × 7 + embedder + temporal ≈ 15.3M; the ladder
  account is literally `L_7d_gae90_30k_ema`). Released `L.yaml`: `depth: 11`
  (~22M) — apparently a post-paper config. **Decided 2026-08-11: paper's
  depth 7** — matches the deployed agent and our CPU budget.
- **D2 — entropy vs magnet.** Paper: tried regularizing toward heuristic
  policies, "observed no improvement, so we kept plain entropy." Code:
  `ppo.py` **unconditionally** replaces the entropy bonus with reverse KL
  toward `expander_magnet` (a fixed heuristic favoring expansion/capture;
  note KL(π‖m) = −H(π) − Σπ·log m, so it *contains* the entropy bonus plus a
  pull toward the heuristic). `L.yaml` even sets `magnet_policy: expander` —
  a key `Config` silently ignores. **Decided 2026-08-11: plain entropy
  only.** The magnet is not ported at all — no flag, no `magnet.py`.
- **D3 — filter ranking.** Paper prose reads signed ("top 25% … ranked by
  the critic's predicted advantage"); code is `top_k(jnp.abs(advs))` — keeps
  large *negative* advantages too, discarding only the low-information
  middle. **Decided 2026-08-11: |advantage|**, per the code — it is what
  actually trained the ladder agent, and dropping all negative feedback
  would be a real behavioral change.
- **D4 — curriculum mechanics.** Paper describes a gradual cap increase;
  code uses discrete win-rate-gated stages (threshold 0.6, evaluated vs
  random every `eval_every`). Adopt the code's version.
- **D5 — vestigial config keys.** `reward_fn`/`composite_reward` in the
  YAML curricula, `opponent`, `population_*`, `ckpt_pool_size`,
  `magnet_policy` are all **ignored** by `config.py` (unknown-key warning) or
  hardcoded over in `ppo.py`. The reward is always sparse win/loss. Do not
  port these keys.
- **D6 — hyperparameter drift across tiers.** `adv_top_frac` 0.5 (S) vs 0.25
  (M/L, paper); `gae_lambda` 0.9 (S/L, paper) vs 0.7 (M); LR exponent 1.1
  (S, paper) vs 1.17/1.2 (L/M). Start every tier from the paper/L values;
  treat the others as tuning noise.

## 4. Competition-rule deltas that actually change code

From `RULES.md` vs the paper's classic rules:

1. **Castles are built, not captured** — the big one. No neutral castles;
   a third action kind `[2, r, c, _, _]` builds on an owned plain cell,
   price 35 + crowding surcharge, paid from the cell's army
   (`generals/modifiers/build_castles.py`). Consequences:
   - **Policy head: H×W×10** (pass, 4×all, 4×half, build) instead of ×9.
     `decode_action`/`encode_action` extended; channel 9 maps to pass-field 2.
   - **Build validity mask**: `compute_valid_move_mask` covers only the 4
     directions; we add a build mask from `build_cost_grid` (owned ∧ plain ∧
     army > cost). Same −1e9 masking scheme as moves.
   - **Observation**: the 38-channel layout carries over nearly 1:1 (our
     `Observation` has identical fields; `castles` ↔ the paper's cities
     channel). **Decided 2026-08-11:** add one channel — own build cost per
     cell from `build_cost_grid` (cheap to compute, tells the net the
     crowding surcharge). 39 channels total.
2. **Deathtouch from turn 800** — env-internal rule; the `timestep` channel
   is already in the obs, the net learns the regime change. No pipeline code.
3. **Draw at 1200 turns** — env `truncation=1200` (vs L.yaml's 2048). Draws
   are reward 0; with γ=1.0 GAE + the truncation-aware carry this is already
   handled. Early training will be draw-heavy at full spawn distance — which
   is exactly what the spawn-distance curriculum fixes.
4. **Board 18–21 per side, pad_to 21, mountains 24–26%, min distance 17** —
   just env constructor numbers. `pad_to=21` with patch 3 gives a clean 7×7
   = 49 patch tokens (+2 temporal +1 value = 52 tokens).
5. **Move-order tiebreak (chase > reinforce > smaller), simultaneous capture
   = draw** — pinned inside the submodule; nothing for us to write, but the
   engine SHA must go into every run manifest (it already flips results
   across the 2026-07-27 change).

**Curriculum vs the mode preset:** `mode="competition"` pins
`min_generals_distance=17`, and mode is authoritative over kwargs. Curriculum
stages therefore construct the env from **explicit kwargs equal to the preset
except the distance window** (start ~2–6, end at the preset's 17+), and the
final stage must be bit-identical to the preset's parameters. A unit test
should assert that equality so the preset can't drift away from training
silently.

## 5. The deployment constraint (do not defer this)

Match limits: 150 ms/move, one CPU core, 2 GB, no GPU, no network. Rough
forward-pass math at 52 tokens (~2 × torso-params × tokens):

| Net | ≈ FLOPs/move | est. 1-core latency |
| --- | --- | --- |
| S ~5M | ~0.5 G | ~15–40 ms — safe |
| M ~8M | ~0.8 G | ~30–80 ms — OK |
| L 15.35M | ~1.6 G | ~60–150 ms — **at the edge** |

So the deployable tier is decided by a measured CPU benchmark, not by
training results — and we run that benchmark in Phase 2 on randomly
initialized nets, before spending GPU money on a size we can't ship.
**Decided 2026-08-11:** inference is Python numpy/jax-CPU first — simplest,
good enough for the arena and the local gate. Before an actual competition
submission the net gets converted to the morpheus-rs candle stack (which
already has submission-smoke tooling in `scripts/morpheus_rs_*`); that
conversion is a deployment step, not a training-pipeline concern.

Deployment always uses the **EMA** checkpoint (paper: EMA beats last iterate
by ~30 Elo consistently).

## 6. Disposition of the current Morpheus code

Nothing is silently kept. Every current piece, explicitly:

| Current piece | Fate | When |
| --- | --- | --- |
| `self_play/league.py` + `sampler.py` (league, roles, mixtures) | **Delete** — paper's core claim is that the outer loop is unnecessary | Phase 5 cleanup |
| `self_play/driver.py`, `seats.py`, `verify.py`, `schema.py`, `explore.py`, `ingest.py` (CPU shard producers, search-based self-play) | **Delete** — replaced by in-graph GPU rollouts; shards cease to exist as a concept | Phase 5 |
| `trainer/` (torch loop, replay buffer, step, calibrate, manifest, metrics) | **Delete** — replaced by the JAX loop; the manifest/checkpoint *ideas* (immutable manifest, resumable state, engine SHA pinning) are re-implemented in the new loop | Phase 5 |
| `objective/` (reward shaping, aux losses, targets) | **Delete** — sparse win/loss only; paper's ablation shows shaping destabilizes at high throughput | Phase 5 |
| `scraped_rebuild/`, `curriculum/` (scraped-replay reconstruction and curriculum), `corpus/` | **Delete from training** — training is self-play from scratch; the scrapers and `competition-replays/` stay as observational analysis per AGENTS.md, they just never feed training | Phase 5 |
| `compute/`, `jax_preflight/`, `export_preflight/`, `measure_*.py` | **Delete** — tied to the torch net and the old cadence | Phase 5 |
| `network.py` → `bots/morpheus` MorpheusNet (torch CNN) | **Keep** — `bots/morpheus` stays in the roster as a reference opponent even after the new bot passes it (decided 2026-08-11) | permanent |
| `scripts/morpheus_modal.py`, `_modal_self_play.py`, `_modal_materialize.py`, `_materialize.py`, `_self_play.py`, `_sp_ingest.py`, `_modal_sp_ingest.py`, `_sp_prepare_league.py`, `_corpus.py`, `_curriculum.py`, `_rebuild_scraped.py`, `_objective.py`, `_shaping_variant.py`, `_modal_preflight.py`, `_modal_export_preflight.py`, `_export_preflight.py`, `_cadence_evidence.py`, probes | **Delete** | Phase 5 |
| `scripts/morpheus_rs_*` (Rust/candle inference lineage) | **Keep** — the competition-submission conversion target (§5); untouched until a submission is actually prepared | permanent |
| `training/morpheus/tests/*` | **Delete with their subjects**; new pipeline brings its own tests under the 12 s suite ceiling | Phase 5 |
| Uncommitted worktree changes (`sp-from-c13-ckpt10k` ingest/league work, `explore.py`, modified `driver.py`/`loop.py`…) | **Discard** (decided 2026-08-11) — `git restore` + drop the untracked files when implementation starts | Phase 2 start |
| `data/morpheus/` local artifacts, Modal Volume `morpheus-training` | Volume stays primary storage; new runs go under a fresh `/vol/joe/` prefix so old runs remain inspectable; local dir stays gitignored | Phase 2 |

New code lives in a fresh package — **decided 2026-08-11: `training/joe/`**,
mirroring the AverageJoe layout (`networks/`, `train/`, `configs/`), with
Modal entry scripts as `scripts/joe_modal_*.py` and the bot as `bots/joe/`.
Old and new never interleave, so the Phase 5 deletion is a clean
`git rm training/morpheus scripts/morpheus_*` minus the kept
`morpheus_rs_*`.

## 7. Phases

Each phase ends runnable and verified. Costs are rough Modal on-demand
estimates (H100 ≈ $4/h, A100-80G ≈ $2.5/h, ballpark — re-check at Phase 1)
and all Modal work follows AGENTS.md: repo imports guarded by
`modal.is_local()`, output redirected to a file (never piped through
`tail`/`head`), and **startup verified via `modal app logs` a few minutes
in** before walking away.

### Phase 0 — decisions and config freeze (no compute)

- Erik resolves the decision list (§8). *(Resolved 2026-08-11 — all nine.)*
- Write `training/joe/configs/{S,M}.yaml` frozen from the released configs
  + paper values + competition env numbers, with the D1–D6 resolutions
  recorded as comments. No L config — out of budget (§8 item 8).
- **Verify:** config review; this doc updated with the decisions.
- **Cost:** $0.

### Phase 1 — env throughput benchmark on Modal

- **Build:** `scripts/joe_modal_bench.py` — competition-mode env (build
  modifier + deathtouch on, pad 21) under `vmap` + `lax.scan`, random
  actions, measuring: steps/s at 1k/4k/16k envs on 1 GPU (A10G vs H100);
  pool-generation wall time and memory at `pool_size=200k` for distance 17
  and for the early curriculum windows; and net-in-the-loop samples/s with
  randomly initialized S/M/L nets (rollout + PPO update, the real cost).
- **Delete:** nothing yet.
- **Verify (runnable):** benchmark report in
  `docs/research/measurements/joe-phase1-throughput.md` with actual numbers;
  this is the paper's Table I equivalent for our fork and it re-anchors every
  cost estimate below.
- **Cost:** < $10 (a few GPU-hours).

### Phase 2 — port the training loop + CPU latency spike

- **Build:** `training/joe/` port of `networks/common.py`,
  `networks/transformer.py`, `train/{ppo,rollout_selfplay,rewards,evaluations}.py`,
  `config.py`, `main.py`, adapted for: our fork's constructor/API drift, the
  10-action head + build mask + the build-cost channel, competition env
  kwargs, curriculum-final-stage == mode-preset assertion. Checkpoints
  (learner + optimizer + EMA + config + engine SHA) to the Modal Volume with
  `Volume.commit()`. Plus the **CPU latency benchmark**: exported random-init
  S/M/L nets, single core, ms/move — decides the deployable tier (§5).
- **Delete:** nothing yet (old pipeline still the fallback).
- **Verify (runnable):** (a) unit tests: GAE truncation handling, HL-Gauss
  targets, 10-action encode/decode round-trip, build mask vs
  `build_cost_grid`, curriculum/preset equality — inside the 12 s suite
  budget; (b) a ~30-min Modal GPU smoke run at spawn distance 2–6 showing
  loss movement and win-rate vs random climbing above 50%; startup checked
  via `modal app logs` per AGENTS.md.
- **Cost:** ~$10–20.

### Phase 3 — S-config run that beats a heuristic bot

- **Build:** full S-tier run on 1 GPU (S net, pad 21, full curriculum to
  distance 17+, paper schedules). Resumability proven by killing and
  resuming the run once.
- **Delete:** nothing yet.
- **Verify (runnable):** (a) curriculum reaches its final stage; (b)
  in-training eval ≥ ~90% vs random at full distance; (c) **the competition
  gate**: export the EMA net into a minimal stdio bot and finish
  `competition-module/competition/matchup.py <aj_bot> <heuristic_bot>
  --mode competition --seed 0` (with `PYTHON=.venv/bin/python`), plus a small
  A/B via `arena/matches/run_match.py` into `data/games/` showing it beats at
  least one mid-tier heuristic bot per `docs/arena/decision-rule.md`.
- **Cost:** 1×H100 ≈ 12–24 h ≈ **$50–100** (Phase 1 numbers refine this).

### Phase 4 — deployment path and arena entry

- **Build:** the real bot under `bots/joe/`: EMA export, obs/temporal-state
  reconstruction from the stdio protocol, numpy/jax-CPU inference (§5),
  `run.sh`, thread pinning as `bots/morpheus` does. Measure p99 ms/move over
  full games on one core; budget is 150 ms with the 10 s first-move grace
  absorbing warm-up/JIT. The morpheus-rs conversion happens only if/when a
  real competition submission is prepared — not in this phase.
- **Keep:** `bots/morpheus` (reference opponent, permanent) and
  `scripts/morpheus_rs_*` (submission conversion target).
- **Verify (runnable):** competition gate vs the current best bot; game
  stored under `data/games/`; ratings refit (full-pool, as always); pairwise
  contrast quoted per the decision rule.
- **Cost:** CPU only, ≈ $0 beyond dev time.

### Phase 5 — delete the old pipeline

- **Build:** nothing.
- **Delete:** everything marked "Phase 5" in §6, one commit, ASD-STE100
  message. Docs under `docs/bots/morpheus/` gain a pointer to the new
  pipeline; `AGENTS.md` file-placement table updated
  (`training/joe/`, `/vol/joe/`).
- **Verify (runnable):** full test suite green under the 12 s budget;
  `matchup.py --mode competition` gate re-run on the new bot from a clean
  checkout.
- **Cost:** $0.

### Phase 6 — M-tier scale-up

- **Build:** M-tier run (~8M net) on 1–2 GPUs with the same frozen recipe.
  **M is the terminal tier** — the L tier is out of budget (decided
  2026-08-11) and stays in this doc only as the paper-scale reference point.
- **Verify (runnable):** same gates as Phase 3/4; the M-tier EMA bot enters
  the arena as a new immutable bot version; before/after pairwise contrast
  with an interval decides promotion.
- **Cost:** 1×H100 × 3–5 days ≈ **$300–500** (or 2×H100 halving
  wall-clock). For scale: the paper's 4×H200×4d is ≈ $1900–2400 at Modal
  rates — deliberately not bought.

Running total to a deployed, arena-rated M-tier agent: **≈ $400–650**.

## 8. Decisions — all resolved 2026-08-11

1. **D1 depth:** paper's 7.
2. **D2 regularizer:** plain entropy; magnet not ported.
3. **D3 filter ranking:** |advantage|, per the code.
4. **Build-cost observation channel:** yes — 39 channels total.
5. **Package/bot naming:** `training/joe/`, `scripts/joe_modal_*.py`,
   `bots/joe/`, `/vol/joe/`.
6. **Deployment inference stack:** Python numpy/jax-CPU; converted to
   morpheus-rs (candle) only for an actual competition submission.
7. **Uncommitted worktree:** discard the in-flight `sp-from-c13-ckpt10k`
   self-play changes when implementation starts.
8. **Budget:** no L-size run — M is the terminal tier.
9. **`bots/morpheus`:** kept permanently as a reference opponent.

Implementation is unblocked; Phase 0's remaining work is writing the frozen
`training/joe/configs/{S,M}.yaml`.
