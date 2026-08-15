# Joe-net port plan

Status: **written 2026-08-16. N0 passed on its middle row; N1 is done.** The
`bots/morpheus-joe/` fork exists, carries joe's weights and joe's forward pass,
and plays finishing games through a network-free evaluator. N2 has not started.
What N1 changed about this document is marked **N1** where it appears, and the
summary is at the end of the N1 phase entry below.

N0's results are in [joe-net-n0.md](../../research/measurements/joe-net-n0.md)
and they revise this document in six places. Each is marked **N0** where it
appears below; the summary is §1.9. The short version: the forward is 98% of
joe's move, `augment_obs` is fifty times cheaper than §5 assumed, `reserve_ms`
is inert so the turn is 140 ms rather than 130, and the completed-simulation
count is 5 on an ordinary turn and 3 on a belief-recovery turn — K0's `4 – 7`
band, not its stop row.

Goal: run **joe's frozen network** — forward pass and weights, unretrained —
under **morpheus's search stack** (MCTS, particle belief filter, tactics
layer, runtime controller). The network being replaced is morpheus's own
249,316-parameter CNN.

This plan does not re-argue the approach. Retraining a morpheus-shaped net,
retraining joe's net with belief planes, and reducing morpheus to joe's greedy
loop are all out of scope. The decision is made; what follows is the port and
the measurements that decide whether it ships.

Two rules carried in from the sibling ports, as rules rather than suggestions:

- **Replay proves agreement, mutation proves the proof**
  ([rewrite-plan.md](rewrite-plan.md) M1).
- **Loose float tolerances cannot see precision bugs**
  ([rewrite-plan.md](rewrite-plan.md) M2).

And one from this repo's decision rule: a contrast is only evidence **inside
one round**, and only after it **replicates** — the M6 `+425 Elo` retraction is
the reason ([decision-rule.md](../../arena/decision-rule.md)).

---

## 0. Where the port lives, and why that is not a detail

**The port is a new sibling bot directory, `bots/morpheus-joe/`, forked from
`bots/morpheus-rs/`. `bots/morpheus-rs/` is never edited.**

This is forced by the measurement, not chosen for tidiness. §9 requires current
morpheus-rs, the ported bot, and joe-rs to play **in the same rating round**.
A round needs three playable `run.sh` paths. If the port edited
`bots/morpheus-rs/` in place, the baseline arm would exist only in git history
and the contrast would have to span rounds — which
[decision-rule.md](../../arena/decision-rule.md) measured at +46 Elo *between
byte-identical programs*. The repo has used this fork pattern three times
already (joe → joe-rs, joe-rs → unclejoe, macaria → `macaria_base`), and it
doubles as the rollback point (§10).

The cost is honest duplication: joe's `nn/{net,gemm,safetensors}.rs`,
`board/obs.rs`, and `xla_math.rs` exist in `bots/joe-rs/`, in `bots/unclejoe/`,
and now in a third place. §8 turns that from a liability into the parity
mechanism.

**The bot id is `morpheus-joe`** (decided 2026-08-16). It breaks the repo's
mythological convention (`macaria`, `proteus`, `sosipolis`) on purpose: the
lineage is the whole point of the name, and §9's three-arm round is unreadable
if the reader has to remember which myth is the fork. Used literally
throughout.

---

## 1. Verified baseline

Checked against the working tree on 2026-08-16. Everything in this section was
read out of the source, not inherited from the brief. **Three items in the
brief did not survive the check** and are marked.

### 1.1 Joe's network — confirmed

`bots/joe-rs/src/nn/net.rs:36-49`: `EMBED 384`, `DEPTH 5`, `N_HEAD 8`,
`HEAD_DIM 48`, `PATCH 3`, `GRID_PATCHES 7`, `N_PATCHES 49`,
`PATCH_DIM = 39·3·3 = 351`, `N_TOKENS = 52`, `FF_DIM 1152`, `NUM_BINS 128`,
`POLICY_OUT = 10·9 = 90`, `N_LOGITS = 10·441 = 4410`. 8,556,250 parameters in
100 leaves; `bots/joe-rs/artifact/model.safetensors` is 34,235,272 bytes.

The forward signature is `Net::forward(aug_norm[39·441], penalties[10·441],
temporal[2·512]) -> ForwardOut` (`net.rs:331`). **Penalties are added
post-hoc**, after unpatchify, to the flat logits — the trunk never sees them.
That fact is load-bearing for §6.

`AugState` (`board/obs.rs:165`) carries `army_stack` and `enemy_stack` at
`7·441` each, seven single planes at `441`, and two `512`-wide ring buffers,
plus `temporal_step` — **11,197 f32 ≈ 44.8 KB per state**.
`bots/joe-rs/src/main.rs:164-166` advances it exactly once per real turn
(`mem::swap`, then copy both ring buffers into the temporal input).

**FLOPs, which is the number that matters and is not 34×.** Counting MACs:
embed 6.6 M, temporal encoder 0.92 M, five blocks at 78.8 M (QKVO 30.7 M,
scores 2.1 M, FF 46.0 M) = 394 M, heads 1.75 M. **≈ 0.40 G MAC ≈ 0.81 GFLOPs**,
matching [port-plan.md](../joe-rs/port-plan.md) §1. Morpheus's CNN is
~0.2 GFLOPs ([rewrite-plan.md](rewrite-plan.md) §1). **The compute ratio is
~4×, not 34×** — a transformer over 52 tokens is parameter-dense and
activation-cheap. The port's budget arithmetic in §5 uses 4×, and §4 measures
it rather than assuming it.

### 1.2 Morpheus's network and action space — confirmed

`crates/core/src/nn/network.rs:38-63`: 49 input channels, trunk 64,
`POLICY_CHANNELS 9`, `N_ACTIONS = 9·441 + 1 = 3970`, `PASS_INDEX 3969`,
11 heads. `artifact/model.safetensors` is 1,009,015 bytes.

`crates/core/src/board/action.rs:25-30`: `CH_BUILD = 8`; channel =
`direction + 4·split`. `board/transition.rs:24` and joe's `board/obs.rs:36`
carry the **identical** `DIRECTIONS = [(-1,0),(1,0),(0,-1),(0,1)]`, so the
direction remap in §6 is an identity, not a permutation.

### 1.3 Correction: belief does **not** read the four auxiliary heads

The brief states the belief filter "reads `hidden_owner`, `enemy_general`,
`hidden_castle`, and `enemy_army_bins` back out". It does not, and nothing
else does either.

`Heads::All` — the only entry point that computes those four heads
(`nn/network.rs:332-343`) — is reached from exactly two places in the crate:

| site | purpose |
| --- | --- |
| `nn/inference.rs:186` | startup warmup loop over all three head sets |
| `parity/surfaces/net.rs:43` | the `net` parity surface |

No file under `belief/`, `search/`, `tactics/`, `runtime/`, or `crates/bot/`
names any of the four. The grep is exhaustive over `crates/core/src`.

What belief *does* take from the network is the **policy head**, through
`ProposalPolicy::policy_logits` (`belief/proposal.rs:57`) — and
`deployment.json` ships `use_policy_proposal: false`, on a measurement quoted
in that file's own header: `-0.07 ± 0.13` paired against macaria over 100
games per arm, with ~13 ms/turn returned to the search. **The deployed belief
filter already runs on uniform legal enemy actions and touches no network.**

Consequence for the port: the "belief loses both directions" problem is
one-and-a-half directions, and the half that is lost is already switched off.
§3 resolves it.

### 1.4 Correction: `min_simulations` is inert

`deployment.json` carries `min_simulations: 8`. It is parsed
(`runtime/deployment.rs:136`) into `RuntimeConfig` (`runtime/config.rs:83`) and
**never read** — no site in `runtime/` or `search/` consults it. There is
therefore no simulation floor that would force a deadline overrun when the
count collapses. Good news for §5; listed as Q9 because a knob that lies is a
knob someone will eventually trust.

### 1.5 Correction: the belief planes are eight, not seven

The brief names planes 22–28. `nn/tensor.rs:54-60` confirms those seven
(`P_BELIEF_ENEMY_OWNER` … `P_BELIEF_OWNER_ENTROPY`), but `nn/tensor.rs:80`
adds `P_BELIEF_ESS = 48` — an eighth belief-derived plane, the effective
sample size broadcast as a constant. It dies with the other seven.

### 1.6 The two `Observation` structs are field-identical

`bots/morpheus-rs/.../io/wire.rs:52` and `bots/joe-rs/src/io/wire.rs:41` carry
the same ten fields in the same order; the grids differ only in element type
(`u8`/`u8`/`i32` against `i32`/`i32`/`i32`). Joe's `frame_to_raw` can be fed
from morpheus's `Observation` with a widening loop and nothing else.

And `board/observe.rs:56-67`: `emit_observation` fills `opp_land` and
`opp_army` from the true state totals, exactly as the wire scalars do. **The
512-step temporal window is therefore exactly computable at any hypothetical
node in the search tree** — which decides half of §5.

### 1.7 Measured latency, both sides

| source | figure |
| --- | --- |
| [m6-latency](../../research/measurements/morpheus-rs-m6-latency.md) | morpheus-rs p50 **16** simulations/move; `root_inference` 4.83 ms p99/turn; `leaf_batch` **18.88 ms p99 per call** at batch 4 → ~4.72 ms per leaf forward; `enemy_prior_batch` 12.77 ms p99/call; `particle_transitions` 26.87 ms p99; degraded to `policy` on **4,027** of 10,994 frames |
| [joe-rs/latency.md](../joe-rs/latency.md) | full per-move path (parse → obs → forward → decode) **21.4 ms p50 / 23.7 ms p99** on Modal x86 one core; 22.2 / 24.0 on dev arm64 |

The two numbers are not directly comparable: joe's is a whole move, morpheus's
is one forward. **Splitting joe's 21.4 ms into obs and forward is the single
most important unknown in this plan**, because it decides both §5 (how many
simulations survive) and §4 (whether history can be advanced per node). N0
measures it first.

**N0 measured it** ([joe-net-n0.md](../../research/measurements/joe-net-n0.md)
§1). On one x86 core the forward is **20.81 ms p50 / 23.01 ms p99** and it is
**98.2% of the move**. The whole observation pipeline — parse, raw, cost,
`augment_obs`, normalize — is 0.067 ms. `augment_obs` alone is **0.042 ms**.
The per-turn `eprintln!` of the free eval costs 0.274 ms, four times what
`augment_obs` costs and more than every other non-forward stage together.

### 1.8 The value head's semantics, read out of `training/joe/`

This was Q4, and it is now answered from source rather than assumed.

**Perspective: the seat whose observation is on the input.**
`train/rollout_selfplay.py:29-34` builds both seats' observations with
`get_observation(state, seat)`; `:80-84` computes
`winners_p1 = jnp.where(winners >= 0, 1 - winners, winners)` and feeds each
seat its own flipped winner. `train/rewards.py` states it in the docstring:
*"winners is already from the acting player's perspective; the rollout flips it
for the p1 seat"*. So the value is always the observing seat's.

**That is exactly the convention `backup_value(from_root)` assumes.** Morpheus
feeds an enemy node the enemy's observation via
`emit_observation(state, enemy_seat)`, so joe's value there is already the
enemy's, and the `from_root` negation converts it to root perspective. **No
sign change is needed.** R5 drops from "highest-consequence unknown" to a
cheap assertion at N3.

**Scale: `gamma = 1.0` with a sparse terminal ±1 reward.**
`config.py:72` and `configs/M.yaml:70`; `train/rewards.py::win_lose_reward`
returns +1 / −1 / **0 for `winner < 0`**, and the deployed checkpoint is a
config-M run (`manifest.json` → `joe-M-vast-20260813-0213`, step 23500,
curriculum stage 4). Undiscounted, sparse, draw-at-zero: the scalar is an
**expected game outcome in [−1, 1]** — the same quantity morpheus's
`p_win − p_loss` was. **It drops into `backup_value` with no rescaling.**

**Decode: matches joe-rs's.** `networks/transformer.py:289` is
`jnp.sum(jax.nn.softmax(value_raw) * bin_centers)` with
`bin_centers = jnp.linspace(v_min, v_max, num_bins)` = `linspace(-1, 1, 128)`,
and `:287` casts to f32 before the softmax. §7.3's f32 rule and the
export-don't-recompute rule for `bin_centers` are both confirmed correct.
HL-Gauss smearing uses `hl_sigma = 0.04` against a bin spacing of 2/127 ≈
0.0157 (`train/ppo.py:93-106`) — a ~5-bin smear, which is why the argmax bin is
not the value and the dot product is mandatory.

**Three caveats that make §7.4 concrete rather than speculative** (all three
survive N0 untouched — N0 priced the budget, not the value):

1. **It is a PPO critic, not a search value.** `train/ppo.py:40-61` computes
   GAE with `gae_lambda = 0.9`, so the target is a bootstrapped λ-return —
   V^π for joe's own π, not V of the state under any search policy.
2. **The opponent distribution is one opponent: joe.** `rollout_selfplay.py`
   is symmetric self-play with a single network on both seats, and
   `config.py:7` records that checkpoint-pool self-play (`ckpt_pool_size`,
   `magnet_policy`) was **not ported**. `pool_size: 200000` is a *map* pool
   (`env.py:20-37`), not an opponent pool. So the value is calibrated against
   a contemporaneous copy of itself and nothing else.
3. **Truncated steps are masked out of the loss** (`train/ppo.py:187-189`,
   *"their delta/advantage is wrong"*). Competition games truncate at turn
   1200. Concrete prediction: **value calibration should be measurably worse
   near truncation**, and §7.4's calibration probe should bucket by turn to
   see it.

### 1.9 What N0 changed, in one place

Measured 2026-08-16;
[joe-net-n0.md](../../research/measurements/joe-net-n0.md) carries the numbers
and the method. Six revisions, listed here so no section below has to be read
against a stale premise.

| # | this plan said | N0 measured | where it lands |
| --- | --- | --- | --- |
| 1 | `augment_obs` ≈ 2 ms, so F ≈ 19 ms | `augment_obs` 0.042 ms, **F 20.81 p50 / 23.01 p99** | §5's budget is tighter, §6.3's fork is decided |
| 2 | 130 ms usable (`normal_deadline_ms − reserve_ms`) | **140 ms** — `reserve_ms` is inert in both bots | §5.1's table is 10 ms low |
| 3 | freezing the 14 history planes might be forced | 0.042 ms is far inside the "< 0.5 ms" row | **advance per node**; R3 and R8 retire |
| 4 | dropping joe's mask build is a saving | 0.007 ms | §6.2 keeps its rationale, loses its saving |
| 5 | ~4 leaf evaluations, one batch of 4 | batch 4 completes **zero**; batch 1 completes **5** ordinary / **3** on recovery | §5.3's batch change is required, not optional |
| 6 | enemy priors are one of four forward consumers | they are **28%** of all forwards | §3.4 frees more than implied |

Two knobs are now known to be parsed and never read: `min_simulations` (§1.4)
and `reserve_ms`. Q9 covers both.

---

## 2. What the port keeps, replaces, and deletes

| subsystem | fate |
| --- | --- |
| `search/{tree,select,backup,matrix,controller}.rs` | **kept**, unchanged except the action-space width |
| `belief/*` | **kept**, demoted to rules-only (§3) |
| `tactics/*` | **kept**, unchanged |
| `board/{state,transition,observe,memory,hashing,symmetry}.rs` | **kept**, unchanged |
| `runtime/*` | **kept**, re-qualified (§5, §7) |
| `nn/{network,gemm,safetensors,inference}.rs` | **replaced** by joe's |
| `nn/tensor.rs` (49 planes, `build_tensor`, `BeliefSummary`) | **deleted** (§3, §7) |
| `belief/summary.rs` (`summarize_belief`) | **deleted** — `build_tensor` was its only consumer |
| `artifact/model.safetensors` (1 MB) | **replaced** by joe's (34 MB) |
| `board/action.rs` codec | **kept**; a remap layer is added (§6) |
| parity surfaces `tensor`, `net`, `prior` | **retired** with the Torch oracle (§8) |
| parity surfaces `decide` | **loses its oracle** (§8) — a named, accepted cost |
| everything else in `parity/surfaces/` | **kept**, Python-morpheus oracle unchanged |

Roughly 700 lines are deleted and roughly 1,400 arrive from joe-rs. Because the
port lives in a fork (§0), none of these deletions touch the shipped
morpheus-rs.

---

## 3. Belief loses the network, and mostly does not care

**Decision: the belief filter survives as a rules-only filter, frozen, with its
network coupling removed in both directions.** It is not deleted, and it is not
reduced to a tactics input — the search cannot run without it.

### 3.1 Why it cannot be deleted

The particle filter is not an auxiliary signal; it **is** the search's model of
the hidden state. `search/tree.rs:439-473` enumerates particles to build enemy
information sets, hash them, and weight the enemy tables. `EvalItem` carries a
`BeliefState` by value. `belief/recovery.rs` is what keeps a legal belief alive
after a mass refutation. Deleting the filter deletes the search.

Its other live consumers, all network-free:

- `tactics/query.rs:229` `believed_enemy_general` → `tactics/shaping.rs:96`,
  `tactics/seek.rs:45` and `:100`.
- `tactics/constrain.rs` — takes the belief as an input since morpheus's final
  pre-rewrite commit.
- `runtime/controller.rs` — admission, degrade level, ESS telemetry.

### 3.2 The write direction, deleted

`summarize_belief` → `BeliefSummary` → `build_tensor` planes 22–28 and 48. Joe's
patch embedding is `PATCH_DIM = 39·3·3 = 351` and its first linear is
`(384, 351)`; appending belief planes changes `PATCH_DIM` and requires
retraining, which is out of scope by construction. So the belief's eight planes
have no reader, `BeliefSummary` has no consumer, and `summarize_belief` has no
caller. All three are deleted rather than left computing a value nobody reads —
the `belief_tensor` runtime component goes with them.

**What the search loses:** the network's leaf evaluations no longer condition on
the belief at all. Two leaves that differ only in the belief now receive
identical priors and identical values. The belief still differentiates them
*outside* the network — through the enemy tables, through the transition kernel
that advances particles, and through the tactics layer — but the learned
component is now blind to hidden-state uncertainty. That is the largest single
strength cost of this port and it is not recoverable without retraining. §12
lists it as R1 and §9 is designed to price it.

### 3.3 The read direction, already off

The proposal step uses **uniform legal enemy actions** and has since the
deployed configuration was fixed (`use_policy_proposal: false`). Nothing
changes: `propose_enemy_actions` keeps sampling from `softmax_masked` over a
uniform logit vector, and `belief/recovery.rs`'s call to `policy_action_probs`
already handles a `None` policy by falling back to uniform.

Concretely:

- `NetworkEvaluator::as_proposal_policy` is **deleted** (it returns `None` in
  the deployed config anyway, and §4 shows the enemy-side inputs joe's net
  would need do not exist).
- `use_policy_proposal` is **removed from `deployment.json`** rather than left
  as a knob that can no longer be turned on. A knob M7 cannot re-qualify is a
  lie in a config file.
- The four auxiliary heads were never read (§1.3), so nothing at all replaces
  them.

### 3.4 Enemy priors also lose the network

`search/tree.rs` builds enemy information sets from `emit_observation(state,
enemy_seat)` and prices them through `enemy_prior_batch`, which today runs a
policy-only forward per enemy info set (12.77 ms p99/call, 4,027 turns already
degrading). Under joe's net this is impossible for two independent reasons:

1. **Budget.** §5 leaves room for 4–6 forwards per turn total. Enemy priors
   cannot have any of them.
2. **Missing state.** Joe's net needs the *enemy's* `AugState` — the enemy's
   seven-turn delta history, the enemy's accumulated `seen`/`castles`/
   `mountains` latches, and the enemy's 512-step window over *our* totals. The
   belief tracks a hypothesized `enemy_memory` per particle, but not a
   hypothesized enemy `AugState`, and `belief/recovery.rs`'s
   `maximum_entropy_reconstruction` has no history to reconstruct one from.
   Building it would mean inventing 44.8 KB of unobservable history per
   particle and resampling it on every belief resample.

**Decision: enemy priors become uniform-over-legal.** `enemy_prior` disappears
as a forward consumer (`runtime/config.rs:26` `FORWARD_CONSUMERS` loses one of
its four entries). This is a strength regression on top of §3.2, and it is
listed as R2.

---

## 4. Two observation pipelines: one authority each

**Decision: joe's `AugState` is authoritative for the network. Morpheus's
`VisibleMemory` stays authoritative for everything else. Neither is deleted,
and the 49-plane `build_tensor` path is deleted outright.**

| pipeline | authoritative for | cost |
| --- | --- | --- |
| joe's `board/obs.rs` + `AugState` | the network's 39-channel input only | one `augment_obs` per network call |
| morpheus's `board/memory.rs` + `VisibleMemory` | legality (`legal_mask`, `play_mask`), build cost, tactics, node identity (`board/hashing.rs`), belief filtering, symmetry | already paid today; unchanged |
| morpheus's `nn/tensor.rs` `build_tensor` | **nothing — deleted** | — |

The port does **not** pay for both pipelines on every node. `VisibleMemory` is
computed once per real turn for the root and folded incrementally along search
paths for legality and hashing — that is what it already does, and it is cheap
(the `memory` parity surface is bit-exact integer work). The 49-plane tensor
build was the expensive half and it goes away.

`bots/joe-rs/src/board/obs.rs` is copied **byte-identical** into the fork, and
fed from morpheus's `Observation` through a thin widening adapter (§1.6). Not
re-derived, not "cleaned up": §8's whole parity argument rests on the file
being byte-equal to the one joe-rs's JAX-oracle corpus already proves.

**What this costs per turn.** The saving is `summarize_belief` +
`build_tensor` (previously inside every `root_inference` and `leaf_batch`
timing). The new cost is `augment_obs` per network call. N0 measures both; the
net change is expected to be small and is dwarfed by the forward itself.

---

## 5. The search budget, and the earliest possible test of it

### 5.1 The arithmetic, stated so N0 can refute it

**N0 refuted it in three places; the original is kept below so the refutation
is legible.** The corrections: the usable turn is **140 ms**, not 130 —
`reserve_ms` is parsed and never read, in this bot and in the Python one. F is
**23.01 ms p99**, not 19, because `augment_obs` is 0.042 ms rather than 2 ms.
And the projected four leaf evaluations are **zero** at the shipped
`pending_leaf_batch: 4`, because `can_admit` needs the whole batch's 92 ms
forecast to fit and it does not. §5.4 has the measured replacement.

Deployed budget as this section assumed it: `normal_deadline_ms 140`,
`reserve_ms 10` → **130 ms** of usable turn.

| line item | p99 ms/turn today | under joe's net |
| --- | ---: | ---: |
| `particle_transitions` + `belief_proposal` | 27.5 | 27.5 (unchanged — network-free) |
| `selection` + `backup` | 5.1 | 5.1 |
| `hashing` + `reply` | ~0 | ~0 |
| **left for inference** | ~97 | ~97 |
| per forward | 4.72 (leaf, batch 4) | **F, unmeasured** |

Joe's full move path is 21.4 ms p50. If `augment_obs` is 2 ms, F ≈ 19 ms, and
97 / 19 = **5.1 forwards**. One goes to the root. `enemy_prior` is removed
(§3.4). So the projection is **~4 leaf evaluations, i.e. one `pending_leaf_batch`
of 4 — four completed simulations at p50**, against 16 today.

That is a 4× loss of search, and it lands below `widen_freeze_below: 16`
(`runtime/controller.rs:507`), which means **progressive widening is frozen on
every turn**: the root only ever considers its initial candidate set. It also
lands at or below `pending_leaf_batch: 4`, so the search is one batch and the
tree is two plies deep at most, whatever `search_depth: 8` says.

**A four-simulation MCTS over a near-argmax prior is joe's move plus a
sanity check.** If that is where N0 lands, the honest reading is that most of
this plan is wasted work — which is why N0 comes first and has a kill
threshold (§11, K0).

### 5.2 Does batching leaves recover anything? Probably ~1.3×, not 4×

Both engines are batch-1 by construction. Morpheus's `evaluate_many` is
explicitly a loop (`search/evaluator.rs:18-24`, on M3's finding that
TorchScript's batched call went superlinear), and joe's `Net::forward` has no
batch axis — its `Scratch` is a single `RefCell`.

At joe's shapes the arithmetic is unpromising but not hopeless. The forward is
compute-bound, not memory-bound: 0.81 GFLOPs in ~19 ms is ~43 GFLOP/s, well
under one AVX2 core's ceiling, while streaming 34 MB of weights once costs
under a millisecond. So batching does not amortize a bandwidth wall. What it
does buy is GEMM shape: the dominant FF GEMMs go from M=52 to M=208, which
gives the microkernel real register blocking to work with. **Expect 1.2–1.5×
throughput, i.e. 4 simulations becoming 5 or 6.** It does not change any
verdict, so it is a phase-N4 optimization behind its own measurement, not a
phase-N1 assumption.

### 5.3 Degrade path and admission controller

`runtime/degrade.rs` needs **no logic change** — its three bands (pass,
highest-prior-legal, search-average) already cover a turn that completes zero
or one simulations, and `select_degraded_action(0, true, ...)` returns the
policy fallback. What changes is which band fires:

| level | today | projected |
| --- | ---: | --- |
| `average` | 6,967 turns | the minority |
| `policy` | 4,027 turns | **the majority** — joe's argmax, unsearched |
| `pass` | ~0 | ~0 (root inference must still complete) |

That is the plan's central risk stated as a table: on most turns the ported bot
would play joe's move, arrived at more expensively than joe-rs arrives at it.

`deployment.json` changes, all of them re-qualified in N4:

| key | today | port |
| --- | --- | --- |
| `offline_p99_ms.leaf_batch` | 35.35 | measured F × `pending_leaf_batch` |
| `offline_p99_ms.root_inference` | 11.86 | measured F |
| `offline_p99_ms.enemy_prior_batch` | 30.09 | **removed** (§3.4) |
| `offline_p99_ms.belief_tensor` | 2.80 | **removed** (§3.2) |
| `target_simulations` | 16 | **8** (N0: 5 is achievable on an ordinary turn; a target the controller cannot reach is not a target) |
| `widen_freeze_below` | 16 | **2** (N0: at 16 against a forecast of 3–5, widening is frozen on every turn) |
| `pending_leaf_batch` | 4 | **1 — required, not optional.** N0 measured **zero** completed simulations at 4, on every turn of 8,166: `can_admit` needs the whole 92 ms batch forecast to fit and it never does |
| `reserve_ms` | 10 | either honored in `deadline_for_turn` or removed; today it is decorative (§1.9) |
| `max_forward_equivalents` | 113 | ~8 |
| `use_policy_proposal` | false | **removed** (§3.3) |
| `network_width` / `trunk_channels` | 64 | **removed** — meaningless for a transformer |
| `resident_memory_target_mb` | 310.3 | re-derived (34 MB weights + per-depth `AugState`s + tree) |
| `prior_temperature` | — | **new** (§7.3) |

The admission controller itself (`can_admit`, the `per_sim` forecast at
`controller.rs:495`) needs no code change: it is a cost model over named
components, and removing a component and re-seeding the rest is a config
operation. The one structural question is whether a 19 ms `leaf_batch`
forecast against a ~97 ms remaining budget admits the batch at all on turns
where the belief ran long — N0 answers that too, because it runs the real
controller.

### 5.4 N0: the cheap test, before any porting

The binary **already contains** the harness this needs:
`RuntimeConfig::fixed_forecasts_ms` and `charge_fixed_forecasts`
(`runtime/controller.rs:42-45, 133, 160-176`) run the whole controller against
injected per-component costs on a charged virtual clock. Nothing about joe's
net has to exist to ask what the controller does when a forward costs 19 ms.

**N0 note on reading a charged clock.** It advances by the *forecast*, not by
what the work costs, so a table of p99s does not model a p99 turn — it models a
run in which every turn is a p99 turn. `particle_transitions` runs 0.059 ms
filtered and 28.9 ms recovered, so that distinction is worth two simulations.
Run the spike at p50 and at p99 and bracket it; one arm asserts, two measure.

N0 is three measurements and one spike, in order:

1. **Split joe's move path.** Add a `--stages` breakdown to joe-rs's existing
   `bench` subcommand and report `parse / obs / forward / decode` percentiles
   on Modal one-core x86, the method
   [joe-rs/latency.md](../joe-rs/latency.md) already uses. Output: **F**, and
   `augment_obs`'s per-call cost.
2. **Count morpheus-rs's forwards.** `forward_by_consumer`
   (`runtime/metrics.rs:22`, four consumers) is already recorded per turn. Read
   it off a 20-game run; no code change. **N0 correction: it is recorded but
   never written out** — `runtime/telemetry.rs` emitted only the
   `forward_equivalents` total, so this step needed a trace-line addition
   after all.
3. **Project.** `97 / F`, minus the root, minus whatever (1) says
   `augment_obs` costs per node if §4's history is advanced per node.
4. **Spike, through the real controller.** Expose `fixed_forecasts_ms` from
   `deployment.json` (a parse addition, no logic), set `leaf_batch` and
   `root_inference` to (1)'s numbers, drop `enemy_prior_batch`, and run 20
   games. Read the completed-simulation histogram and the degrade-level table
   straight out of the existing telemetry.

The spike gives a **simulation count produced by the actual admission
controller**, including the widening freeze and the batch interaction, without
porting a single line of joe's network. Estimated cost: a day.

---

## 6. The net is stateful; the search is not

Joe's input carries three stateful things. They get three different answers.

### 6.1 The 512-step temporal window — **per node, and free**

`opponent_army_history` / `opponent_land_history` are ring buffers of the wire
scalars `opp_army` / `opp_land`. §1.6 verified that `emit_observation` fills
those two fields from the true state totals at any node. Appending one value
and bumping `temporal_step` is O(1) per node. **Advance it per node. There is
no reason to freeze something that is exact and costs nothing.**

### 6.2 The 10×21×21 penalties tensor — **neither; pass zeros**

Joe's penalties are `-1e9` entries added to the flat logits *after*
unpatchify (§1.1). The trunk never sees them. And the port does not want joe's
mask anyway: morpheus masks with `play_mask` (house rules) at leaves and
`legal_mask` elsewhere, which are strictly different sets.

**Decision: pass an all-zeros penalties tensor at play time and never build
joe's mask on the play path.** Take the raw 4,410 logits, remap to 3,970
(§7.1), and apply morpheus's mask through the existing
`legal_normalized_policy`. This removes `compute_valid_move_mask` +
`compute_build_mask_from_raw` + `prepare_action_mask` from every node — a
saving, not a cost.

`prepare_action_mask` stays in the crate for the parity surface only, where it
must still match joe-rs bit-for-bit (§8).

One thing to verify in N2, listed as Q10: joe's build-legality rule and
morpheus's `live_build_cost` must agree cell-for-cell, since after this change
only morpheus's is consulted. Both implement 35 base + `max(0, 14 − 2·manhattan)`
within radius 6; a parity assertion is cheap and the disagreement would be
silent.

### 6.3 The seven board-history planes — **measure, then decide**

This is the one genuine fork in the plan, and N0 decides it.

`augment_obs` produces 14 delta-history planes (`army_stack`, `enemy_stack`)
plus the `seen`/`castles`/`generals`/`mountains` latches and the channel-21
decay counter. Along a search path all of it is *computable* — morpheus's
transition kernel produces full `GameState`s and `emit_observation` produces
the frames joe's pipeline consumes. The question is only whether it is
affordable.

**Cost of per-node advance.** One `augment_obs` per node. State is carried per
*path step*, not per node — the search re-walks from the root each simulation,
so `search_depth: 8` needs 9 buffers at 44.8 KB = **403 KB**, not
`max_tree_nodes: 4096 × 44.8 KB = 180 MB`. That part is fine. The time is not
obviously fine: if N0.1 says `augment_obs` costs 2 ms, then at depth 3 a
simulation pays 6 ms of obs against a 19 ms forward — a 30% tax on a budget
that already only affords four simulations.

**The rule, decided in advance so N0 resolves it rather than reopening it:**

| N0.1 measures `augment_obs` at | decision |
| --- | --- |
| < 0.5 ms | **advance per node**, full history |
| 0.5 – 1.5 ms | advance per node, but cap network evaluation at depth 2 |
| > 1.5 ms | **freeze the 14 history planes at the root**; still advance the temporal window (§6.1) and the scalar/turn channels per node |

**N0 measured 0.042 ms — the top row, by an order of magnitude.** The 14
history planes are advanced per node and nothing is frozen. At depth 8 that is
nine `AugState`s (403 KB) and nine `augment_obs` calls costing 0.38 ms
together, against a forward that costs 23 ms; the history is 1.6% of one leaf
evaluation.

**Everything below in this section is therefore moot** and is kept only because
it documents a risk that was priced and found not to exist. The off-distribution
probe is not needed, R3 has no trigger, and R8 — which required both branches to
close — cannot fire.

**How far off-distribution freezing puts the net.** The frozen stack describes
turns `[t−7, t]` while the board is at `t+d`. Concretely at depth 3–4: the
delta planes read all-zero for the last `d` turns, which is the frame joe sees
when the opponent passed `d` times; the channel-21 decay counter understates
sight age by `d`; and the board's own army totals have advanced by roughly the
growth rate — on a ~40-land board that is ~4 land and ~10 army of change that
no history plane accounts for. Each individual plane stays inside the range joe
saw in training, so the failure mode is **silent miscalibration, not a crash or
an obvious garbage move**. That is the dangerous kind.

**How to find out, cheaply and offline.** Over joe-rs's existing recorded
corpus (`data/joe/joe-rs-parity/games/`), for each frame at turn `t` and each
`d ∈ {1,2,3,4}`: build the prior twice — once from an `AugState` advanced
through turns `t..t+d`, once from the root-frozen `AugState` paired with turn
`t+d`'s board — and report **top-1 agreement rate** and **KL divergence**
between the two priors, plus the same for the scalar value. No games need to
be played; the corpus already holds the frames. If top-1 agreement at d=3 is
above ~95% and KL is small, freezing is defensible and cheap. If it collapses,
the middle row of the table above is the only survivable option, and it caps
the search at depth 2 — which, at four simulations, it effectively is anyway.

---

## 7. Action space, value, and prior semantics

### 7.1 The channel remap

| joe channel | contents | morpheus destination |
| --- | --- | --- |
| 0 | full move, dir 0 (up) | channel 0 — indices `0·441 + cell` |
| 1 | full move, dir 1 (down) | channel 1 |
| 2 | full move, dir 2 (left) | channel 2 |
| 3 | full move, dir 3 (right) | channel 3 |
| 4–7 | half moves, dirs 0–3 | channels 4–7 |
| 8 | **pass, 441 cells** | collapses to the single `PASS_INDEX = 3969` |
| 9 | build | channel 8 (`CH_BUILD`) — indices `8·441 + cell` |

Indices `0 … 3527` (joe channels 0–7) are an **identity map** — the direction
orders are the same array (§1.2). Joe channel 9 moves down one slot to
morpheus channel 8. Only the pass channel needs a decision.

**The pass collapse is a real fork, and the training code argues against the
obvious answer.** Joe's decode maps all 441 pass cells to the same wire reply
`1 0 0 0 0`. Two collapses are defensible and they disagree by up to
`ln 441 ≈ 6.1` nats.

*The training code says `logsumexp`.* `bots/joe/joe_obs.py:387-388` sets the
pass channel's penalty to `0.0` on **all 441 cells** whenever `allow_pass`, and
`networks/transformer.py:314` samples with
`jax.random.categorical(key, logits)` over the flat 4,410. So during training
the probability of the *action* "pass" **was** the sum over 441 cells, and the
policy gradient acted on that sum. `logsumexp` is what the network was
optimized to mean.

*Deployment says `max`.* `bots/joe/agent.py` and joe-rs both play greedy
argmax over the 4,410, so the shipped joe passes only when a single pass cell
beats every move cell. Every rating joe and joe-rs have ever earned was earned
under that rule — including the strength the candidate must beat in §9.

**Decision: `max` by default, and N3 measures whether that was right.** `max`
preserves the one property that ties the prior to joe's measured strength —
*pass wins the remapped prior's argmax if and only if it wins joe's argmax* —
whereas `logsumexp` can hand a 4-simulation search substantial mass on an
action the deployed joe never plays, and passing forfeits tempo. But this is a
preference stated against evidence, not a settled question.

*The experiment that settles it*, cheap and offline on joe-rs's existing
corpus: per frame, report `P_sum(pass)` (softmax mass over all 441 pass cells),
`P_max(pass)` (the single largest), and whether greedy joe actually passed.
If `P_sum(pass)` is routinely large while greedy joe's pass rate is near zero,
`max` is confirmed and the gap is a deploy-time artifact of the head's shape.
If the two track each other, `logsumexp` is free and more faithful. Added to
N3's gate; Q5 records what remains open.

### 7.2 `legal_mask` and `legal_normalized_policy`

Neither function changes. Both are already written against a 3,970-long
`f32` logit vector and a 3,970-long boolean mask
(`nn/network.rs:751`), and the remap produces exactly that. The pipeline
becomes:

```
Net::forward(aug_norm, zeros, temporal) -> 4410 raw logits
  -> remap_joe_logits()  -> 3970 f32          [new, ~30 lines]
  -> legal_normalized_policy(logits, mask)     [unchanged]
```

`legal_mask` and `play_mask` are unchanged and remain the only source of
legality — joe's mask is not consulted at play time (§6.2). One behavioral
note worth writing down: joe's net was trained under joe's mask, and
`play_mask` is strictly tighter (house rules remove actions joe considered).
The net sees no difference — masking is post-hoc — but the renormalized prior
is over a smaller set, which is a benign distribution shift and not a
correctness issue.

`decode_action` / `encode_action` are unchanged; morpheus's codec stays the
wire authority.

### 7.3 The value head: 128 bins → one scalar

Replace `backup_value(logits: [f32; 3], from_root: bool)` with:

```
softmax over the 128 bin logits (f32, mirroring joe's forward)
  -> dot with the exported `bin_centers` leaf  -> scalar in [-1, 1]
  -> negate iff !from_root
```

Three details that are not decoration:

- **`bin_centers` is exported, not recomputed.** It is one of joe's 100
  serialized leaves ([export.md](../joe-rs/export.md)); a `linspace`
  reimplementation is a parity risk with zero upside. joe-rs already made this
  call and the port inherits it.
- **Width is f32, then widened once.** Same rule as the retired `wdl_value`
  (`nn/network.rs:715-733`): joe computes in f32, so the port computes in f32
  and casts at the boundary. Computing in f64 is more accurate and disagrees in
  the eighth decimal, which is enough to reorder a near-tie.
- **Perspective and scale are confirmed, not assumed** (§1.8, was Q4). The
  value is the observing seat's, undiscounted, sparse-±1, draw-at-zero — the
  same quantity `wdl_value` produced. `backup_value`'s `from_root` negation is
  correct as written and needs **no rescaling and no sign change**. joe-rs has
  never *used* the value (it computes it and logs it to stderr), so the
  convention is still unexercised in Rust; N3 keeps a cheap assertion —
  root value vs eventual outcome over recorded games must correlate
  positively — as insurance against a transcription slip, not against a
  semantic unknown.

### 7.4 The part that is not mechanical

**Joe's policy is a PPO policy played greedily, not a search prior.** It was
trained by sampling (`jax.random.categorical`) and *deployed* by argmax, so its
off-argmax mass was shaped by a policy-gradient objective and an entropy bonus,
never by a requirement that the second- and third-best moves be ranked
usefully. Morpheus's PUCT uses the prior in two places where that matters: the
exploration term, and the candidate ordering that progressive widening walks.

*What to expect.* A near-one-hot prior at four simulations means the search
visits the argmax child and, at most, one alternative. Combined with the frozen
widening (§5.1), the ported bot's decision is **joe's argmax unless the tactics
layer's hard rules override it** — `tactics/constrain.rs` commits actions
independently of the prior, and `apply_pre_contact_prior`'s heuristic blend
(λ = 1.0 in the shipped config) still reshapes the root. So the realistic
description of the ported bot is: *joe's move, filtered through morpheus's
tactics, with a two-ply sanity check.* That may still be worth shipping — the
tactics layer is measurable value — but it is not "morpheus with a better net",
and §11's kill criteria are set against that honest description.

*The mitigation that stays inside a frozen net.* A **softmax temperature `T`
applied to joe's logits before masking**. `T > 1` flattens the prior and gives
PUCT something to explore; it retrains nothing and costs one multiply. It
becomes `prior_temperature` in `deployment.json` and is qualified as a knob in
N4 over `T ∈ {1.0, 1.5, 2.0, 3.0}`.

*Joe's value was trained against exactly one opponent: itself.* §1.8 confirms
this from source — symmetric self-play, no checkpoint pool, no magnet policy.
Morpheus's leaves are hypothetical successors reached through belief-sampled
**uniform** enemy actions (§3.3), which is about as far from joe's opponent
model as this game allows. And the value is a bootstrapped PPO critic (V^π for
joe's π), not a search value, so it answers *"what happens if joe plays on from
here"* — not *"what is this position worth"*. Expect miscalibration that grows
with depth, which is the same axis §6.3's probe measures, plus a specific
weakness near turn 1200 where truncated steps were masked out of the critic's
loss.

**What to measure, all of it offline on the existing corpus:**

| quantity | how | what it tells us |
| --- | --- | --- |
| prior top-1 mass, and entropy, per frame | both nets over the corpus, histogram | how sharp the new prior is, in one number |
| `P_sum(pass)` vs `P_max(pass)` vs greedy pass rate | §7.1's experiment | which pass collapse is right |
| "who's deciding" rate | `last_unshaped_prior` (`search/evaluator.rs:92`) already records the pre-shaping prior; count how often the committed action differs from its argmax | whether the search is doing anything at all |
| value calibration, **bucketed by turn** | root value vs eventual outcome over stored games; Brier score and a reliability curve, both nets | whether joe's value survives off its own distribution, and whether §1.8's truncation prediction holds |
| depth-`d` prior agreement | §6.3's probe | the frozen-history cost, separately from everything else |

All five run on recorded frames without playing a game, and all five are cheap
enough to run at N3 before the expensive round in N6.

---

## 8. Parity and packaging

### 8.1 Which oracle survives

Both. They divide cleanly, and neither one is re-derived.

| question | oracle | mechanism |
| --- | --- | --- |
| Is the ported forward pass joe's forward pass? | **JAX/eqx**, transitively | §8.2 |
| Is the ported search morpheus's search? | **Python morpheus**, unchanged | the 28 surviving surfaces in [parity-harness.md](parity-harness.md) |
| Is the ported *decision* right? | **none exists** | §8.4 |

The **Torch oracle dies** with the 249k net. `tests/parity_cases.py` loses its
torch import and its three TorchScript loads — which, per
[parity-harness.md](parity-harness.md), is where ~7 of the smoke slice's
~11.4 s went. The smoke suite gets faster.

### 8.2 The forward pass is proved by byte-identity, not by a second corpus

`bots/joe-rs/` already carries a JAX-oracle corpus that proves its forward pass
matches joe to pinned relative bounds (2.331e-6 logit, 2.244e-6 bin, pinned at
3.0e-6; tier-3 decision parity 731/731 frames; 14 games / 4,760 turns of
byte-equal wire replay — [joe-rs/parity.md](../joe-rs/parity.md)). Rebuilding
that corpus for a third bot buys nothing.

**Instead: a digest test asserts that the fork's copies of
`nn/{net,gemm,safetensors}.rs`, `board/obs.rs`, and `xla_math.rs` are
byte-identical to joe-rs's.** If they are equal and joe-rs's corpus is green,
the fork's forward pass is joe's forward pass. If someone edits one, the test
goes red and names the file. This is exactly the trick
`bots/unclejoe/tools/sync_artifact.py` uses for weights, applied to source.

The test costs four `sha256` calls over ~1,400 lines and belongs in `tests/`,
inside the [15 s budget](../../../AGENTS.md).

Consequence, stated plainly: **the fork may not "improve" those five files.**
Batching (§5.2) would break byte-identity and therefore has to either land in
joe-rs first and propagate, or accept a real second corpus. N4's batching work
is scoped as *upstream into joe-rs*, for that reason.

### 8.3 `parity-smoke.jsonl.gz` and the corpus

`bots/morpheus-rs/tests/fixtures/parity-smoke.jsonl.gz` is **kept as is**. It
feeds the 28 surfaces that do not involve the network — `transition`, `observe`,
`mask`, `cost`, `memory`, `hash`, `symmetry`, `npsum`, `argsort`, `propose`,
`filter`, `rejuvenate`, `maxent`, `reservoir`, `toplegal`, `initbelief`,
`matrix`, `runtime`, `evict`, `playmask`, `candidates`, `planners`, `shaping`,
`constrain`, `search` — none of which change. The frames carry the recorded
RNG stream, which is what those surfaces actually need.

Three surfaces are **retired**: `tensor` (the 49-plane build), `net` (11
heads), `prior` (the WDL-based backup value). Their fixture strata become dead
weight in the file, which is acceptable — regenerating the corpus to remove
them would cost more than the bytes.

The `summary` surface is retired with `summarize_belief` (§3.2).

joe-rs's own corpus under `data/joe/joe-rs-parity/` is **untouched and
unmoved** — the fork reads it only for §6.3's and §7.4's offline probes.

### 8.4 The cost that cannot be paid: tier-3 decision parity

`decide` — "the whole no-search decision, network included" — has no oracle
after this port, because **no Python program plays this combination**. Building
one means porting joe's JAX net into Python morpheus, whose only product would
be an oracle. That is not worth it and the plan does not propose it.

What replaces it, in descending order of what it proves:

1. **A new `prior` surface with a real oracle.** A ~50-line Python driver runs
   JAX joe on a corpus frame, applies §7.1's remap and §7.3's bin→scalar
   conversion in NumPy, and compares against `morpheus-joe parity prior`. This
   covers the genuinely *new* logic — the remap and the value conversion — which
   is where the port's own bugs will live. **This is the most valuable single
   test in the plan** and it is cheap.
2. **Determinism.** Same frame twice → same reply; same seed → same game,
   twice.
3. **Transitive coverage.** §8.2 for the forward, the 28 surviving surfaces for
   the search.

The residue — "did the combination decide correctly" — is answered by §9's
rating round and nothing else. Recorded as an accepted loss, not smoothed over.

### 8.5 Mutation coverage

`bots/morpheus-rs/tools/mutation_check.py` keys its map on paths relative to
`crates/core/src` and lists which surfaces each file's mutations must break
(`mutation_check.py:48-85`). The fork's copy needs:

- **rows removed**: `nn/tensor.rs` (`tensor`, `net`, `prior`),
  `nn/network.rs` (`net`, `prior`), `belief/summary.rs` (`summary`).
- **rows added**: the joe-side files, with the surfaces they can break —
  `nn/net.rs` and `nn/gemm.rs` → (`prior`), `board/obs.rs` → (`prior`).
- **mutations added**: at minimum the pass-channel collapse (`max` → first
  element), the joe→morpheus channel-9→8 remap (off by one channel), the
  bin→scalar dot (`bin_centers` reversed), and the value sign
  (`from_root` negation inverted). Each must make the new `prior` surface fail.

`bots/joe-rs/tools/mutation_check.py`'s 9 planted bugs are **unaffected** —
they pin source snippets in joe-rs's own tree, including two in `nn/net.rs`
(the `q_proj`/`k_proj` swap at lines 66–74 and the `sqrt(HEAD_DIM)` softmax
scale). Because §8.2 forbids editing the fork's copies, those pins stay valid
for both bots. **If a future refactor moves those lines in joe-rs, both bots
break together**, which is the correct coupling.

### 8.6 Every tool that hardcodes a module path

| tool | what it names | action |
| --- | --- | --- |
| `bots/joe-rs/tools/mutation_check.py` | `board/obs.rs` ×6, `nn/net.rs` ×2, `board/action.rs` ×1 (exact source snippets) | unchanged; §8.5 |
| `bots/joe-rs/tools/package_submission.py:16-17,62` | `src/nn/gemm.rs`, `src/nn/net.rs`, `src/io/wire.rs` (prose) | unchanged |
| `bots/joe-rs/tests/test_parity.py:132,378` | `src/nn/gemm.rs`, `src/xla_math.rs` (prose) | unchanged |
| `bots/joe-rs/tests/parity_lib.py:5` | `src/parity.rs` | unchanged |
| `bots/unclejoe/tools/sync_artifact.py` | `bots/joe-rs/artifact/` | extended by §8.7 |
| `bots/morpheus-rs/tools/convert_artifact.py:20` | `crates/core/src/nn/network.rs` | **retired** in the fork — the fork has no converter, it syncs (§8.7) |
| `bots/morpheus-rs/tools/mutation_check.py:48-85` | 30+ paths under `crates/core/src` | edited per §8.5 |
| `bots/morpheus-rs/tools/bench_inference.py` | the old net's shapes | rewritten for joe's |
| `bots/morpheus-rs/tools/package_submission.py` | member list, minifier boundaries | member list updated; the minifier is path-agnostic |
| `bots/morpheus-rs/tools/{dependency_budget,vendor_probe}.py` | `src/main.rs` of a synthetic probe crate | unchanged (they build their own crate) |
| `bots/morpheus-rs/tests/parity_cases.py` | the Torch oracle, the `tensor`/`net`/`prior` cases | edited per §8.3 |
| `bots/morpheus-joe/run.sh` | **must name no other bot's path, in code or in comment** | see below |

**The closure trap, restated because it has already bitten this repo once.**
`fingerprint._SHELL_REF_RE` scans shell sources for bot-relative paths and
cannot tell a comment from a `source` line. An early morpheus-rs `run.sh`
mentioned the Python bot's launcher in prose and pulled `bots/morpheus/run.sh`
into morpheus-rs's content hash ([packaging.md](packaging.md)). The fork's
`run.sh` must refer to joe-rs by description, never by path.

### 8.7 Artifact fan-out: the fifth surface

A joe re-export already silently staleens four things (`joe-rs`'s
safetensors, the JAX parity corpus, the committed smoke fixture, and
unclejoe's byte copy). `bots/morpheus-joe/artifact/` is the fifth.

**It tracks `joe-rs`, one hop down the chain `joe → joe-rs → {unclejoe,
morpheus-joe}` — never joe's `.eqx` directly.** Same reason unclejoe does:
the joe-rs / morpheus-joe contrast in §9 only isolates the search stack if both
run byte-identical weights, not merely weights from the same checkpoint.

Three mechanisms, in order of how much they prevent:

1. **A shared sync tool.** Generalize `bots/unclejoe/tools/sync_artifact.py`
   into one script with a target argument (or a `scripts/joe_artifact_fanout.py
   --check` that walks every downstream copy at once). It already does the
   right things: digest-verifies both sides, writes through a temporary file so
   an interrupted copy cannot leave a half-written `model.safetensors` beside a
   manifest that swears it is complete, and `--check` exits non-zero on
   staleness.
2. **A test, which is the part that actually fires.** A new case in `tests/`
   asserting that every downstream `manifest.json`'s `safetensors_sha256`
   equals joe-rs's. It reads four small JSON files, costs milliseconds, fits
   the 15 s budget, and converts a silent staleness into a red suite. Nothing
   in the current scheme does this — the checklist in
   [joe-rs/export.md](../joe-rs/export.md) is discipline, and discipline is
   what already failed here twice.
3. **The checklist.** Add the fork to "After a joe re-export" in
   [export.md](../joe-rs/export.md), and update the `joe-artifact-fanout`
   memory note.

A sync forks the fork's content hash, exactly as it forks unclejoe's. That is
intended and the version registry records it — **and it invalidates any rating
contrast that spans the sync** (§9).

### 8.8 Packaging: not a problem, with numbers

| bundle | size | files | cap |
| --- | ---: | ---: | --- |
| `morpheus-rs-3f06212b5532.zip` (today) | 1.06 MB | 80 | — |
| `joe-rs-9db7e6f589c0.zip` (today, dependency-free) | 31.85 MB | 21 | — |
| **`morpheus-joe` (projected)** | **~32 MB** | **~85** | 50 MB zip / 10,000 files / 512 MB unpacked |

**~64% of the zip cap and under 1% of the file cap.** The 34.2 MB of fp32 weights
compress to ~31.5 MB in the archive, as joe-rs's bundle demonstrates. Both
crates are dependency-free today, so `vendor/` stays empty and the offline
`build.sh` compiles one crate from source — the same intake path joe-rs proved
at 12.4 s.

`vendor_probe.py` and `dependency_budget.py` stay as guards against a future
`cargo add`; the port adds no dependency, because joe's inference path is
`gemm.rs` plus in-house safetensors and JSON readers. Re-run both at N5 anyway
and quote the counts, per [packaging.md](packaging.md).

Memory: 34 MB weights + 403 KB of per-depth `AugState`s + the tree, against a
2 GB cap. `resident_memory_target_mb` needs re-deriving (currently 310.3) but
there is no risk here.

---

## 9. Phases and gates

Every phase that touches a bot ends with a finished competition matchup, run
exactly this way — the absolute `PYTHON` is load-bearing (a relative path
silently `BrokenPipe`s the Python seat) and `cargo` must be off `PATH` for the
Python arms:

```bash
PYTHON=$PWD/.venv/bin/python .venv/bin/python competition-module/competition/matchup.py \
  bots/<a>/run.sh bots/<b>/run.sh --mode competition --seed 0
```

The match must reach a normal end — win, loss, draw, or truncation. Games are
stored under `data/games/<round>/` before any rating refit.

### N0 — Does the budget exist? *(no port; the kill gate)* — **DONE 2026-08-16**

§5.4 in full: joe-rs `bench --stages` on Modal one-core x86; morpheus-rs's
`forward_by_consumer` over 20 games; the projection; and the
`fixed_forecasts_ms` spike through the real controller.

**Gate:** a measured **F** and `augment_obs` cost, and a completed-simulation
histogram plus degrade-level table from the spike. Matchup gate on `joe-rs`
(touched by `--stages`) and on `morpheus-rs` if the spike needs a
`deployment.json` parse addition — the spike itself runs with the injected
forecasts *off* for the gate match.

**Decision:** §11's K0. If p50 simulations < 4, stop here.

**Result: passed on K0's middle row.**
[joe-net-n0.md](../../research/measurements/joe-net-n0.md). F = 20.81 p50 /
23.01 p99; `augment_obs` = 0.042 ms; completed simulations **5** on an ordinary
turn and **3** on a belief-recovery turn at `pending_leaf_batch: 1`, and
**0** at the shipped batch of 4. Both matchups finished normally. §1.9 lists
what it revised.

One thing N0 did **not** deliver and N1 inherits: `forward_by_consumer` had to
be added to the trace before N0.2 could be read at all, so the plan's "no code
change" for that step was wrong. The additive edits N0 actually made — joe-rs's
`bench --stages`, morpheus-rs's `forward_by_consumer` line and the
`fixed_forecasts_ms` / `charge_fixed_forecasts` parse — are the ones §10's
rollback list names, plus that one.

### N1 — Fork, artifact, forward pass — **DONE 2026-08-16**

Create `bots/morpheus-joe/` from `bots/morpheus-rs/`. Copy joe's five source
files byte-identically. Wire the sync tool (§8.7). Delete `nn/{network,tensor}.rs`,
`belief/summary.rs`, the old artifact, and the three retired parity surfaces.
The bot does **not** play the new net yet: the evaluator still returns the
uniform stub, so the crate compiles and the seat runs.

**Gate:** crate builds; `Net::load` accepts the synced artifact and refuses a
schema mismatch; the §8.2 digest test passes; the §8.7 staleness test passes;
the 28 surviving parity surfaces are green on the smoke slice; matchup finishes
with the stub-evaluator bot (it loses or truncates — both are normal ends).

**Result: every gate met.** The crate builds warning-free. The seat resolves
the artifact, verifies its digest, schema-checks it, loads 8,556,250 parameters
and warms one forward in 131 + 15 ms, then decides through
`ShapedUniformEvaluator`. `tests/test_joe_source_fanout.py` covers both §8.2
and §8.7 and costs 0.07 s; the default suite holds at 13.5 s warm.
`tests/test_morpheus_joe_selfcheck.py` proves the loader refuses a digest
mismatch *and* a `depth: 6` manifest, which is the gate's "refuses a schema
mismatch" in the two shapes that can actually occur. The 28 surfaces are green
in **6.3 s**, against 11.4 s before — the retired Torch oracle was most of it.
The gate match against `cm_expander` truncated at turn 1200, a normal end, with
two castles built. Page: [morpheus-joe](../morpheus-joe/index.md).

**Seven things N1 found that this plan did not say.** None changes a decision;
all seven change what a later phase will find.

1. **Joe's files need a crate, not a directory.** §8.5 keys the mutation map on
   paths under `crates/core/src`, which assumes the copies live there. They
   cannot: byte-identity includes the imports, and `board/obs.rs` opens with
   `use crate::io::wire::{Observation, ...}` where joe's `Observation` carries
   `i32` grids and morpheus's carries `u8`. Dropping the files into
   `morpheus-joe-core` resolves those paths to different types with the same
   names. They live in a **third crate**, `crates/joenet/`, which gives them
   joe's `crate::` root; §8.5's rows move to `crates/joenet/src/` and nothing
   else about §8 changes.
2. **Eleven files, not five.** `board/mod.rs` and `io/mod.rs` name `action` and
   `wire`, so the whole subtree is copied and the whole subtree is pinned. That
   is strictly more coverage than §8.2 asked for.
3. **§3.3 is forced at N1, not chosen at N3.** The policy-proposal path is the
   only other consumer of `build_tensor`, so deleting `nn/tensor.rs` deletes
   it. `ProposalPolicy`, `enemy_info_tensor`, the counting shim and
   `use_policy_proposal` all went with it, and `propose_enemy_actions` lost its
   `policy` and `max_proposal_batch` parameters. The `propose` parity surface
   is unaffected: its integer layout keeps `n_unique_policy_inputs` and
   `n_policy_batches`, both zero on the uniform branch the oracle has always
   taken.
4. **A third dead knob.** `max_proposal_batch` capped policy batches and
   nothing else, so it joined `min_simulations` and `reserve_ms` the moment the
   policy branch went. It is **removed** rather than added to Q9's list.
5. **`decide` is retired, not merely oracle-less.** §2 says the surface "loses
   its oracle"; in practice its Rust half cannot compile without a `Session`
   and a `Heads` switch, so the kind is gone from the dispatch table. §8.4's
   accounting is unchanged — what replaces it is N3's `prior` surface,
   determinism, and the N6 round.
6. **`legal_normalized_policy` needed a home.** §7.2 keeps it unchanged, and
   `network.rs` was its only home. It is now `nn/head.rs`, alone, with N3's
   remap and value decode landing beside it. Copying it forward rather than
   deleting and re-deriving it matters because its oracle retired with the
   Torch one.
7. **Test module names collide across bots.** `tests/` has no `__init__.py`, so
   a second `test_parity_tier1.py` breaks collection outright and a second
   `parity_cases.py` would silently hand one bot the other's harness. Every
   test module in the fork is bot-prefixed, as joe-rs's and unclejoe's already
   were.

Two deliberate non-actions, both scope calls rather than oversights:

- **`deployment.json` is not re-tuned.** K0's four conditions
  (`pending_leaf_batch` 4 → 1, `widen_freeze_below` 16 → 2,
  `target_simulations` 16 → 8, `prior_temperature`) are non-optional *before
  the candidate is measured*, which is N4. Applying them at N1 would tune
  search knobs against a flat prior — a configuration for a bot nobody will
  ship. What N1 did remove is only what became meaningless: the two dead knobs
  above and the two CNN shape knobs (`network_width`, `trunk_channels`). The
  file's own `belief_limitation_note` says all of this in the place someone
  reading the knobs will look.
- **The bot is not registered** in `data/bot_versions/`. Its content hash will
  fork at N2 and again at N3, and a registration written from a dirty tree
  records a closure ref that describes nothing. The first registration belongs
  to the first phase that measures something.

### N2 — Observation bridge

Joe's `AugState` advanced once per real turn from morpheus's `Observation`
(§4). Zero penalties (§6.2). The temporal window from `emit_observation`'s
scalars (§6.1). No search integration yet — the bot plays joe's argmax through
morpheus's wire.

**Gate:** the ported obs pipeline reproduces joe-rs's `sequence` surface
(CRC-32 per turn of the augmented tensor, plus final state) over three full
corpus games; the build-cost agreement assertion of Q10 passes; **the bot's
replies are byte-equal to joe-rs's over at least one full recorded game**;
matchup finishes.

That last check is strong: at N2 the fork *is* joe-rs wearing morpheus's I/O,
so any divergence is a bridge bug and is localized before search complicates it.

### N3 — Search integration

The remap (§7.1), the bin→scalar value (§7.3), the new `prior` parity surface
and its Python/JAX oracle (§8.4.1), enemy priors to uniform (§3.4), belief
demoted (§3), `prior_temperature` plumbed but pinned at 1.0.

Run the five offline probes of §7.4 (which include §7.1's pass-collapse
experiment and §6.3's depth-`d` probe) here, on recorded frames, before
spending a round.

**Gate:** the `prior` surface green against the NumPy/JAX oracle at a pinned
relative bound; the value-sign assertion passes (§1.8 makes this insurance, not
discovery); the pass-collapse experiment reported and §7.1's default either
confirmed or reversed **before** N4 qualifies anything; the mutation additions
of §8.5 all kill; the five probes reported with numbers; matchup finishes with
the real bot.

### N4 — Runtime re-qualification

`deployment.json` rewritten per §5.3 from N0's measured costs. Qualify on one
x86 core the way M7 did: percentile table, count of normal moves over 150 ms,
completed-simulation distribution. Qualify `prior_temperature` over
`{1.0, 1.5, 2.0, 3.0}`. Optionally land §5.2's batching **in joe-rs first**, so
§8.2's byte-identity survives.

**Gate:** a qualification verdict with a named host and CPU, zero normal moves
over 150 ms at p99.9, and the simulation count quoted; matchup finishes.

### N5 — Packaging

`package_submission.py` for the fork; counts quoted against all three caps
(§8.8); offline one-core container build passes; `vendor_probe.py` and
`dependency_budget.py` re-run.

**Gate:** a built bundle at `data/bundles/morpheus-joe-<hash>.zip` with the
artifact and `deployment.json` inside, byte-reproducible across two runs;
matchup finishes from the unpacked bundle.

### N6 — The measurement round

**Three arms in one round.** This is not optional and it is not a matter of
discipline: ratings are fitted one round at a time, and
`fit.delta` raises on an entity that is not in the round.

| arm | what it is |
| --- | --- |
| `morpheus-rs` | current lineage head, unedited (§0) |
| `morpheus-joe` | the candidate |
| `joe-rs` | the thing the candidate has to beat to justify existing |

Panel: ≥ 5 bots spanning the rating range and including the anchor
`cm_expander`. Both morpheus arms and joe-rs share it, which satisfies the
connectivity gate automatically.

```bash
python -m arena.tournaments.competition \
  bots/morpheus-joe/run.sh bots/morpheus-rs/run.sh bots/joe-rs/run.sh \
  bots/cm_expander/run.sh bots/<panel...>/run.sh \
  --round joe-net-r1 --games-per-pair 50 --round-seed 7 \
  --seat-policy alternate --strict-versions
```

Requirements from [decision-rule.md](../../arena/decision-rule.md): ≥ 200 games
per arm, ≥ 30 per (arm, opponent), ≥ 60 decisive per arm, `alternate` seats, a
pinned round seed, one engine version, all games `mode == "competition"`.

**Host state must be noted by hand.** The round manifest records roster, seeds,
seat policy, engine, and job count — and nothing about what else the machine
was doing. A round whose host state is unknown is unpublishable, and this is a
deadline-driven bot, which is the case where that bites hardest.

**Replicate before publishing.** A second, separately scheduled round
(`joe-net-r2`) with the same three arms. The gates in the decision rule all
passed on the round that was wrong by 300 Elo; they cannot substitute for
replication.

**Any artifact sync (§8.7) between r1 and r2 voids both.** Freeze the weights
for the duration.

**Gate:** two rounds, both contrasts reported with intervals, verdicts quoted
per the decision rule.

---

## 10. Rollback

**The rollback point is `bots/morpheus-rs/`, which this plan never edits.**

That is the whole mechanism, and it is why §0 chose a fork. Concretely:

- Tag the repo `joe-net-n0` at the commit that closes N0, before `bots/morpheus-joe/`
  exists. Rolling back is `git rm -r bots/morpheus-joe/` plus removing
  `data/bot_versions/morpheus-joe.json`.
- `data/bot_versions/morpheus-rs.json`'s lineage head is untouched throughout,
  so the shipped bot's rating identity never forks and its submitted bundle
  stays valid.
- The three retired parity surfaces, `nn/tensor.rs`, `nn/network.rs`,
  `belief/summary.rs`, and the 1 MB artifact all still exist in
  `bots/morpheus-rs/` — nothing is recoverable only from git history.
- The only edits outside the fork are **additive**: joe-rs's `bench --stages`,
  the generalized sync tool, the fan-out test, and the `fixed_forecasts_ms`
  parse in morpheus-rs. Each is independently revertible and each is gated by
  its own matchup.

Rolling back after N6 costs the round's compute and nothing else.

---

## 11. Kill criteria

### K0 — after N0, before any porting — **RULED 2026-08-16: middle row**

Read from the spike's completed-simulation histogram.

| p50 simulations | verdict |
| --- | --- |
| ≥ 8 | **proceed** |
| 4 – 7 | proceed **only** with §3.4 (enemy priors dropped), §6.2 (no penalties build), §7.4 (`prior_temperature`) all counted in, and `widen_freeze_below` lowered so widening can run at all |
| < 4 | **stop.** Ship joe-rs. |

Rationale: below four simulations the tree is one batch deep over a near-argmax
prior, and the bot is joe with overhead. The rest of the plan cannot fix that,
because the net is frozen.

**Measured: 5 on an ordinary turn, 3 on a belief-recovery turn, at
`pending_leaf_batch: 1`.** The middle row, and it is the middle row by two
simulations rather than by a comfortable margin. Four conditions attach, and
none of them is optional:

1. **`pending_leaf_batch: 4` → `1`.** Not a tuning preference. At batch 4 the
   controller admits **zero** simulations on every turn of 8,166 and the bot
   plays joe's argmax with a 52 ms overhead. This is the single largest
   configuration finding of N0.
2. **`widen_freeze_below: 16` → 2.** At 16 against a forecast of 3–5, widening
   is frozen on every turn and the root never leaves its initial candidate set.
3. **§3.4 and §6.2 are counted in already** — the spike zeroed
   `enemy_prior_batch` and the mask build is 0.007 ms. Neither can be spent
   twice.
4. **`prior_temperature` (§7.4) is now load-bearing, not a mitigation.** Five
   simulations over a near-one-hot prior visits the argmax and perhaps one
   alternative. R4's tripwire — "who's deciding" below ~5% — is the number to
   watch at N3, and it is likelier to fire than the plan assumed.

The honest description of the candidate is unchanged from §7.4 and is now
measured rather than predicted: **joe's move, filtered through morpheus's
tactics, with a three-to-five-simulation sanity check.** K1 is what decides
whether that is worth 32 MB.

### K1 — after N6, the shipping decision

The candidate must clear **both** contrasts, in the same round, per the
decision rule's `improvement` verdict — `P(B > A) ≥ 0.95` **and**
`CI₉₅.low > +10`:

1. `morpheus-joe` vs `morpheus-rs` → **improvement**
2. `morpheus-joe` vs `joe-rs` → **improvement**

And both must **replicate** in `joe-net-r2`.

| outcome | action |
| --- | --- |
| both improvements, replicated | ship `morpheus-joe` |
| beats morpheus-rs, not joe-rs | **stop. Ship joe-rs.** The search stack is not paying for itself. |
| beats joe-rs, not morpheus-rs | **stop.** The port is a regression on its own lineage; joe's net is not the problem morpheus had. |
| neither | **stop.** Roll back per §10. |
| r1 and r2 disagree by more than their intervals | **unproven.** Do not publish; investigate host state before spending more compute. |

The second row is the one this plan exists to test honestly. A morpheus
running four simulations over joe's net, with the belief filter cut off from
the network in both directions and enemy priors reduced to uniform, has to earn
its 32 MB and its complexity against a 21 ms greedy bot that already works.

### K2 — the latency tripwire, at N4

If more than 1% of normal moves exceed 150 ms on one x86 core after
re-qualification, the configuration is unqualified. Re-tune once
(`normal_deadline_ms`, `pending_leaf_batch`, network evaluation depth); if a
second qualification fails, stop.

---

## 12. Risks

| id | risk | tripwire | fallback |
| --- | --- | --- | --- |
| **R1** | The network is blind to the belief (§3.2), so leaves differing only in hidden-state uncertainty evaluate identically. Unfixable without retraining. | — (structural) | Priced by K1. If it is the reason the port loses, the answer is joe-rs, not a workaround. |
| **R2** | Enemy priors go uniform (§3.4), weakening the enemy tables that the regret-matching in `search/matrix.rs` consumes. | `matrix` surface still green, but strength drops in N6 | None inside a frozen net. Priced by K1. |
| ~~**R3**~~ | ~~Frozen history (§6.3) puts the net off-distribution at depth, silently.~~ **Retired by N0** — `augment_obs` is 0.042 ms, so history is advanced per node and nothing is frozen. | — | — |
| **R4** | The prior is too sharp for PUCT (§7.4); the search never explores. **N0 raised this**: the budget is 3–5 simulations, so the search visits the argmax and perhaps one alternative. | "who's deciding" rate below ~5% | `prior_temperature > 1`, qualified at N4. It is now load-bearing rather than a mitigation. |
| **R5** | ~~Joe's value sign or perspective convention is wrong.~~ **Retired by §1.8** — the value is the observing seat's, undiscounted, sparse-±1, so `backup_value`'s negation is correct as written. Residual risk is a transcription slip only. | N3's sign assertion | Flip and re-gate; cheap. |
| **R9** | The pass collapse is wrong (§7.1). `max` under-weights an action the training distribution rated highly; `logsumexp` over-weights one the deployed joe never plays. | N3's `P_sum` / `P_max` / greedy-pass-rate experiment | Switch collapse and re-run N3's probes; it is a one-line change and must happen before N4 qualifies a config. |
| **R10** | The value is a bootstrapped PPO critic under self-play (§1.8), used as an MCTS leaf value under uniform enemy actions. Miscalibration grows with depth and worsens near turn 1200. | N3's turn-bucketed calibration curve materially worse than morpheus's old WDL head | None inside a frozen net. If the curve is bad late, cap network evaluation depth and lean on the tactics layer past turn 800; priced by K1. |
| **R6** | A joe re-export staleens the fork's weights mid-measurement (§8.7). | the fan-out test goes red | Freeze weights between r1 and r2; treat any sync as voiding both rounds. |
| **R7** | Someone "improves" a copied joe file and breaks the transitive parity argument (§8.2). | the digest test goes red and names the file | Revert, or land the change in joe-rs and re-sync both downstream bots. |
| ~~**R8**~~ | ~~Both branches of §6.3 closed.~~ **Retired by N0** — it needed `augment_obs` > 1.5 ms; it is 0.042 ms. | — | — |

Packaging is a **non-risk** (§8.8: 64% of the binding cap). Memory is a
non-risk. Intake build time is a non-risk (dependency-free, 12.4 s measured on
joe-rs).

---

## 13. Open questions

Listed rather than decided.

- **~~Q1.~~ RESOLVED 2026-08-16 by N0.1.** joe's forward is **20.81 ms p50 /
  23.01 ms p99** on one x86 core — **98.2%** of the 21.2 ms move. Everything
  else in the path, parse through decode, is 0.37 ms and a third of that is one
  `eprintln!`.
  [joe-net-n0.md](../../research/measurements/joe-net-n0.md) §1.
- **~~Q2.~~ RESOLVED 2026-08-16 by N0.1.** `augment_obs` is **0.042 ms p50 /
  0.087 ms p99** — fifty times cheaper than §5 assumed. §6.3's fork takes its
  top row: advance the full history per node.
- **Q3.** Does batching four leaves in `gemm.rs` pay at joe's shapes, and by
  how much? §5.2 guesses 1.2–1.5× from arithmetic intensity; it is a guess.
  *N4, and only if it lands in joe-rs first (§8.2).*
- **~~Q4.~~ RESOLVED 2026-08-16 from `training/joe/`.** The value is the
  observing seat's, `gamma = 1.0`, sparse ±1, draw-at-zero — an expected game
  outcome in [−1, 1], the same quantity `wdl_value` produced.
  `backup_value`'s `from_root` negation is correct as written and no rescaling
  is needed. Evidence and the three caveats it exposed: §1.8.
- **Q5.** *Partly resolved, and it moved the answer.* Joe's training **did**
  normalize over all 4,410 logits including the 441 unmasked pass cells
  (`bots/joe/joe_obs.py:387`, `networks/transformer.py:314`), so `logsumexp` is
  what the trained policy meant by "pass" — while `max` is what the deployed,
  rated joe does. §7.1 keeps `max` as the default and N3's experiment decides;
  what stays open is only whether the two agree in practice. Now tracked as R9,
  and it must close before N4 qualifies a configuration.
- **Q6.** What `prior_temperature` range is worth qualifying? §7.4 proposes
  `{1.0, 1.5, 2.0, 3.0}`; the sensible range depends on the prior-entropy
  histogram, which is measured at N3.
- **~~Q7.~~ RESOLVED 2026-08-16.** The bot id is `morpheus-joe`; the
  mythological convention is set aside so the name carries the lineage. See
  §0.
- **Q8.** Should the three retired parity surfaces' fixture strata be stripped
  from `parity-smoke.jsonl.gz`, or left as dead weight? §8.3 says leave them;
  that is a cost/benefit call, not a fact.
- **Q9.** Is `min_simulations` being inert (§1.4) intentional in
  `bots/morpheus-rs/` as well? If it was meant to enforce a floor, the port
  inherits a latent behavior change the moment someone fixes it. **N0 found a
  second one: `reserve_ms`.** It is parsed, stored on `RuntimeConfig`, carried
  through `to_runtime_config`, and read by nothing — `deadline_for_turn` is
  `turn_start + normal_deadline_ms` — and the Python sibling is the same. So
  every turn spends the full 140 ms and the 10 ms the file's own naming
  promises to hold back is not held back. Two dead knobs in one config is a
  pattern, and the question is now whether anything else in
  `deployment.json` is decorative.
- **Q10.** Do joe's build-legality rule and morpheus's `live_build_cost` agree
  cell-for-cell? Both implement the same documented formula, but after §6.2
  only morpheus's is consulted, and a disagreement would be silent.
  *Assert at N2.*
- **Q11.** Does the fork need its own `AugState` sequence fixtures, or does
  joe-rs's corpus suffice? The code is byte-identical (§8.2) but the *driver*
  differs — morpheus feeds it from `emit_observation`, joe-rs from the wire.
  N2's byte-equal-replies check probably settles this; if it does not, three
  full-game sequences must be re-captured through the fork's own path.
- **Q12.** `resident_memory_target_mb` needs re-deriving. Trivially safe
  against the 2 GB cap, but the number in `deployment.json` should not be a
  leftover.

---

## 14. Docs to update when this lands

Per docs-keeper rules, implementation knowledge splits into small topic files
rather than growing this plan:

- `docs/bots/morpheus-joe/` — `index.md`, `net.md` (the remap and value
  conversion), `parity.md` (what §8 actually achieved), `packaging.md`.
- [`docs/bots/joe-rs/export.md`](../joe-rs/export.md) — the fork joins "After a
  joe re-export".
- [`docs/bots/morpheus-rs/belief.md`](belief.md) — a pointer noting the fork's
  belief filter is network-free; the morpheus-rs page itself is unchanged.
- `docs/index.md` — new entries.
- `docs/research/measurements/` — N0's latency split, N4's qualification, N6's
  two rounds.
- The `joe-artifact-fanout` memory note — a fifth downstream surface.
