# joe-net N0 — does the search budget survive a 21 ms forward?

Phase N0 of the [joe-net port plan](../../bots/morpheus-rs/joe-net-plan.md):
the kill gate, run **before any of joe's network is ported**. It answers one
question — how many simulations morpheus's search completes when a forward
costs what joe's forward costs — and it answers it with the real admission
controller rather than with arithmetic.

Measured 2026-08-16. Six things came out of it that the plan did not have, and
three of them move the verdict: the forward is essentially the whole move, the
turn is 10 ms longer than anyone thought, and the shipped leaf batch completes
no simulations at all.

- Bots: `joe-rs` (stage split), `morpheus-rs` (baseline and spike)
- Latency host: Modal, one x86 core, `cargo` 1.97.1, `target-cpu=x86-64-v3`
- Controller host: Apple M3 Pro, serial schedule, panel `cm_expander` /
  `macaria`, seeds 7000–7019
- Raw records: [joe-net-n0-latency-modal.json](joe-net-n0-latency-modal.json),
  [joe-net-n0.json](joe-net-n0.json)

## Verdict: K0's middle row

**p50 completed simulations is 5 at `pending_leaf_batch: 1`** — 4 at batch 4 on
an ordinary turn, 3 at batch 1 when the belief filter recovers, and **0 at the
shipped batch 4 on a recovery turn**. That is K0's `4 – 7` band:

> proceed **only** with §3.4 (enemy priors dropped), §6.2 (no penalties build),
> §7.4 (`prior_temperature`) all counted in, and `widen_freeze_below` lowered
> so widening can run at all.

It is not the `< 4` stop row, and it is not the `≥ 8` clean proceed. Every
number below is what that reading rests on.

## 1. Where joe's move goes — the forward is 98% of it

`joe-rs bench --stages`, 2,280 turns of `synthetic-long.in.log`, one x86 core.

| stage | mean ms | p50 | p99 | share of mean |
| --- | ---: | ---: | ---: | ---: |
| parse | 0.012 | 0.010 | 0.027 | 0.06% |
| raw + cost | 0.010 | 0.009 | 0.026 | 0.05% |
| mask | 0.008 | 0.007 | 0.019 | 0.04% |
| `augment_obs` | 0.045 | 0.042 | 0.087 | 0.21% |
| normalize | 0.004 | 0.003 | 0.007 | 0.02% |
| **forward (F)** | **20.879** | **20.811** | **23.010** | **98.2%** |
| decode + reply | 0.003 | 0.003 | 0.004 | 0.01% |
| value on stderr | 0.291 | 0.274 | 0.514 | 1.37% |
| total | 21.251 | 21.197 | 23.444 | — |

Stages are sorted independently, so the p99 column does not sum; only the mean
column does. Detail and the arm64 comparison: [joe-rs
latency](../../bots/joe-rs/latency.md).

**Q1 resolved: F = 20.81 ms p50, 23.01 ms p99.**
**Q2 resolved: `augment_obs` = 0.042 ms p50.**

Three consequences, all against the plan as written.

**§5.1's F is too low.** The plan assumed `augment_obs` ≈ 2 ms and inferred
F ≈ 19 ms from the 21.4 ms whole-move figure. The obs pipeline is 50× cheaper
than that, so F is essentially the whole move. The budget is *tighter* than
§5.1 projected, not looser.

**§6.3's fork closes on its cheapest branch.** The rule was "< 0.5 ms →
advance per node, full history". 0.042 ms is an order of magnitude inside it.
The 14 history planes are advanced per node, R3's off-distribution risk never
arises, and R8 — both branches closed — cannot fire.

**§6.2's saving is not a saving.** Dropping `compute_valid_move_mask` +
`compute_build_mask_from_raw` + `prepare_action_mask` from every node saves
**0.007 ms**. The decision to pass zero penalties still stands on its own
merits (morpheus's mask is the one that must be consulted), but it should not
be described as a performance win.

## 2. What the shipped bot spends its forwards on

20 games, 9,577 normal moves. `forward_by_consumer` had to be added to the
trace — the controller has tracked it since the port and only the total was
written out.

| consumer | forwards/turn (mean) | share |
| --- | ---: | ---: |
| `root` | 1.00 | 4% |
| `enemy_prior` | 6.66 | 28% |
| `leaf_batch` | 15.84 | 67% |
| `belief_proposal` | 0.00 | 0% (`use_policy_proposal: false`) |
| total | 23.50 | |

**§3.4 removes 28% of the forwards**, which is a larger share than the plan
implies. The baseline reproduces M6: p50 16 completed simulations, move p50
102 ms / p99 134 ms, 1 of 9,577 normal moves over the judge's 150 ms.

## 3. `reserve_ms` is inert, and it is worth half a forward

`deployment.json` carries `reserve_ms: 10.0`. It is parsed
(`runtime/deployment.rs`), stored on `RuntimeConfig`, carried through
`to_runtime_config`, and **read by nothing**. `deadline_for_turn` is
`turn_start + normal_deadline_ms`, full stop. The Python sibling is the same:
`bots/morpheus/runtime.py` declares the field and never consults it.

So the usable turn is **140 ms, not the 130 ms §5.1 assumes**, and the 10 ms of
safety margin the config's own naming promises is not held back. The spike
confirms it from the other side: a median-belief turn reports a virtual
`move_ms` of exactly 140.

This is the same defect as the inert `min_simulations` the plan already records
as Q9, found the same way, and it should join it. Whether it explains the
baseline's one move over 150 ms is untested — a 154 ms move would have had 10 ms
more headroom under an honored reserve, which is suggestive and not a
measurement.

## 4. The spike: the real controller at a 21 ms forward

`fixed_forecasts_ms` + `charge_fixed_forecasts` run the real `can_admit`, the
real widening freeze and the real degrade bands over injected component costs.
Both keys are new parse additions; the controller fields they reach have
existed since the port and only its tests could set them. Mechanism and its
three caveats: [telemetry](../../bots/morpheus-rs/telemetry.md).

Every arm injects `root_inference` = F and `leaf_batch` = F × batch, zeroes
`enemy_prior_batch` (§3.4) and `belief_tensor` (§3.2), and leaves every other
component at its measured per-call cost. 20 games per arm.

**One caveat decides how to read this.** A charged clock advances by the
*forecast*, so a table of p99s does not model a p99 turn — it models a run in
which every turn is a p99 turn. That matters because `particle_transitions` is
bimodal by a factor of ~500:

| component | p10 | p50 | p90 | p99 | mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| `particle_transitions` | 0.028 | **0.059** | 6.552 | **28.922** | 1.913 |
| `belief_proposal` | 0.370 | 0.418 | 0.472 | 0.543 | 0.421 |
| `selection` | 0.092 | 0.107 | 0.122 | 0.150 | 0.108 |
| `backup` | 0.035 | 0.057 | 0.095 | 0.194 | 0.063 |

`filter_step` and the `recover_belief` it may trigger are charged to one
component — `runtime/controller.rs` says so at the top of the file — so p99 is
a recovery turn and p50 is a filtered one. The arms therefore bracket it:
non-network components at p50 (the ordinary turn) and at p99 (the recovery
turn).

Five arms, 20 games each, ~10,000 normal moves apiece. `F` is what
`root_inference` and one leaf were injected at.

| arm | non-network at | F | batch | completed simulations | degrade `policy` | tree p50 | virtual move ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline (no spike) | measured | 4.99 | 4 | **16** | 51% | 14 | 102 (real) |
| spike | p99 | 23.01 | 4 | **0** | **100%** | 1 | 52 |
| spike | p99 | 23.01 | 1 | **3** | 46% | 4 | 123 |
| spike | p50 | 23.01 | 4 | **4** | 48% | 3 | 117 |
| spike | p50 | 23.01 | 1 | **5** | 46% | 6 | 140 |
| spike | p50 | 20.81 | 1 | **5** | 46% | 6 | 126 |

Every count is a point mass — a charged clock is deterministic, so an arm
answers "how many simulations fit under *these* costs", and the spread across
arms is the distribution.

**Batch 4 collapses on recovery turns.** The batch forecast is 92.04 ms against
87.5 ms left after the root, `can_admit("leaf_batch")` is false, the tree stays
at one node, and the bot plays **joe's argmax on 100% of moves** at a 52 ms
cost. On an ordinary turn the same configuration fits exactly one batch and
stops — 4 simulations, and no second batch, because 92 ms does not fit twice.

**Batch 1 dominates it on both kinds of turn**: 5 against 4 on an ordinary
turn, 3 against 0 on a recovery turn. §5.3's proposal to drop the batch is
therefore not a preference; batch 4 is a configuration that fails on the tail.

**Widening is frozen throughout.** `widen_freeze_below: 16` against a forecast
of 3 to 5 means the root never leaves its initial candidate set on any arm,
which is what the tree sizes show — 6 nodes at 5 simulations is a root plus its
visited children and nothing widened.

**The gate/spend conflation does not move the answer.** The last arm injects F
at its p50 rather than its p99 — the closest a charged clock can get to "the
controller forecasts p99 and then spends what it spends" — and returns the same
5. So the arm at F = p99 is not pessimistic in a way that matters.

An analytic model over the same measured costs reproduces **all five** arms
exactly (0, 3, 4, 5, 5), which is what licenses reading unmeasured cells off it.
Since `particle_transitions` is at or below its p90 on 90% of turns, the
turn-weighted p50 for the port's configuration is **5**.

## 5. What N0 changes in the plan

| plan item | status after N0 |
| --- | --- |
| Q1 — forward-only cost | **resolved**: 20.81 p50 / 23.01 p99 |
| Q2 — `augment_obs` cost | **resolved**: 0.042 ms |
| §5.1 — 130 ms usable, F ≈ 19 ms | **both wrong**: 140 ms usable, F ≈ 21 ms |
| §6.3 — the history fork | **closed**: advance per node, full history |
| §6.2 — mask build a saving | **not a saving**: 0.007 ms |
| §5.3 — `pending_leaf_batch` 4 → 1 | **required, not optional**: batch 4 completes zero simulations |
| §5.3 — `widen_freeze_below` | must drop to ~2; at 16 widening never runs |
| R3, R8 — frozen history off-distribution | **retired**: the branch that needed them is not taken |
| Q9 — inert knobs | **a second one**: `reserve_ms`, in both bots |

## 6. What this does not answer

The spike prices the *budget*, not the *strength*. A four- or five-simulation
search over a near-one-hot prior may still be worth shipping or may not, and
nothing here says which — that is K1's job, and §7.4's "who's deciding" probe
at N3 is what would predict it. Two known distortions in the spike itself:

- **A component forecast at `0.0` is modelled as deleted, not as free.** The
  enemy-prior forwards still happen and still shape the tree; the port makes
  them uniform. The simulation *count* is unaffected, the tree shape is not.
- **Gate and spend are the same number.** The real controller forecasts p99 and
  then spends what it spends; a charged clock cannot separate the two. The
  F = 20.81 arm bounds it: same 5 simulations, so the conflation costs nothing
  here.

## Reproducing

```bash
modal run scripts/joe_rs_modal_stages.py
python scripts/morpheus_rs_m7.py sweep --config n8-s16-b4-d8 --games 20 --seed 7000 \
  --out $PWD/data/morpheus/morpheus-rs/joe-net-n0/baseline
python scripts/joe_net_n0.py spike-config <baseline-dir> --forward-ms 23.01 --batch 1 --quantile p50
python scripts/joe_net_n0.py report --baseline <dir> --spike b1 <dir> --forward-ms 23.01
```
