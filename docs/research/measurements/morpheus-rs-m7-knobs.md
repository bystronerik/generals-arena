# Morpheus-rs M7 — deriving the knobs

Milestone M7 of [the rewrite plan](../../bots/morpheus-rs/rewrite-plan.md),
§8's procedure. Everything here is **unrated**: the sweep drives `matchup.py`
directly, so no game record and no lineage step. The strength half of the
decision is a separate round; see
[the contrast](morpheus-rs-m7-strength.md).

Hosts: Apple M3 Pro for the sweep, and a one-core Modal x86 Linux container for
the qualification — which is the only one with deployment authority, because
the competition host is one x86 core running Linux.

## The cost model is one number

Every network forward costs **~4.9 ms** on the M3 Pro and **~5.1 ms** on x86,
and a turn affords about twenty-six of them. That single figure explains the
whole grid:

| component | calls per turn | what it is |
| --- | ---: | --- |
| `root_inference` | 1 | one forward |
| `leaf_batch` | ≈ 4 | `pending_leaf_batch` forwards each |
| `enemy_prior_batch` | ≈ 4–5 | a forward per uncached enemy view |
| `particle_transitions` | 1 | the belief, and the only non-inference cost that matters |

Everything else — selection, backup, hashing, the reply — is under 7 ms per
turn combined at the swept configurations.

## The sweep

Thirteen configurations, two games each on the M3 Pro (~1,000 turns per
config), one axis moved at a time off the parity knobs.

| config | move p50 | p99 | p99.9 | max | >150 ms | sims p50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `n8-s16-b4-d2` (parity) | 102 | 133 | 140 | 140 | 0 | 16 |
| `n8-s32-b4-d2` | 128 | 140 | 140 | 140 | 0 | **20** |
| `n8-s64-b4-d2` | 128 | 139 | 140 | 140 | 0 | 20 |
| `n8-s16-b4-d8` | 106 | 137 | 139 | 140 | 0 | 16 |
| `n8-s32-b4-d8` | 129 | 139 | 140 | 157 | 1 | **20** |
| `n8-s32-b2-d8` | 134 | 140 | 140 | 145 | 0 | 20 |
| `n8-s32-b4-d16` | 128 | 140 | 148 | 148 | 0 | 20 |
| `n8-s32-b2-d16` | 134 | 140 | 143 | 152 | 1 | 20 |
| `n16-s16-b4-d2` | 114 | 139 | 140 | 146 | 0 | 16 |
| `n16-s32-b4-d8` | 128 | 139 | 140 | 140 | 0 | 16 |
| `n32-s16-b4-d2` | 125 | 139 | 151 | 151 | 1 | 16 |
| `n64-s16-b4-d2` | 126 | 140 | 148 | 148 | 0 | **12** |
| `n8-s16-b8-d2` | 97 | 126 | 163 | 163 | 1 | 16 |

Four readings:

- **`target_simulations` 16 → 32 buys four real simulations.** 64 buys nothing
  more: the deadline binds at ~20, so the third value on §8's axis is
  unreachable rather than expensive. M6's note was right that 16 was a ceiling
  fitted to a bot that could not reach it.
- **Particles trade against search, at about four simulations per doubling.**
  n=64 finishes 12 simulations where n=8 finishes 20. The belief is not free
  and the search is what the deadline is short of.
- **`search_depth` 2 → 8 is nearly free** — +4 ms at the median, same
  simulation count. The cost of a simulation is its leaf forward, not the tree
  walk; depth 16 starts to show in the tail.
- **`pending_leaf_batch = 8` has the best median and the worst tail** (97 ms
  and 163 ms). That is exactly what a zero admission guard predicts: a bigger
  batch is more work committed between two chances to say no.

Two games is ~1,000 turns, which resolves a p99 and not a p99.9 — the ">150 ms"
column flips on a single move at this sample size and should be read as a hint,
not a gate. The gate is measured below, on 10,000 turns.

## The reserve does not exist

§8 says to derive `normal_deadline_ms` and `reserve_ms` from the measured reply
cost and scheduler jitter. **`reserve_ms` is parsed, stored, and never read** —
in the Python oracle as much as in the port, so this is a faithful transcription
of a dead knob rather than a porting gap. The turn deadline is
`turn_start + normal_deadline_ms` and nothing subtracts a reserve from it.

The reply cost, meanwhile, is 0.0002 ms at p99. Sizing a reserve from it would
have been sizing the wrong thing anyway.

What does protect the tail is `admission_guard_ms` — the margin `can_admit`
demands beyond a component's forecast — and it ships at **0.0**. M6's two moves
over the judge's limit were forecast misses, not budget misses: `leaf_batch`
cost 72 ms and 110 ms against a forecast near 20, and with a zero guard the
controller admitted them.

Measured, on four games per cell:

| `admission_guard_ms` | config | p50 | p99 | p99.9 | max | >150 ms | sims |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `n8-s32-b4-d2` | 129 | 140 | 149 | 149 | 0 | 20 |
| 0 | `n8-s32-b4-d8` | 129 | 140 | 140 | 153 | 1 | 20 |
| 15 | `n8-s32-b4-d2` | 114 | 125 | 127 | 130 | 0 | 16 |
| 15 | `n8-s32-b4-d8` | 112 | 125 | 128 | 135 | 0 | 16 |

**Fifteen milliseconds of guard costs four simulations and buys twenty
milliseconds of tail.** That is the trade §8 wanted derived; it is available,
priced, and — per the qualification below — not needed on the host that
decides.

## Qualification, on one x86 core

Twenty games per configuration, ~10,000 normal moves each, `cpu=1`, rustc
1.97.1, `admission_guard_ms = 0`.

| config | turns | p50 | p99 | p99.9 | max | >150 ms | sims p50 | belief ok |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `n8-s16-b4-d2` (parity) | 10,382 | 94 | 131 | 138 | 140 | **0** | 16 | 100% |
| `n8-s32-b4-d8` (candidate) | 10,595 | 130 | 140 | 140 | **141** | **0** | **20** | 100% |

**The gate's latency half is met**: p99.9 = 140 ms against a 150 ms limit, zero
overruns over twenty games, at twenty simulations per move.

The container was *tighter* than the laptop — max 141 ms against 153 for the
same configuration — which is the opposite of what M0 found for the Python bot
(p99.9 407 ms there against 177 locally). The difference is not the host being
kind: it is that this binary is not spending 100 ms of every turn in a belief
update it cannot finish, so it is never in the position where a scheduling
hiccup lands on top of an already-overrun turn.

### Per call, and where the two hosts disagree

| component | M3 Pro ms/call | x86 ms/call | x86 / M3 Pro |
| --- | ---: | ---: | ---: |
| `particle_transitions` | 17.05 | 29.43 | 1.73× |
| `leaf_batch` (4 forwards) | 19.78 | 16.23 | 0.82× |
| `root_inference` | 4.96 | 5.12 | 1.03× |
| `enemy_prior_batch` | 12.86 | 11.24 | 0.87× |

x86 is **faster at inference and slower at the belief kernel**, by enough that
a table fitted on the laptop would misprice both. This is the concrete reason
§8 puts the qualification on the reference host, and it is why the shipped
`offline_p99_ms` comes from the x86 column.

## What this does not settle

- **One container is not the judge.** M0's CPU probe found Modal placing
  containers on different fleet generations, including one without AVX-512, and
  this run reports 17 visible CPUs against its one-core reservation. It is a
  qualification against *a* one-core x86 Linux host, which is the closest thing
  that can be measured.
- **The guard is priced but untested at the margin.** 0 and 15 ms were
  measured; nothing between them was. If a future host shows the tail M6 saw
  locally, the table above says what a guard costs before anyone has to guess.
- **`belief_tensor` has no p99 here.** Both bots time only `first_move_setup`
  with it, so it has one sample per game rather than one per turn; the shipped
  table carries its first-move measurement and says so.
