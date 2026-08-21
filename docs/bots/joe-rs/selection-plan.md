# joe-rs move selection plan

Plan for improving move selection from the network output. Today the bot is
one forward pass, `prepare_action_mask`, first-max `argmax`, `decode_action`
(`src/main.rs::act_staged`, `src/board/action.rs`). No search, no sampling,
no temperature. This plan proposes what to change, in what order, and how
each change is proved.

Two standing facts frame everything here:

1. **Training samples, deploy argmaxes.** PPO collected its data by sampling
   the policy; deployment takes the argmax. That mismatch is the measured
   cause of the castle limit cycle
   ([joe-argmax-limit-cycle](../../research/measurements/joe-argmax-limit-cycle.md)).
2. **joe-rs deliberately omits Python joe's repetition penalty**
   (`REPEAT_PENALTY = 2.0`, `REPEAT_DECAY = 0.90` in `bots/joe/agent.py`,
   2026-08-16 to 2026-08-20). This plan does **not** port that penalty — the
   Python implementation is treated as one measured data point, not as a spec
   (see S1's design-space review). Python joe then removed it and went back to
   the plain argmax, which does not change this plan: the review stands on the
   diagnostic's numbers, not on what the sibling ships. The sibling divergence
   **stays deliberate and widens**: joe-rs gets its own selection layer, and
   the parity boundary moves to the network output for every candidate (see
   "Parity stance", once, below).

Out of scope, all candidates: the stdio protocol, the export/artifact
pipeline (`convert_artifact.py`, manifest, safetensors), and everything under
`src/nn/` — selection reads `ForwardOut { logits, value, value_bins }` and
nothing upstream of it changes. The two `net.rs` lines pinned by
`tools/mutation_check.py` (the q/k-proj order and the softmax scale) are
untouched by every proposal.

## Step 1 — measured headroom (done, 2026-08-20)

Per the rule that a budget is sized from a fresh measurement, not remembered
numbers: `joe-rs bench` replays of three **real recorded competition games**
(`data/joe/joe-rs-parity/games/`, M7F4 corpus) through the full per-move
path, current HEAD, dev arm64 (M-series, unpinned):

| game | turns | p50 | p90 | p99 | max | startup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| castle_rush-seed3 | 803 | 26.21 | 26.75 | 26.97 | 28.09 | 68.7 ms |
| macaria-seed1 | 656 | 26.40 | 26.67 | 27.16 | 28.21 | 56.0 ms |
| aegis-seed0 | 262 | 26.61 | 26.80 | 27.06 | 27.42 | 45.0 ms |

`bench --stages` on castle_rush-seed3: **forward 26.44 ms mean of a 26.48 ms
move (99.8%)**. Parse + raw + mask + augment + normalize together are
0.03 ms; decode 0.005 ms; the value line on stderr 0.007 ms. One outlier
turn hit 45.2 ms (unpinned dev box). Selection today is five microseconds;
everything a new selection layer could add short of another forward is free.

**Headroom arithmetic.** The working deadline is 140 ms (RULES.md grants
150 ms per move; the repo budgets 140 everywhere). On this host:
140 − 27.2 (p99) ≈ 113 ms ≈ **4 extra forwards** per move.

**Caveat that decides the design:** the competition runs x86, and identical
code has measured 29.1–77.1 ms p50 across Modal fleet generations
([latency.md](latency.md)). On the slow generation even **one** fixed extra
forward can break the budget (2 × 77 > 150). So no candidate may hardcode a
forward count: any multi-forward path needs a runtime per-forward EMA (the
warmup forward seeds it) and must degrade to zero extra forwards when the
EMA says so. S1/S2 are compute-invariant and immune to this.

Step 1b, before any multi-forward candidate is built: re-measure on Modal
x86 via `scripts/joe_rs_modal_bench.py`, reading the *slow*-generation
number as the sizing input, same-host interleaved per
[modal-jobs](../../engine/modal-jobs.md) practice.

## The anti-repetition design space (why nothing is ported)

Any mechanism against the limit cycle picks a point on three axes: **what it
remembers** (nothing / cells acted from / exact actions / board recurrence),
**when it intervenes** (always-on vs detector-gated), and **how** (logit
penalty / hard veto / restored randomness).

Python joe's penalty was (cells-acted-from, always-on, logit penalty). Judged
on the diagnostic's own numbers, it is a weak point in the space — and joe
removed it on 2026-08-20 for these same reasons:

- **Always-on distortion.** It perturbed the policy on every revisit, cycle
  or not — and the diagnostic measured cycling *increase* under it
  (14.0% → 18.1% of turns; longest span 17 → 29). The win came from cycling
  at positive value instead of negative. It works, but not by the stated
  mechanism — a bad foundation to copy.
- **Unprincipled constants.** 2.0/0.90 were admitted guesses; outcomes are
  non-monotone in them (2.0 and 4.0 win, 3.0 loses on the diagnosed seed),
  and no round ever fitted them.
- **Granularity.** It penalised a cell's whole 10-channel action column,
  builds included, because the bot recently *left* that cell.

Other points considered and set aside: **exact-board-recurrence memory**
(measured useless — 641 turns, 641 distinct logit hashes; the board never
exactly recurs), **no-undo veto** (zero constants, but only catches
period-2; the measured loops include periods 1 and 4 and, worse, loose
confinement without strict periodicity), **strict cycle detectors**
(measured to undercount — confinement shows as few distinct source cells,
not as clean periods).

The strong point in the space is the one training itself validated:
**restored randomness**. Sampling dissolved every loop within ~21 turns
in-cycle, and it is the distribution the weights were trained under.
S1 makes it deterministic.

## Candidates

### S1 — deterministic Gumbel selection (recommended; implemented 2026-08-20)

**Status: shipped.** `src/board/select.rs`; knob `JOE_RS_TEMPERATURE`
(default 1, `≤ 0` restores the plain argmax); harness changes from the
"File and tooling impact" table all landed (self-golden reference, four
mutation plants, [parity.md](parity.md) grading row). At T = 1 the played
move departs from the old argmax on 12.9% of corpus turns. The smoke
decision stays a move (`selfcheck ok`, verified once as required below).
Rated round `s1-gumbel-r1` (2026-08-20/21, vs the same network's argmax
after joe dropped its penalty): **+3.78 ± 10.69, proven flat** —
[joe-selection-s1](../../research/measurements/joe-selection-s1.md). T = 1
stands; the T shortlist is moot unless the pending r2 replication
disagrees.

The Gumbel-max trick: `argmax_i(logits_i + T · G_i)` with i.i.d. standard
Gumbel noise `G_i = −ln(−ln u_i)` is an **exact** sample from
`softmax(logits / T)`. Derive `u_i` from a hash of
`(turn, action index, board digest)` — splitmix64 or similar, no
dependencies — and the draw is exact temperature sampling while the bot
**remains a pure function of the game**: replays reproduce byte-for-byte,
and the "replies byte-identical" invariant used by every kernel sweep
survives. The turn number in the hash guarantees a fresh draw on every
revisit, so escape from any loop is geometric in time — the same property
that dissolved loops in training.

- **Benefit.** It is the mechanism training validated, with the smallest
  possible distribution shift (T=1 *is* the rollout policy). It is
  naturally targeted with zero state: a margin `m` flips with probability
  ≈ `e^(−m/T)`, so the measured 11.3-margin structure-pulls flip at ~1e-5
  while the 1.3-margin departure near-ties flip at ~22% (T=1). That is the
  surgical profile the penalty family approximates with counters — for
  free. Exploratory Python data: T=1.0 and T=0.25 both 5/5 on the
  diagnostic grid, T=0.5 3/5 (5 games — real effect, unranked grid).
- **Knobs.** One: `T`, with a principled default (T=1 = training
  distribution; lower sharpens toward argmax). Optional variant flag, off
  by default: an explicit margin gate `M` (argmax unchanged when
  top-1/top-2 gap ≥ M; noise only below). The gate is a *hard* version of
  what the softmax already does exponentially; add it only if round data
  shows a T high enough to break loops also dithers on moderately-confident
  turns. Start with one knob.
- **Cost per move.** ≤ ~0.1 ms (4410 hash + log evaluations; less if the
  noise is restricted to the top-k logits). Compute-invariant.
- **Risk.** Sampling can pick a genuinely worse move on near-ties where the
  policy's ranking was right — the round prices this. Two knob settings per
  round at most; shortlist T on the unrated 5-seed diagnostic grid first
  (adhoc games, deleted after, per the limit-cycle doc's precedent).

### S2 — detector-gated repetition response (fallback family, from zero)

Kept only as the fallback if S1's round shows a decisive-speed cost. A
fresh design, not a port: **detect the measured symptom** — confinement,
i.e. the count of distinct source cells over a trailing ~50-turn window
collapsing relative to owned land (the diagnostic's own table: 44 distinct
cells normally, 4–6 confined) — and only then apply a bounded, temporary
response (a targeted penalty on the confined region, or a forced top-2
deviation). Zero distortion outside detected confinement, which is exactly
the property the Python penalty lacks. Costs: two detector constants that
have to be fitted, and a rule bolted onto a pure policy. Latency ~µs.

### S3 — depth-1 value re-rank (killed at Gate 1, 2026-08-21)

**Status: dead.** Gate 1 ran offline over the step-10000 corpus
(`tools/s3_gate1.py`, 5,632 turns) and hit the kill criterion:
[joe-s3-gate1](../../research/measurements/joe-s3-gate1.md). 68.7% of
near-tie top-2 pairs fabricate byte-identical successors (half/full
twins), the rest carry a median value gap (0.0033) below the median
fabrication error (0.0037), and p90 fabrication error is 2.6× the p90
move-to-move value movement. The value head does not discriminate
adjacent moves — the plan's own suspicion, now measured. Gate 2 never
needed to run; no game was spent. The original design follows for the
record.

Today the value head is computed every turn and discarded on stderr. S3
spends measured headroom to use it. Per turn: take the top-k masked
actions; for each, **fabricate** the successor observation — apply own-move
army arithmetic to the 14-channel raw tensor per RULES.md (source loses,
target gains or flips owner), advance the scalar channels, hold opponent
and fog static; run each fabricated frame through a **scratch copy** of the
persistent obs state (never the real history rings); forward each; pick by
blended `logit + λ · value`. Depth-1 greedy re-rank — no tree, no opponent
model.

- **Benefit.** The policy is reactive and myopic; the confinement data
  shows the critic knew the game was being lost (value → −0.9) while the
  policy shuffled. A value re-rank can refuse a stack-losing move or take a
  capture the policy is indifferent about. Still speculative: adjacent
  candidate moves may score near-identically, so it may not even break
  loops.
- **Cost per move.** k extra forwards: 26.5 ms each on dev arm64, 17–77 ms
  across the x86 fleet. Mandatory per-forward EMA + deadline guard,
  degrading to k=0. Largest new code surface: a forward model
  (`src/board/model.rs`) that does not exist, plus snapshot semantics for
  `AugState`.
- **Gate 1 (free, offline, first).** The net was trained on real
  trajectories; a fabricated frame is off-distribution and its value may be
  noise — and the needed signal (value differences between adjacent moves)
  is intrinsically small. Over the existing corpus, fabricate the successor
  for the action joe actually took at each turn and correlate
  `value(fabricated)` against `value(real next frame)`. If fabrication
  noise drowns typical move-to-move value differences, S3 dies without
  costing a game.
- **Gate 2.** Step 1b's slow-fleet x86 number must leave room for k ≥ 2
  under the EMA guard.
- **Budget-safest variant: margin-gated lookahead.** Spend the extra
  forwards only on near-tie turns (~5% of turns have margin < 0.1) — the
  policy has abstained, the value head gets the tiebreak. This is the
  "value-aware tie-breaking" shape, and it is compute-light enough to
  survive the slow fleet with a fixed small k.

### Rejected

- **Porting the Python penalty** — see the design-space section.
- **Unconditional true-RNG sampling** — dominated by S1 (same distribution,
  loses replay determinism and the byte-identical sweep methodology).
- **Explicit periodicity detectors as the primary mechanism** — measured to
  undercount the symptom; folded into S2's confinement detector instead.

## Does search make sense? (assessed: no beyond depth 1)

With the big net as the only evaluator, the budget caps everything:
140 ms ÷ 17–77 ms per forward = **2–8 evaluations per turn, worst case 2**.
Any tree collapses to that node count, so "search" *is* S3 — depth-1
re-ranking. Deeper hits three walls at once: **eval cost** (depth-2 over
k own moves × m replies is k·m forwards — over budget at 2×2 on a slow
host; real search needs a microsecond leaf eval, i.e. a handcrafted
evaluation or a distilled net — the first turns joe into a heuristic bot,
the second is a training project), **simultaneity** (generals resolves
moves simultaneously; alternating minimax is the wrong game), and **fog**
(the true state is a belief, not the observed board). This repo already
has the bot that pays those costs — morpheus — and even it affords 3–5
simulations with a joe-scale net at the leaves. Verdict: S3, plain or
margin-gated, is the sensible maximum for joe-rs; real search is out of
scope for this bot.

## Recommendation and order

1. **S1 now.** Shortlist T on the unrated diagnostic grid; then a rated
   round with two T arms against the frozen baseline.
   *(Done: shipped 2026-08-20; round `s1-gumbel-r1` proven flat, T = 1
   stands — r2 replication pending.)*
2. **S2 only if** S1's round verdict shows the cost lands on decisive play.
   *(Dead: no such cost shown.)*
3. **S3 behind its two gates**, cheapest variant (margin-gated) first, and
   only if S1/S2 leave losses on the table that a one-step value could have
   refused. *(Dead: killed at Gate 1, 2026-08-21 —
   [joe-s3-gate1](../../research/measurements/joe-s3-gate1.md).)*

**The selection layer is complete: S1 Gumbel T = 1 is the end state of
this plan.** Any future value-guided selection needs a value head trained
to discriminate adjacent moves, which is a training project, not a
selection-layer change.

## Parity stance (one policy for every candidate)

Nothing is ported, so the `.out.log` (deployed joe's own replies) never
becomes joe-rs's reference and the sibling divergence stays deliberate. One
harness policy covers S1–S3:

- **The graded contract stays the network**: tiers 1–2 and the
  `decide`/`forward`/`sequence` surfaces against the `.npz` oracle —
  unchanged, since no candidate touches `src/nn/`.
- **The full played path gets a joe-rs self-golden**: the binary's own
  recorded replies over the corpus, regenerated on export alongside the
  other fixtures (`capture_fixtures.py` grows a mode for it), so an
  unexplained selection change still fails a test. Every candidate is a
  deterministic function of the game, so byte-stable goldens remain
  possible — this is a hard requirement on any future selection idea.
- **Selection gets its own unit tests + mutation plants**: hash ignores the
  turn (same draw every turn — the exact failure that would silently
  restore the limit cycle), noise applied to masked entries lifting an
  illegal action, temperature sign/scale, and for S3 forward-model plants.
  Each must be killed.
- [parity.md](parity.md)'s "who is graded on what" table gains a row:
  joe-rs's selection layer is graded on **its own** goldens, nobody grades
  it against deployed joe.

## Validation — every candidate, no exceptions

**Matchup gate first** (repo verification gate; note the absolute
interpreter path — a relative one BrokenPipes silently):

```bash
PYTHON=$PWD/.venv/bin/python python competition-module/competition/matchup.py \
  bots/joe-rs/run.sh bots/cm_expander/run.sh --mode competition --seed 0
```

**Then a measurement round per [decision-rule](../../arena/decision-rule.md),
all arms inside one round:**

- Freeze the pre-change bot as `bots/joe_rs_base/` for the duration (the
  `macaria_base` precedent), register every arm's hash, and make sure the
  frozen copy's `run.sh` prose names no outside paths (the closure scan
  follows path strings in comments).
- `python -m arena.tournaments.competition` with the arms, the frozen
  baseline, and a panel of ≥ 5 bots including the anchor `cm_expander`;
  pinned `--round-seed`; `--seat-policy alternate`; `--strict-versions`;
  ≥ 200 games per arm minimum, ~1150 per arm to afford a ±25 CI.
- Verdict from `fits["<round>"].delta(base, candidate)` quoted as
  Δ ± SE, CI₉₅, P(B > A), games per arm — never a rank, never across
  rounds. Games stored under `data/games/<round>/` before the refit.
- **Replicate before publishing**: a second, separately scheduled round.
  S1/S2 are compute-invariant, so the deadline-driven fragility that
  produced the M6 replication failure binds them least — but the rule
  stands. S3 is exactly the deadline-driven case it binds hardest; note
  host state by hand for its rounds.

The parity suite (`pytest bots/joe-rs/tests/ -m joe`) and
`tools/mutation_check.py` must pass before the round, after the harness
changes above. The default 15 s suite budget is unaffected (the `joe`
marker sits outside it).

## File and tooling impact

| Area | S1 | S2 | S3 |
| --- | --- | --- | --- |
| new `src/board/select.rs` | Gumbel noise, hash-PRNG, T (+ optional gate) | + confinement detector, response | + candidate ranking, λ blend |
| `src/main.rs` (`Seat`) | call select in `act_staged` | + trailing source-cell window | + EMA, deadline guard, scratch state |
| `src/board/obs.rs` | — | — | snapshotable `AugState` |
| new `src/board/model.rs` | — | — | own-move forward model |
| `src/nn/*`, artifact, export pipeline | — | — | — |
| `tests/test_wire_replay.py` | reference → joe-rs self-golden | same | same |
| `tools/capture_fixtures.py` | + record self-golden | same | same |
| `tools/mutation_check.py` | + turn-blind hash, masked-lift, T plants | + detector plants | + forward-model plants |
| [parity.md](parity.md) | + self-golden row in the grading table | same | same |
| `run.sh`, stdio protocol, `selfcheck` | — (Gumbel noise is unbounded, but the draw is a deterministic function of the frame: verify once at implementation that the smoke decision stays a move, and it stays one forever) | — | — |

Every candidate forks the bot's content hash (any `src/**.rs` edit does) —
that is normal: register the new version, and the round contrasts it against
the frozen baseline hash.

## Standing-divergence ledger (fact 2 resolved)

joe-rs and deployed joe diverge on wire replays today (Gumbel selection
against joe's plain argmax) and diverge **more** after any candidate here
ships. The divergence is
deliberate, permanent under this plan, and documented in
[parity.md](parity.md); the graded boundary is the network output, and the
full-path reference is joe-rs's own golden, never the sibling's.
