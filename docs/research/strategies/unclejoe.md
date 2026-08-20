# unclejoe — joe-rs's network, with the whole turn budget spent on top of it

> **The bot was removed on 2026-08-20.** `bots/unclejoe/`,
> `bots/unclejoe/fork-plan.md`, and `bots/unclejoe/tactics-plan.md` exist only
> in git history from that date, so the links to those two pages below are dead
> on disk. This spec stays as the record of what the bot claimed and what the
> shadow measurements tested.

Spec, written before the tactics code; revised 2026-08-15 after a design
review changed the goal from "override rarely, finish early" to "spend the
whole 150 ms turn budget". Bot: `bots/unclejoe/`. Substrate and milestone U1:
[`../../bots/unclejoe/fork-plan.md`](../../bots/unclejoe/fork-plan.md).
Module layout and milestones U2–U6:
[`../../bots/unclejoe/tactics-plan.md`](../../bots/unclejoe/tactics-plan.md).

Rule references: [`RULES.md`](../../../RULES.md) sections 02 (move order),
03 (castle building), 04 (growth), 05 (combat), 06 (visibility), 07
(deathtouch, draw), 08 (150 ms per move).

## 1. The claim

`bots/joe-rs` plays the frozen joe network greedily: one forward pass, argmax
over the action logits, no search. That spends about 24 ms of a 150 ms turn
(RULES.md §08) and throws the rest away. unclejoe keeps the policy unchanged
as the move *prior* and spends the whole clock — on the only two computations
the budget can trust: exact rule arithmetic, and the network's own value
head.

Two hypotheses, separately falsifiable:

> **H1 — provable moments.** A greedy network policy loses measurable
> strength at the moments where the correct move is provable — a forced
> general capture within a few plies, and the reply that stops one — and a
> bounded exact search that fires only there recovers part of it, without
> touching the policy elsewhere.
>
> **H2 — one-step improvement.** Given ~100 ms of spare clock, choosing
> among the policy's own top-k moves by the network's **own value head**,
> evaluated on one-ply afterstates, beats playing the argmax.

The failure modes are specific rather than mysterious. H1 failing looks like
a flat contrast while the shadow counters show the triggers firing. H2 fails
earlier and cheaper: if the shadow re-rank (milestone U4) shows candidate
value gaps sitting inside the head's noise, the re-rank never goes live and
the negative is recorded without spending a round.

**H2 took that exit, and came back narrowed** (U4, 2026-08-16). Playing the
best afterstate value would have moved the reply on 48.8% of 4,037 turns, and
on 83% of those the value gap was smaller than **one bin of the value head** —
128 bins over [−1, +1], so 0.0157 wide. The head's output is a continuous
expectation, so a smaller difference is representable; what it is not is
evidence, because bin width is the scale the head was trained to separate
outcomes at. **H2 as written is not supported.**

What the same data argued for is a minimum gap, and the layer now carries one:
a rival takes the turn only by clearing a bin. That refuses 84% of the value
head's preferences and leaves a mechanism acting on 6.6% of turns. So the claim
under test is no longer "the value head ranks the policy's top-k better than
the policy does" but the narrower **"where the value head separates two
candidates by a margin it was trained to resolve, it is right"** — which the
shadow data cannot answer, because a gap says the head distinguishes those
positions and not that it distinguishes them correctly. That is a rated arm
behind `UNCLEJOE_RERANK`, sized by §9, and it is U6's.

Numbers in [`../../bots/unclejoe/shadow.md`](../../bots/unclejoe/shadow.md)
§U4. No round was spent on the version that failed, which is what this
hypothesis was written to make possible. H1 is untouched by the finding and
still needs §9's contrast.

## 2. Why a fork, not a flag

unclejoe is a copy of joe-rs's crate, weights included by byte copy. joe-rs is
the **A arm of the contrast that decides this bot**, and the content hash
covers `src/`, `Cargo.*`, `run.sh`, and `artifact/model.safetensors` — so a
tactics hook added to joe-rs itself would move joe-rs's hash and retire the
rated entity the baseline refers to
([`../../arena/decision-rule.md`](../../arena/decision-rule.md),
*connectivity*). A fork makes that mistake unreachable.

It also makes the contrast mean one thing. Identical weights are not a nicety
here: two conversions of the same checkpoint would already be a second
difference between the arms, so unclejoe tracks **joe-rs's converted
artifact**, not joe's `.eqx`, and `bots/unclejoe/tools/sync_artifact.py` is
the only knob that moves it.

The cost is honest duplication — a fix to the shared board/nn code does not
reach across — and it is the right trade while the two are being *compared*
rather than composed.

## 3. Concept

Every turn, against an internal 130 ms deadline (20 ms of slack under the
limit for host jitter and reply emit), in this order:

1. update per-game memory from the frame (enemy general location, remembered
   mountains, last-seen general army);
2. run the full existing pipeline — frame → raw tensor, masks, augmented
   observation, one forward pass — unchanged and unconditional; read the
   argmax, the top-k candidates, and the root value;
3. evaluate two cheap trigger predicates; if one fires, run a bounded exact
   search for a *proof*; a proof plays immediately;
4. otherwise filter the candidates by exact rules (castle masks, refutation
   veto — §7) and re-rank the survivors by afterstate value — anytime, in
   policy order, until the clock reserve;
5. play the best surviving value; degraded (early cutoff, everything
   filtered), the argmax.

The forward pass always runs even when an override wins. Every move played
is one of: the argmax, a top-k policy candidate promoted by the network's
own value head, or a proof-backed override. No hand-written evaluation
function ever ranks a move.

## 4. Triggers (exact, and a deliberate superset)

Both are integer scans over the parsed frame plus memory. A trigger that fires
on a dead position costs one search that declines; a trigger that misses a
live one costs the whole point of the bot. So both are loose on purpose.
`D` is the search's depth budget in our moves (proposed 3, §5), and range is
measured in **moves** — a bounded BFS over every cell not remembered as a
mountain, which excludes nothing that could arrive in `D` moves and does
exclude armies a permanent wall stands between.

**Kill.** The enemy general has been seen (its cell is remembered — a general
never moves), and some cell we own with `army > 1` lies within `D` moves of
it, and either

- `obs.turn ≥ 800` — deathtouch is live, so any unit is potentially lethal
  (RULES.md §07); or
- our total army within that neighborhood exceeds the general's **last seen**
  army — a stale lower bound, which keeps the superset property while the
  search applies the honest pessimistic bound before it proves anything.

Before the general has been seen the trigger cannot fire. That is correct, not
a gap: no exact kill is provable against an unlocated general.

**Immediate defense.** Never before first contact — the turn we first see any
enemy cell, remembered and latched. Ungated, the fog arm below fires on
essentially every pre-contact turn, since `hidden` with nothing of the
opponent ever seen is their entire army; gated, it is silent until they show
themselves. This is the layer's **one assumption rather than a bound**: it is
exact only through turn 13 (spawn distance ≥17 BFS steps, one move per turn,
so nothing enemy can be within `D` of us yet), and after that it trades the
superset property for the pre-contact searches. After contact, with `g` our
general (always known), fire if

- any *visible* enemy cell within `D` of `g` holds `army ≥ army(g)` — loose:
  it ignores the leave-one-behind rule and multi-step attrition; or
- `obs.turn ≥ 800` and any visible enemy cell with `army > 1` is within `D`; or
- a fogged cell lies within `D` of `g` **and** the hidden-army budget (§6)
  reaches `army(g)` — a stack could be sitting one step outside vision.

Vision is the 3×3 pool around owned tiles (RULES.md §06), so every cell
adjacent to our general is always visible: the fog arm only matters at
`D ≥ 2`.

Expected fire rate is low — most turns have nothing within three tiles of
either general. U2 measured it rather than assuming it
([`../../bots/unclejoe/shadow.md`](../../bots/unclejoe/shadow.md),
2026-08-15). The expectation holds for the kill trigger (0–7% of turns) and,
once the contact gate is in, for the defense trigger in three of four gate
matches (0–0.6%). It does **not** hold in the fourth: a general with a fog
pocket permanently inside `D` fires the fog arm on 88% of turns. The arm is
bimodal, not merely loose, and tightening it further — a hidden-army bound
per reachable fogged cell instead of the global budget at every cell — is a
decision this spec owes its next revision.

## 5. Search model and caps

**Forward model.** A local simulator over `(passable, owner, army)` per cell,
implementing the engine's own resolution exactly: simultaneous moves under the
§02 priority ladder (chasing > reinforcing > smaller-army-first, ties falling
through), §05 combat (strictly more captures, a tie keeps the defender, the
mover leaves one behind, full or half split), §04 growth by turn parity, and
§07 deathtouch from turn 800 with its chase defense. Both generals captured on
the same turn is a **draw**, and a draw counts as failure for both the kill
proof and the defense proof — conservative in both directions.

**Search.** Depth-limited minimax over *(our move, their reply)* pairs, ours
maximizing. Move generation is windowed to Manhattan distance `D + 1` of the
relevant general; ours come from owned cells with `army > 1`, theirs from
every enemy-possible cell in the window — fogged cells included, under §6 —
plus a pass. It returns a move **only when the goal holds against every
opponent reply at every ply**. Anything less is not a proof and is not an
override.

Proposed caps, all constants, all quoted here so a change to them is a change
to this spec:

| Cap | Proposed | Why |
| --- | --- | --- |
| Depth, kill | 3 of our moves | Unchanged: beyond three plies the fog bound (§6) dominates and proofs stop landing — the binding constraint is knowledge, not compute. |
| Depth, defense | 1 | Set in U3, and not for budget. At one ply the fog bound costs nothing — vision is the 3×3 pool around owned cells, so every cell that can reach our general in one move is visible. At two it costs everything, since a fogged cell two steps out is credited with their whole unaccounted army. |
| Node budget | 300,000 | The clock-independent ceiling, scaled from 100k at the old 20 ms deadline. |
| Deadline | 60 ms per search, checked every 1,024 nodes | Up from 20 ms: trigger turns are where exactness matters most, and the clock exists to be spent. Per search, not per turn — both triggers can fire on one turn, and a declining kill must not starve the defense behind it. |
| Fog growth margin | 3 | Added to the hidden bound: a fogged cell could be a castle or general producing on the §04 clock, and a 50-turn tick can land inside the horizon. An aged bound has to still be a bound. |

Hitting either cap declines, and a decline is never a half-searched move: it
falls through to the filtered re-rank (§7).

Measured in U3 ([`../../bots/unclejoe/shadow.md`](../../bots/unclejoe/shadow.md),
2026-08-16): across four gate matches neither cap ever bound. The kill search
declined 87 of 88 fires by exhausting its tree, at ~2–3k nodes and ≤26 ms per
fire. The override fired **once in 2,900 turns** — one turn where the layer,
not the network, produced the reply — and that game ends the same way with the
override switched off. So no game moved. It is the first evidence on H1, and it
is weak.

## 6. Fog, pessimistically

The wire frame gives the opponent's total army exactly, so
`hidden = opp_army − Σ(visible enemy army)` is a sound bound on everything we
cannot see. The model: any currently-fogged cell that we have never seen to be
a mountain may be enemy-owned and may hold up to `hidden` army, independently
at every such cell.

This is deliberately unfair to us in both searches. For defense the fog is the
adversary's, so pessimism makes proofs harder and the override safer: we only
play a "saving" move that saves against the worst hidden army. For the kill it
bounds enemy reinforcement and chase sources, so a kill is proven only if it
lands against the last-seen general army plus growth since, plus the
worst-case reinforcement that resolves first.

The pessimistic model is a **proof device, not a predictor**. It is never
used to score re-rank lines (§7) — as a predictor it is paranoid, and a
search steered by it would collapse into never leaving home.

## 7. Spending the rest of the clock: filters and afterstate re-rank

Two mechanisms, both strictly weaker than the override, consume the budget on
ordinary turns.

**Exact-rule filters** remove candidates and never rank them. All three are
arithmetic from RULES.md, not judgment:

- *Castle crowding* (§03): the build price is 35 plus `max(0, 14 − 2·d)` for
  each own structure at Manhattan distance `d`. Mask any build whose total
  surcharge exceeds a cap (proposed **8** — only builds clear of own
  structures pass).
- *Castle lateness* (§03 + §04): production is 1 army per 2 turns, so payback
  takes `2 × price` turns. Mask any build after a turn threshold (proposed
  **650**).
- *Refutation veto*: a depth-1 exact check — mask a candidate iff some
  visible opponent reply captures our general after it. Sound, because it
  quantifies over every visible reply from the true root. This closes the
  first revision's known gap: the defense search declining and the bot then
  playing a move the search had already seen lose. **The check itself ships
  in U3**, one milestone early, as the gate on the defense override: without
  it a defense proof would answer "some move survives" on nearly every quiet
  turn and the override would replace the network's move with an arbitrary
  safe one. Masking candidates with it is still U4's.

The two castle thresholds are the only uncalibrated constants in this spec.
They sit behind their own switch, and the U4 shadow counts precede their
going live.

**Afterstate value re-rank** — one-step policy improvement. For each
surviving top-k candidate (`k = 4` — U4 measured that five does not fit the
turn) in policy order: advance it one ply through the exact forward model with
the opponent passing, render the resulting position as an observation, run the
forward, read the value head. Play the best value **that beats the policy's own
move by a minimum gap** — one bin of the value head, `2/127`, below which the
two positions are a tie and the tie belongs to the policy. U4 measured why the
gap is not optional: without it the mechanism overrules the policy on half of
all turns, almost always on a difference below the head's training scale.

Stop starting evaluations when the clock reserve is reached — the argmax is
evaluated first, so a cutoff at any point degrades to the old bot. The reserve
is **measured, not declared**: it is what the slowest evaluation this game
cost, seeded by the warmup forward, because a fixed reserve is a claim about a
host this code has never run on.

Known approximations, named rather than hidden:

- *Opponent-pass afterstates are optimistic.* The refutation veto covers the
  worst single reply against our general; every other reply is unmodeled.
  Modeling it is impossible in principle here, not merely expensive: the
  opponent's observation is not derivable from ours under fog, so there is
  nothing to run a mirrored policy on. Depth stops at 1 for that reason, not
  for budget.
- *Afterstates are slightly out of distribution.* The value head trained on
  real post-resolution frames; one-sided advances are near that distribution,
  not in it. The U4 shadow re-rank measures whether the head's
  discrimination survives before any move changes.

## 8. Latency budget

The budget is a resource, not a hazard to stay far from: the design rides an
internal deadline of **130 ms** and must never cross 150 (RULES.md §08); the
20 ms of slack absorb host jitter and reply emit. joe-rs's measured full-path
cost is **p99 23.7 ms, max 34.5 ms** on the Modal one-core x86-64-v3 proxy
([`../../bots/joe-rs/latency.md`](../../bots/joe-rs/latency.md), 2026-08-15),
and the forward pass dominates it — so the spare clock holds about four extra
forwards, which is exactly what the re-rank spends it on; a trigger turn
spends a 60 ms proof search plus roughly two forwards instead. Sim, filters,
and afterstate rendering are integer work, far below one forward.

The guarantee is structural, not statistical: the anytime loop starts no
evaluation past the reserve, a tiny-deadline test proves the degradation path
to argmax, and U5 measures p99 **and max** on the Modal proxy before any
rated round. Constants and the full table:
[`tactics-plan.md`](../../bots/unclejoe/tactics-plan.md) §4.

## 9. Evaluation

Per [`../../arena/decision-rule.md`](../../arena/decision-rule.md), and the
whole reason the fork exists:

- Arms: `A` = joe-rs at its registered content hash (not provisional, ≥30
  games), `B` = unclejoe. **Both in the same round** — no number is fitted
  across rounds. Both bots exist side by side, so no frozen copy is needed.
- One round through `arena.tournaments.competition` with a shared panel of ≥5
  bots including `cm_expander`, a pinned `--round-seed`, `--seat-policy
  alternate`, `--strict-versions`; ≥200 games per arm, ~1150 per arm for a
  ±25 interval.
- Quote `Δ ± SE, CI₉₅, P(B>A)`, games per arm, and the verdict — never a rank.
- **Replication in a second, separately scheduled round is mandatory.**
  Every turn now runs to the clock, so unclejoe is maximally deadline-shaped
  — exactly the class the replication rule exists for. Note host state by
  hand.
- **Decomposition on a flat result.** Three mechanisms share the headline
  contrast; the per-mechanism switches (`UNCLEJOE_OVERRIDE`,
  `UNCLEJOE_MASKS`, `UNCLEJOE_RERANK`) make follow-up arms a one-line
  `run.sh` export each — which forks the content hash into its own
  registered identity, as intended. Each decomposition contrast runs inside
  one round.
- Both arms must be on the same network. If joe re-exports mid-evaluation,
  resync both bots or neither.

Secondary, from shadow mode: trigger fire rate, search decline rate, override
count per game; and from the U4 shadow re-rank: would-change rate, the
candidate value-gap distribution, mask and refutation counts. A flat result
with zero overrides, a flat result with many overrides, and a flat result
whose value gaps sat inside noise are three different findings.

## 10. Non-goals

- **No new network, no retraining, no fine-tune.** The weights are joe's, byte
  for byte. Anything else makes the contrast about the network again.
- **No depth ≥ 2 lookahead, and no opponent move model.** The opponent's
  observation is not derivable from ours, so a mirrored policy is
  unbuildable, and anything past one ply is a guess. The pessimistic fog
  bound stays what it is — a proof device, never a predictor.
- **No hand-written evaluation.** Exact rules from RULES.md may *remove*
  candidates; only the network's own value head ever *ranks* them. The
  override plays proofs only; the re-rank never overrides a proof.
- **No strategic layer.** Expansion, gathering, and targeting stay entirely
  the network's. The castle masks veto two provably-bad build shapes and
  propose nothing.
- **No dependencies.** The crate keeps an empty `[dependencies]`; intake
  compiles one crate from source. Sim, filters, and afterstate rendering are
  integer work, and the re-rank reuses the crate's own forward.
- **No edits to joe, joe-rs, or the competition module.** The fork is the
  mechanism that makes that possible. (Mechanical, non-behavioral edits to
  unclejoe's *copied* files are allowed and listed —
  [`tactics-plan.md`](../../bots/unclejoe/tactics-plan.md) §1.)
