# unclejoe — joe-rs's network, with an exact search allowed to overrule it

Spec, written before the tactics code. Bot: `bots/unclejoe/`. Substrate and
milestone U1: [`../../bots/unclejoe/fork-plan.md`](../../bots/unclejoe/fork-plan.md).
Module layout and milestones U2–U5:
[`../../bots/unclejoe/tactics-plan.md`](../../bots/unclejoe/tactics-plan.md).

Rule references: [`RULES.md`](../../../RULES.md) sections 02 (move order),
04 (growth), 05 (combat), 06 (visibility), 07 (deathtouch, draw), 08 (150 ms
per move).

## 1. The claim

`bots/joe-rs` plays the frozen joe network greedily: one forward pass, argmax
over the action logits, no search. A network trained on self-play answers
*every* position with the same amortized guess, including the small class of
positions where the right move is not a guess at all — where a handful of
plies of exact simulation decides the game. unclejoe keeps that policy
**unchanged** as the default move source and adds one thing: on the turns
where an exact answer is cheap and reachable, compute it and play it instead.

The hypothesis is narrow and falsifiable:

> A greedy network policy loses measurable strength at the moments where the
> correct move is provable — a forced general capture within a few plies, and
> the reply that stops one — and a bounded exact search that fires only there
> recovers part of it, without touching the policy elsewhere.

If that is wrong, the failure is specific rather than mysterious: unclejoe and
joe-rs come out flat while the shadow-mode counters show the triggers firing.
That is a different result from "the search never ran", which is why U2 exists
as its own milestone and reports fire rates before any override is live.

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

Every turn, in this order:

1. update per-game memory from the frame (enemy general location, remembered
   mountains, last-seen general army);
2. run the full existing pipeline — frame → raw tensor, masks, augmented
   observation, one forward pass, argmax — unchanged and unconditional;
3. evaluate two cheap trigger predicates;
4. if one fires, run a bounded exact search for a *proof*; play its move only
   if it proved one;
5. otherwise emit the network's move.

The forward pass always runs even when a tactic overrides it. Skipping it
would fork the state pipeline into two paths, one of them rarely exercised,
and the pass is already inside the latency budget.

## 4. Triggers (exact, and a deliberate superset)

Both are integer scans over the parsed frame plus memory. A trigger that fires
on a dead position costs one search that declines; a trigger that misses a
live one costs the whole point of the bot. So both are loose on purpose.
`D` is the search's depth budget in our moves (proposed 3, §5).

**Kill.** The enemy general has been seen (its cell is remembered — a general
never moves), and some cell we own with `army > 1` lies within Manhattan
distance `D` of it, and either

- `obs.turn ≥ 800` — deathtouch is live, so any unit is potentially lethal
  (RULES.md §07); or
- our total army within that neighborhood exceeds the general's **last seen**
  army — a stale lower bound, which keeps the superset property while the
  search applies the honest pessimistic bound before it proves anything.

Before the general has been seen the trigger cannot fire. That is correct, not
a gap: no exact kill is provable against an unlocated general.

**Immediate defense.** With `g` our general (always known), fire if

- any *visible* enemy cell within `D` of `g` holds `army ≥ army(g)` — loose:
  it ignores the leave-one-behind rule and multi-step attrition; or
- `obs.turn ≥ 800` and any visible enemy cell with `army > 1` is within `D`; or
- a fogged cell lies within `D` of `g` **and** the hidden-army budget (§6)
  reaches `army(g)` — a stack could be sitting one step outside vision.

Vision is the 3×3 pool around owned tiles (RULES.md §06), so every cell
adjacent to our general is always visible: the fog arm only matters at
`D ≥ 2`.

Expected fire rate is low — most turns have nothing within three tiles of
either general. U2 measures it rather than assuming it.

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
| Depth | 3 of our moves | Beyond three plies the fog bound (§6) dominates and proofs stop landing. |
| Node budget | 100,000 | A hard ceiling that does not depend on the clock. |
| Deadline | 20 ms, checked every 1,024 nodes | The real guard: wall clock, not an estimate. |

Hitting either cap declines, and a decline is a pass-through, never a
half-searched move.

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

## 7. Latency budget

joe-rs's measured full-path cost is **p99 23.7 ms, max 34.5 ms** on the Modal
one-core x86-64-v3 proxy ([`../../bots/joe-rs/latency.md`](../../bots/joe-rs/latency.md),
2026-08-15). unclejoe's worst turn is that, plus two integer scans over at
most 441 cells (microseconds), plus the search's 20 ms deadline: **≈ 55 ms
against the 150 ms limit** (RULES.md §08). The deadline is enforced by the
clock, so the bound does not depend on the node estimate holding.

U4 proves it rather than arguing it: `unclejoe bench` over the 1,572-turn
synthetic-long log, locally and on the Modal proxy, plus a constructed
worst-case position as a Rust test that asserts the search returns inside the
node cap.

## 8. Evaluation

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
- **Replicate in a second, separately scheduled round before the result is
  written down.** unclejoe is mildly deadline-shaped (the 20 ms cap), which is
  the class the replication rule binds hardest on.
- Both arms must be on the same network. If joe re-exports mid-evaluation,
  resync both bots or neither.

Secondary, from shadow mode and the probe: trigger fire rate, search
decline rate, and override count per game. A flat result with zero overrides
and a flat result with many overrides are different findings.

## 9. Non-goals

- **No new network, no retraining, no fine-tune.** The weights are joe's, byte
  for byte. Anything else makes the contrast about the network again.
- **No general search.** The search fires only where a *proof* is reachable;
  it is not an evaluation-based tree search over the whole game, and it never
  plays a "best guess" move.
- **No heuristic override.** If the search cannot prove the outcome, the
  network's move stands, even when the tactic looks obviously right.
- **No strategic layer.** Expansion, castle building, gathering, and targeting
  stay entirely the network's.
- **No dependencies.** The crate keeps an empty `[dependencies]`; intake
  compiles one crate from source, and the search is integer work.
- **No edits to joe, joe-rs, or the competition module.** The fork is the
  mechanism that makes that possible.
