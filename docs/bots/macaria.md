# macaria

`bots/macaria/`. Grounded idea: **take blitz's policy, and on the turns where a
priority ladder is most likely to be locally wrong, search for a better move
and play it instead.**

Since r1 the vendored core also carries one strategic change of its own — the
[general hunt](#the-general-hunt), which decides where an unsighted enemy
general is from the belief over where it *can* be rather than from whichever
enemy tile happens to be in vision.

Spec: [`../research/strategies/macaria.md`](../research/strategies/macaria.md).
Round report:
[`../research/measurements/macaria-r1.md`](../research/measurements/macaria-r1.md).

## Layout

| File | What it is |
| --- | --- |
| `blitz_core.py` | a **copy** of `bots/blitz/agent.py` at content hash `b4a69aad6389` |
| `params.py` | every behavioural constant of both halves, in one frozen dataclass |
| `search.py` | the move search: flat paranoid maximin over a pruned move matrix |
| `agent.py` | runs the core, then offers the search what is left of the turn budget |
| `probe.py` | arena-owned per-turn introspection; never imported by the agent |

### Why the core is vendored, not imported

blitz is the A arm of every contrast that decides this bot. Editing it — even
to add a hook — moves its content hash, and the rated entity the baseline
refers to stops being the program the baseline was measured on
([decision-rule.md](../arena/decision-rule.md), *connectivity*). A copy makes
that mistake unreachable.

The copy was behaviour-identical to blitz at r1: verified move-for-move over 6
seeds against `cm_expander` (same capture turn on each:
476/254/232/570/331/179). Its deltas then were purely mechanical — every
module-level behavioural constant blitz kept (`DEATHTOUCH_TURN`,
`CHASE_DEFEND_FROM`, the never-reserve `StrategyContext`, the `or 50` wave
period, the `// 2` general regen) is now a `BlitzConfig` field, and the
module-level `_STRATEGY` singleton became `strategy_context(config)`.

The cost is honest duplication: a later fix to blitz does not reach macaria.
That is the right trade when the two are being *compared* rather than composed
— and it is also what made the general hunt below safe to add, since nothing
macaria changes can move the baseline's content hash.

## The general hunt

The defect, from 318 scraped leaderboard games
([replay analysis](../engine/replay-analysis.md)): in **97 of 127 losses the
bot never once had vision of the opponent's general**, while its gather waves
were aimed at whatever enemy tile happened to be visible. Canonically match
15932 — closest approach 8 steps, reached at tick 36 and never again, land
capped at 55 for 224 ticks while the opponent reached 144.

The cause is that upstream reads "the enemy's side of the map" off
`visible_enemy_tiles`: `enemy_anchor` takes the enemy cell farthest from us and
`contact_target` takes the fog just behind their front line. In a game we are
losing, their front line is *inside our own half*, so both point home. The
anchor drives the expansion bias and the target drives the wave, so the whole
bot orbits its own territory while the opponent walks in and finds the general.

The fix reuses the belief macaria already maintained and never read:
`BeliefState.candidates`, seeded with every cell at least
`min_general_distance` BFS steps from ours (RULES.md §01) and pruned as we look
at them.

| Where | Before | After |
| --- | --- | --- |
| `hunt_target` | — | best remaining candidate: belief pruned per visit, discounted by hops from the enemy footprint we have ever seen (the mirror of our general before contact) and, much more cheaply, by travel |
| `contact_target` | any fog behind their front | only fog the belief still allows to be a general |
| `directed_expansion_step` | fog revealed first, direction on ties | direction first while the general is unsighted |
| `required_army` | 70% of their mobile army | with the general **located**, what it takes to kill it — two army from the deathtouch turn (RULES.md §07) |

Travel is priced an order of magnitude below the direction prior on purpose.
Scored alike, the hunt degenerates into "explore the nearest unknown", which on
a board mountain-padded to 21x21 is our own empty corner — the candidate ring
sits ≥17 steps away in *every* direction, including behind us.

Knobs: `blitz_hunt_*`, `blitz_expand_bias_first`, `blitz_finish_margin` in
`params.py`. `MACARIA_TUNE='{"blitz_hunt_enabled": false,
"blitz_expand_bias_first": false}'` restores upstream targeting.

### What the arena said about it

2,624 games against `macaria_base@862ac0189a1a` (rounds `macaria-hunt-*`;
the evaluator's report is the record):
**+18.82 ± 12.41 Elo, unproven.** The premise does not transfer from
generals.bot to this panel, and the numbers say why.

| | leaderboard (scraped) | arena panel (1,008 paired games) |
| --- | --- | --- |
| games with no sighting | 76% of losses | 10% of all games, 49% of losses |
| where the old estimator pointed | our own half | median **6–8** hops from the true general |
| never-sighted games | — | land margin ≈ 0 throughout, ~330 turns |

So the bot was never lost on the wrong side of the map here. In never-sighted
games its frontier parks a median **5** hops from the enemy general and spends
35% of the game within 6 of it, and never reaches the Chebyshev-1 needed to
see it — the wall is the defended ring around a general that has been growing
since turn 0, not a failure to look. Sighting mostly *tracks* the economy:
games sighted in both arms ran a +8 → +49 land margin, never-sighted games
hovered at 0.

The hunt does move vision — net **+16/+28/+34** paired games newly sighted with
20/50/100 turns still to play — but it cost land (`land_mean` −3.20 ± 1.71,
`land_max` −4.91 ± 4.01, `strikes` −0.31 ± 0.21), and in even games land is
what decides. That is the whole result: a real information gain that the
economy pays for at par.

**`hunt_drives_anchor` (r2, default off).** Replay of the recorded
trajectories showed `strike_target` reaching the hunt rung on only **15%** of
post-opening turns in never-sighted games — `contact_target` preempts 84% —
while `enemy_anchor` consulted the hunt on *every* turn, pointing the whole
expansion bias grid at a belief cell that could be far away and behind their
front. The hunt now sets the wave's target only. Paired over 240 seeds against
six panel bots it buys back `land_mean` +1.81 ± 2.34 and `land_max`
+2.05 ± 6.29 for a flat winrate (−0.004 ± 0.048) and flat sighting (raw +3,
early −3) — a per-turn cost removed from a mechanism that was not using it.

## Library survey — verdict: nothing fits, built from first principles

Checked live PyPI metadata for `mcts` 1.0.4, `monte-carlo-tree-search` 2.1.0,
`mcts-simple` 1.1.0 and `mctspy` 0.1.1, and the sandbox's own library set
(`competition-module/competition/requirements.txt`).

Every one of those packages assumes:

| Assumption | This game |
| --- | --- |
| perfect information | fog of war (RULES.md §06) |
| **alternating turns** | simultaneous, and the resolution-order rule (chasing > reinforcing > smaller army, RULES.md §02) **is** the tactics |
| a cheap, hashable state with `get_possible_actions` / `take_action` | 92–490 legal moves per turn, so a joint branching factor of ~10⁴–10⁵ per ply |
| a budget in seconds, or whole playouts to terminal | tens of milliseconds, against a 150 ms cap (RULES.md §08) |

None of them ship in the sandbox, and matches have no network — nor does
`build.sh` — so each would additionally have to be vendored into the
submission zip. What they would contribute is ~50 lines of control flow, and
it is the wrong control flow. What actually costs something here is the
forward model, which has to be hand-built against `generals/core/game.py` and
`generals/modifiers/deathtouch.py` either way.

`numba` **is** in the sandbox and could multiply the rollout rate ~10×, but its
first-call JIT costs seconds against a 150 ms/50-fault rule. Noted as a future
option; not taken.

## The search

**Flat paranoid maximin over a pruned move matrix, with iterative deepening —
not UCT.** At ~50–150 µs per rollout and a budget that ranges from ~55 ms down
to nothing, a tree would give each root child a few dozen visits and each
grandchild almost none, which is below where UCB's bookkeeping pays for
itself.

- **Rows** — up to 10 of our candidate moves. The core's own move is always
  row 0 and `PASS` always survives the cap.
- **Columns** — up to 6 opponent replies per row, including two row-specific
  dangers: a chase onto our move's *source*, and an attack on its
  *destination*.
- Each cell is **one deterministic rollout**. Row score is the min over its
  replies (paranoid: the opponent is assumed to see our move); the decision is
  the argmax over rows, and it must beat the core's own row by
  `mcts_override_margin` to be played.
- Determinism means zero variance, so every rollout buys a distinct
  hypothesis. The deepest **completed** deepening pass decides — a
  half-sampled matrix must never outvote the core.

### What the forward model does and does not know

It mirrors the engine step for step: validity re-judged at execution time (so
a chase that captures an attacker's source really does cancel the attack),
the full move-order rule including the seat tie, strictly-more combat with
ties to the defender, deathtouch from turn 800, **mutual capture scored as the
draw it is**, and `time += 1` *then* growth at the new time.

It does **not** model fog army, castle builds, or the opponent's real policy.
Each omission is bounded at a ≤10-ply horizon and named in `search.py`'s
docstring. The one thing it fabricates is the core's own fabrication: a
once-seen, currently-fogged enemy general carries the core's `1 + turn/2`
garrison estimate, because without it the finishing race the core is running
could not be scored at all.

### When it fires, and what it never overrides

Scoped to tactical contact: the finish window after a sighting, an enemy stack
near our general, the endgame from turn 780, and contested contact around the
core's own move **in assault phase only**. It never overrides a checked
winning capture, and never overrides the §07 chase defence.

## Knobs

All of them are in `params.py`, including the ones blitz keeps as module
constants. `MACARIA_TUNE` (JSON, environment) overrides fields at
construction, so a sweep varies one knob without forking a content hash per
cell; malformed, non-dict and unknown-key input all fall back to the shipped
defaults rather than raising, because §08 forfeits a bot that crashes.

The control arm for the search is:

```bash
MACARIA_TUNE='{"mcts_enabled": false}'
```

which is macaria with the search off — i.e. the vendored core alone.

## The turn budget

RULES.md §08 allows 150 ms and forfeits at 50 late replies. macaria spends at
most 100 ms on the whole move and keeps 50 ms in reserve. The core runs first
and is not interruptible, so the search's deadline is what is left of the cap
after the core has been paid — a slow core shrinks the search instead of
pushing the turn over. The deadline is absolute and is checked **inside the
rollout loop**, once per simulated ply.

Note that the local `matchup.py` does not enforce §08's timeout — it blocks on
`readline` — so a clean local match proves nothing about latency. Latency is
measured directly, per move, through the probe channel.

## Verification

```bash
source .venv/bin/activate
python competition-module/competition/matchup.py \
  bots/macaria/run.sh \
  bots/cm_expander/run.sh \
  --mode competition --seed 0
```
