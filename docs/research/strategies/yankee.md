# Strategy spec — yankee

Bot: `bots/yankee/`. A fork of [`proteus`](proteus.md) at content hash
`7ae237e99d34`, tuned, given a scoped Monte Carlo tree search and a third
strategy core for the [`RULES.md`](../../../RULES.md) §07 endgame.

`bots/proteus` is unchanged and stays the A arm of every contrast below.
Verdicts come from the pairwise contrast in
[`../../arena/decision-rule.md`](../../arena/decision-rule.md), never from a
rank or a bare winrate.

## 1. What the fork owns

yankee vendors the two strategy cores rather than importing them:

| file | upstream | why |
| --- | --- | --- |
| `blitz_core.py` | `bots/blitz/agent.py` | in proteus's closure — editing it moves the baseline |
| `boom_core.py` | `bots/boom/agent.py` | same |
| `params.py` | — | every knob this effort moves, in one frozen dataclass |
| `search.py` | — | the MCTS (§4) |
| `deathtouch.py` | — | the §07 endgame core (§5) |
| `classifier.py` / `switcher.py` / `signals.py` | proteus | unchanged logic, params threaded through |

After the fork yankee's source closure contains nothing another rated bot
depends on except `bots/_common/`, which yankee does not edit. That is the
property the whole comparison rests on: nothing yankee does can move proteus's,
blitz's or boom's content hash
([`../../arena/decision-rule.md`](../../arena/decision-rule.md),
"connectivity").

The cost of vendoring is that upstream improvements to blitz and boom no longer
propagate. Accepted — both are rated roster entities, not libraries.

## 2. The baseline, re-measured

The brief's baseline (46-4 over 50 games, all four losses at turns 202-314, all
four with cm_hunter in seat A) comes from `round5`, which played proteus
`91d2a88f8316`. The lineage head is `7ae237e99d34`, two steps later. Re-measured
on the head, seat-alternated, 300 games:

| | games | W | L | D | winrate |
| --- | ---: | ---: | ---: | ---: | ---: |
| proteus vs cm_hunter | 300 | 278 | 22 | 0 | **92.7%** |

The pooled winrate reproduces. **The description of the losses does not**, and
both corrections matter for where to spend effort.

**The losses are early, not mid-game.** Turn at which each of the 22 losses
ended:

```
69  72  96  97  98  99 100 122 122 127 139 152 165 187
258 280 310 315 330 357 406 630
```

13 of 22 land before turn 160; the round5 sample's 202-314 band holds 4. With
n=4 that band was the whole sample, which is why the brief points there; with
n=22 it is the tail.

**The seat split is not established, and what signal there is points the other
way.** 15 of the 22 losses came with yankee in seat A, 7 in seat B — 90.0%
(135-15) versus 95.3% (143-7). A two-sided binomial on 15/22 gives p ≈ 0.13, so
this is not a seat effect; it is 22 games. It is reported because the brief
asked for it explicitly, and because it points *opposite* to round5's 0-of-4.

The engine does have a real seat asymmetry — `game._determine_move_order` gives
a full §02 tie (same chase status, same reinforce status, equal source armies)
to player 0 — but nothing measured here separates its effect from noise.

## 3. Why the losses happen

Five of the early losses were replayed from the trajectory and read the same
way. Taking seed 1156835504 (loss at turn 97) as the type specimen, with our
general at (15,6):

| turn | our general | our stacks (army @ distance) | their fist |
| ---: | ---: | --- | --- |
| 93 | 7 | 11@7, 11@5, 11@7 | 23 @ 4 |
| 94 | 8 | 11@7, 11@4, 11@7 | 21 @ 3 |
| 95 | 8 | 11@3, 11@7, 11@7 | 19 @ 2 |
| 96 | 9 | 11@2, 11@7, 11@7 | 17 @ 1 |
| 97 | — | — | captured |

Nothing is mispriced. blitz's `home_threat` correctly reports a deficit, its
`defense_move` correctly walks the nearest stack home, and that stack correctly
arrives — two turns after the general falls. The army was in the wrong place
*before the threat was observable*, and blitz has no mechanism that could have
put it anywhere else: it empties its home by design and its defence is purely
reactive.

The other four replays differ only in detail. In seed 253385209 the general
held 20 army on turn 96 and the core walked that stack *out* on turn 97, into a
10-stack standing two cells away.

## 4. Increments

Each is a separate registered step with its own contrast, so a regression is
attributable ([`../experiment-protocol.md`](../experiment-protocol.md)).

### 4.1 Parameter tuning — null, and the null is the finding

A 200-game paired screen (identical map seeds and seats for every cell) over
the knobs the brief names, against cm_hunter:

| config | winrate | paired flips vs base |
| --- | ---: | --- |
| base | 0.915 | — |
| `defense_dist` 16 | 0.915 | +1 / −1 |
| `defense_dist` 21 | 0.910 | +1 / −2 |
| `defense_margin` 6 | **0.925** | +10 / −8 |
| `defense_margin` 12 | 0.905 | +11 / −13 |
| `min_strike_floor` 26 | 0.900 | +7 / −10 |
| `opening_end` 65 | 0.880 | +7 / −14 |
| `rally_ticks` 8 | 0.905 | +5 / −7 |
| `rally_ticks` 22 | 0.920 | +4 / −3 |
| `strike_ratio` 1.0 | 0.915 | +0 / −0 |

The best cell is +2 games in 200 with a paired net of +2; a sign test on
+10/−8 is p ≈ 0.8. Nothing here is a result.

The switcher and classifier knobs (`min_confidence`, `leave_spine_streak`,
`return_spine_streak`, `cooldown`, `DUEL_DEADLINE`, the `HomePressure`
thresholds) cannot be screened against cm_hunter at all, and this is structural
rather than a gap in the sweep: cm_hunter classifies as `aggressor`, and
`COUNTER[aggressor]` and `COUNTER[unknown]` are both `blitz`, so every value of
every switching knob plays the identical game. They are screened against the
economy cluster instead (§6).

**Kept at proteus's values.** The screen says the reachable knobs do not reach
the failure, which is what justified changing core logic instead — the brief's
"where measurement justifies it".

### 4.2 The standing guard

`blitz_core.home_guard` / `guard_move`. One idea: size a garrison against an
assault we cannot see yet.

`opp_army - opp_land` is the opponent's army minus the one unit §02 pins on
each of its cells — the part that can be concentrated into a fist. Both terms
are scoreboard scalars the engine sends every turn, and §06 hides *cells*, not
the score, so this is **fog-proof**. The guard keeps
`guard_ratio × (opp_army − opp_land)` within one move of our general
(the general's own stack plus its owned neighbours), capped, only between turns
`guard_from` and `guard_until`, and only once `HomePressure` has latched a real
fist — against the expander cluster it would otherwise be pure cost.

It sits *below* `defense_move` in the ladder: once a threat is visible, killing
or covering it beats topping up a number.

### 4.3 MCTS — `search.py`

Full design, fidelity table and belief model are in that file's header. The
load-bearing points:

**Scope.** The search runs only where being wrong is already the alternative:
an enemy stack within `mcts_window` BFS steps of our general *that could
actually take it* (`army − 1 > general army`), a §07 touch threat past turn
800, or a kill we can actually land on a known enemy general. Everywhere else
the core's move is returned untouched and the search never starts. Measured
firing rate on a full game: 1-3 turns in ~390, against 175 in 580 for the first
version.

That narrowness is measured, not stylistic. The first version fired on any
enemy stack in the window and **cost 47 points of winrate** — 0.917 → 0.450
over 200 paired games, and widening the window to 12 took it to 0.183. A
monotone dose-response like that identifies the search as the cause. Two things
were wrong:

1. *Overriding a stateful core desynchronises it.* blitz and boom write
   `mem.stack`, `mem.chain_head` and `mem.chain_visited` as a side effect of
   choosing. Replace the move and the core plans its next turn from a board
   that never happened.
2. *A 12-ply horizon cannot price giving up the attack.* Interrupting a wave is
   free inside the window and costs the game 300 turns later. Mean game length
   went 399 → 683 turns, which is what "stopped attacking" looks like from
   outside.

The evaluation was also a posture-reward: it scored army standing on our own
general at half the total weight, so every root move that carried army home
scored well and every attacking move scored badly. It now scores only facts —
army share, land share, and whether a stack our general cannot stop is still
alive.

**Time.** 40 ms cap on the search, checked inside the rollout loop once per
simulated ply, not merely between iterations. The cap is
`min(40 ms, 110 ms − time already spent this turn)`, so a slow core shrinks the
search rather than pushing the move past §08's 150 ms. On expiry the best root
child so far is returned, and root child 0 is *always* the core's own move — a
search that learns nothing returns the heuristic by construction. Rollouts that
are interrupted return no value at all rather than a partial one, which would
bias whichever child was unlucky.

**Fog.** The belief is the observation taken literally: visible cells exact,
**fog assumed empty and neutral**. The wire protocol reports army 0 outside
vision, and the search invents no distribution over what is there — so an enemy
stack one step outside our vision does not exist as far as the search is
concerned, and the search *systematically underestimates incoming force*. This
is the least defensible thing in the design and it is why the search is scoped
to the defensive window, where the threatening stack is visible by construction.

Sampling positions for the opponent's off-screen army from `OpponentModel` was
considered and rejected: that model tracks aggregates and a latched
nearest-approach, not a spatial distribution, so any sample would be a prior
dressed as evidence.

**§02 fidelity.** `_Sim._order` implements chasing > reinforcing > smaller
army, with the full tie going to player 0 — which is why the search is
constructed with the real seat and not the perspective-relative owner code.
§05 combat, §04 growth and §07 deathtouch are modelled exactly. Castle builds
are **not** modelled, in either direction; against a castle programme the
search's opponent model is wrong.

### 4.4 The deathtouch core — `deathtouch.py`

Deathtouch is not new. `StrategyConfig.deathtouch_turn = 800` and a
`DEATHTOUCH_TURN = 800` constant already exist in blitz, aegis, metro,
garrison, late_rush and splitter, and all six use it identically: one extra
clause in a finishing check. This is a **third switcher core**, selected on the
clock from turn 800, because from that turn three consequences of §07
contradict what a pre-800 core is optimising:

1. **Army on our own general buys nothing.** One unit is lethal, so a 60-army
   general dies exactly as fast as a 2-army one.
2. **The only defence is a chase, from a third tile.** Capturing the attack's
   source cancels the touch, because the engine re-tests `owns_source` at
   execution. Reducing the source is not enough — the attacker moves
   all-but-one, so a source left holding 2 still sends 1 and still touches; the
   chase must *take* the cell. And the counter cannot come from the general:
   that is a mutual chase, §02 falls through to smaller-army-first, the general
   moves first, is too small to strip the source, and the touch lands anyway.
   A tile that can chase a threat standing next to our general is adjacent to
   it and is not the general — so **the endgame garrison belongs at distance 2,
   not at home.** That is the positional content of the rule, and it is why
   this is a core and not a clause.
3. **Reach beats size.** A 2-stack that arrives wins. But a 2-stack moves one
   unit and §05 needs strictly more than the defender, so it can cross our own
   land and empty neutral land and nothing else. Routing a touch is a path
   problem over friendly ground — the opposite of blitz's wave cycle and boom's
   strike fist.

**What this cannot buy.** Against cm_hunter it is close to irrelevant: 1 game
in 50 of the round5 sample reached turn 800, and the median win in the 300-game
baseline came at turn ~384. Its value is the draw-heavy matchups — `fog_scout`
(34.0% draws), `army_convey` (15.8%), `metro` (13.0%), `aegis` (11.8%) — and it
is measured there, as a draw-conversion result. It is not evidence about the
cm_hunter target.

## 5. Results

See [`../measurements/yankee-r1.md`](../measurements/yankee-r1.md).

## 6. Diversity check

vs `proteus`: same classifier and switcher; yankee adds a search layer, a
predictive home guard and a third core. vs `phase_switch`: yankee's core
selection is opponent-evidence-driven with hysteresis, and only the deathtouch
core is clock-driven — deliberately, because §07 *is* a clock rule. vs the pure
cores: yankee is exactly one of three at any instant.
