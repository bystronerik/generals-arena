# Strategy spec — proteus

Bot: `bots/proteus/`. Migrated from the generals-bot repo (source name:
adaptive/Proteus). Grounded idea: **classify the opponent, switch strategy
cores with hysteresis.** Composes the cores of [`blitz`](blitz.md) and
[`boom`](boom.md); baselines for comparison are those two pure bots and
`phase_switch` (the roster's clock-driven switcher).

Proteus is exactly one of its cores at any instant, so its measurement
target is the **worst-case** matchup, not the best-case one.

## 1. Architecture

- **One shared `OpponentModel`** — `Agent.act` updates it once per turn;
  both cores were constructed around the same instance, and each core's
  `observe()` is idempotent per turn, so warming never double-counts.
- **`HomePressure`** (`signals.py`) — proteus-owned, because it joins two
  facts the shared model keeps apart (§3).
- **Classifier** (`classifier.py`) — pure function of the model, the turn
  and the pressure latch. Labels: aggressor, economy, unknown.
- **Switcher** (`switcher.py`) — debounced, asymmetric selection (§4).
- **Warm handover** — the inactive core's `observe(obs)` runs every turn so
  its beliefs, latches and threat memories are current at switch time.

## 2. Why two cores and not four

Every previous version constructed and warmed four cores while `COUNTER`
could select only two; `metro` and `aegis` were unreachable, and the
switcher's `defense_streak` / `leave_defense_streak` pair was dead with
them (the unit test had to monkeypatch `COUNTER` to exercise it).

A 32-game-per-cell grid of every pure core against every roster bot, seat
alternated, says the unreachable cores should stay unreachable:

| core | pooled over 12 contested opponents | uniquely best against |
| --- | --- | --- |
| boom | **0.816** | aegis, army_convey, late_rush, metro, fog_scout |
| blitz | 0.788 | blitz, boom, cm_hunter |
| aegis | 0.603 | *nothing* |
| metro | 0.596 | *nothing* |

Aegis is actively harmful in two cells: 1-0-31 against `garrison`, where two
turtles simply run out the [`RULES.md`](../../../RULES.md) §07 draw, and
0.14 against `boom`. Adding labels that route to either core would widen the
repertoire without widening the set of *good* answers.

The same grid showed the old proteus was byte-identical to pure blitz in 6-8
games of 8 against every hard opponent — the switching machinery bought
nothing measurable (proteus 0.797, pure blitz 0.802, pure boom 0.839 over
the same 12 opponents on identical seeds).

## 3. The one signal that changes the answer

`OpponentModel` keeps a running **minimum** enemy distance and a running
**maximum** enemy stack, latched over the whole game and independently. The
old classifier's `deep_incursion` test —
`closest_enemy_dist <= 8 and biggest_enemy_stack >= 15` — therefore fired on
a pair of facts that need never have been true at the same time: a border
cell nibbled at distance 8 on turn 90, and an unrelated 20-stack seen across
the map on turn 300, together read as "a big stack came to our door".

`HomePressure` latches the **conjunction**: the largest stack ever seen
*while that stack stood within 8 BFS steps of our general*, and the turn it
first reached 15. That is the mechanical difference between an opponent
whose expansion happened to reach us and one that walked a sized fist at our
general — and it is the only distinction measured to change which core wins.

**When** the fist arrives is as diagnostic as whether it did:

| opponent | first fist at our door (median turn) | what it is |
| --- | --- | --- |
| blitz | 169 | the rush |
| cm_hunter | 181 | the hunt |
| late_rush | 278 | the timed commit |
| metro | 378 | a pressure wave |
| aegis | 411 | the counterattack window |

Hence `DUEL_DEADLINE = 250`. Without it aegis latches as an aggressor in 8
games of 12 and takes blitz (0.75) instead of boom (0.86).

### What is deliberately *not* used

`structure_estimate` recovers the opponent's structure count — general plus
castles — from the exact `opp_army` aggregate alone, through
[`RULES.md`](../../../RULES.md) §04: production is the only thing that
raises total army (moving onto neutral plain conserves it per §02, combat
only lowers it), and it fires every *other* turn, so a high quantile of the
per-turn delta with the every-50 land-bonus turns dropped reads the count
straight off. It is fog-proof — it counts castles that were never seen — and
it separates cleanly (no castles estimates 0-1, two or three estimates 2-3).

It is **probed, not consumed**: both sides of that split want boom, so
acting on it would be motion without effect. It is recorded per turn so the
claim stays falsifiable.

## 4. The counter map and the hysteresis

| Classified as | Play | Why |
| --- | --- | --- |
| aggressor | blitz | They walk fists at generals. Race them; do not bank. |
| economy | boom | Castle programmes, slow expanders, turtles and timed committers alike — the grid gives boom the edge on every one. |
| unknown | default (blitz) | Not enough evidence to leave the spine. |

`aggressor` and `unknown` share blitz by design, not oversight: blitz is the
spine, so "they will come at us" and "we do not know yet" are the same
instruction — stay put.

**The spine stays blitz.** Always-boom wins the mean (0.815 against 0.787)
and loses the floor (0.36 against 0.50), and proteus is bought for its
floor. Mid-game switches *into* blitz also cannot replay its opening
(`opening_end = 50`), and a blitz that never acted holds no chain and no
strike stack, so it restarts from rebuild.

**Hysteresis is asymmetric on the pair that matters** rather than on the
dead one. Leaving the spine takes `leave_spine_streak` turns at confidence
≥ `min_confidence`; returning takes `return_spine_streak = 6` and ignores
the cooldown. The asymmetry is priced: being boom against a real aggressor
is the worst cell on the grid (0.36), while being blitz against an economy
costs at most 0.16 — so the cheap error is the one to make.

## 5. The unresolved tension

The evidence that would stop a wrong switch arrives *after* the switch must
be made to be worth making. Measured: proteus leaves for boom at turn ~162,
and a blitz opponent's fist reaches the door at turn ~226. Widening
detection cannot fix this, because the clusters are not separable earlier —
at turn 150, `blitz`, `boom`, `army_convey`, `fog_scout` and `classic_duel`
agree to within noise on every observable tried (land rate 0.48-0.53, army
per land 3.5-3.7, contact turn 78-88, structures ≤1, cells we lost 0-2.5).

That collapse is **fine**, because within that cluster the counter barely
matters: playing boom against all five scores 0.674 and playing blitz
against all five scores 0.662. The errors cancel. What is *not* fine is the
tail — `boom` against `blitz` is 0.36 — which is why the exit from the spine
is timed against the fist window rather than against confidence alone.

## 6. Opponent clusters (measured, not asserted)

Clustered by what proteus can observe through fog, not by bot identity.

| cluster | members | signature | counter |
| --- | --- | --- | --- |
| castle economy | aegis, castle_rush, metro, phase_switch | structures ≥2 by turn ~145 | boom |
| fast / aggressive | blitz, boom, army_convey, fog_scout, classic_duel | contact 78-88, land rate ~0.5, army/land ~3.6 | boom (blitz on a latched early fist) |
| hunter | cm_hunter ≡ cm_harvester | early contact with a *low* land rate | blitz |
| slow land | expand_plus, splitter, smoke, cm_random, cm_expander, choke_control, general_hunter, castle_builder | late contact, land rate ≤0.15 | either (both ≈1.00) |
| timed committer | late_rush | no castles, fist after turn ~625 | boom |
| pure turtle | garrison | land rate 0.00, army per land 40 | anything but aegis |

**Behaviourally indistinguishable, and that is fine:**

- `cm_hunter` ≡ `cm_harvester` — every measured number is identical.
  Harvester's only addition is banking *neutral* castles, and competition
  maps have none (`strip_neutral_castles`), so it *is* Hunter. Same
  behaviour, same counter.
- The slow-land cluster is eight bots that both counters beat ≈1.00. A
  distinction that cannot change a decision is not worth detecting.
- The fast/aggressive cluster does not separate at all (§5).

Identifying the specific bot is a non-goal.

## 7. Diversity check

vs `phase_switch` (clock-driven phases): proteus switches on *opponent
evidence* with hysteresis, not on the turn number, and switches between two
whole strategies rather than phases of one. vs the two pure cores: proteus
is exactly one of them at any instant, so its value is picking which.

## 8. Experiment

Verdict comes from the pairwise contrast in
[`../../arena/decision-rule.md`](../../arena/decision-rule.md), never from a
rank. Baseline is proteus HEAD before this change; candidate is after.

Verdict for this rework: **no change (proven flat)**, `delta = -3.37 +- 9.89`,
`CI [-22.76, +16.01]`, `P(B>A) = 0.367` over 20,376 pooled games. It ships on
simplicity and instrumentation, not strength. See
[`017-proteus-detection-rework.md`](../experiments/017-proteus-detection-rework.md)
and [`016-proteus-adaptive-switching.md`](../experiments/016-proteus-adaptive-switching.md).
