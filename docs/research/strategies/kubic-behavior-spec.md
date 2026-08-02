# Kubic — merged behavior specification

Canonical reverse-engineered reading of `competition-replays/Kubic/` under
competition rules (`RULES.md`). This is observed behavior, not a bot to ship.

Merges
[`fable-kubic-behavior-spec.md`](fable-kubic-behavior-spec.md) and
[`grok-kubic-behavior-spec.md`](grok-kubic-behavior-spec.md).
Where the sources conflict, this document states the merge rule and cites both.

**Merge decisions (locked).**

1. **Defense is present but rare.** Default path is feed-forward gather/wave
   with scheduled general drain. A recall branch fires only when a visible
   (or known) enemy tile is close to the home general. Most wins never create
   that conflict.
2. **Prefer exact action reconstruction** (`scripts/fable-analyze/`) for move
   rates. Prefer **spend-detector** castles over raw EventLog
   `castle_built` (EventLog tick ~10 is a false positive).
3. **Keep the mod-50 gather/wave clock** as the main mid-game scheduler.
   Keep grok contact / sight / tip statistics as phase overlays.
4. Do **not** average unrecovered scorers. Keep both best local rules and mark
   the residual UNKNOWN.

Evidence tags: **MEASURED**, **INFERRED**, **UNKNOWN**, **MERGED** (explicit
choice across sources). Percentages are fit-set unless marked `holdout`.

**Do not** write these games into `data/games/`, `data/ratings/`, or
`data/remote_games/`.

---

## Corpus and provenance

| Item | Value |
| --- | --- |
| Seat-resolved played | **378 win / 11 lose / 1 draw**, plus **6 forfeits** (`total_ticks <= 1`) |
| Outcome source | Replay `winner` / `Replay.outcome` (name → seat), **never** the scraper folder |
| Split | Sorted by numeric `match_id`; every 10th → holdout |
| Fable fit / holdout | 351 games (340 W / 10 L / 1 D) / 39 (38 W / 1 L) |
| Grok fit / holdout | **341 / 37 wins** only; losses and the draw are failure skim only |

Manifests:

- `docs/research/measurements/fable-kubic-split.json`
- `docs/research/measurements/grok-kubic-corpus-split.json`

**Action recovery.** Fable forward-simulates candidate action pairs until the
next frame matches bit-for-bit: **4 unresolved ticks / 92,160** (0.004%),
~0.5% ambiguous. Grok frame-diff inference leaves ~30% of ticks
`multi`/`unknown`. Use fable actions for rates that need source/destination
identity.

Tick index: this spec uses **1-based** ticks (first forced passes = t=1,2;
first move = t=3). Grok scripts often use 0-based frames (passes on `{0,1}`);
the two early passes are the same events.

---

## 1. Architecture

Kubic is a **single-task conveyor on a 50-tick economic clock**, with a
**rare defensive recall** that overrides the clock when home is under
imminent threat (**MERGED**).

- Every tick with a legal move, it moves (voluntary passes ≈ 0 outside lag
  sessions).
- Exactly one stack task at a time: a chain of all-but-one moves where each
  move starts on the previous destination when that cell still has army ≥ 2.
- On a mod-50 schedule it alternates **gather** (collect through own
  territory; drain general and castles to 1) and **wave** (roll the stack
  outward along a near-shortest path toward the current objective).
- Attacks are mostly a side effect of routing toward the objective.
- Castles are a mid-game economy move (~one per 50-tick cycle in contested
  games), placed near min price on the frontier.
- Defense does **not** garrison. The general still drains on schedule in the
  common case. Recall only redirects the tip home when an enemy tile is close.

---

## 2. Decision procedure (priority order)

```
each tick t (state = frame before the action):

  1. if can capture enemy general this tick:          # deathtouch / lethal
       FULL move killer → enemy general               # UNDER-SAMPLED (n≈1
                                                      # deathtouch game; still
                                                      # top priority when legal)

  2. if no owned cell has army >= 2 and a passable orthogonal neighbor:
       PASS                                           # only routine pass

  3. DEFENSE (rare):                                  # MERGED present-but-rare
       d_home = min Manhattan(visible_or_known enemy tile, own general)
       if d_home is not None and d_home <= RECALL_PROX_D:   # D UNKNOWN; see §7
         if tip distance to own general > 2:
           FULL move tip toward own general           # latency median ~5
         else reinforce threatened cells
         # skip gather/wave this tick

  4. if t <= ~50: run OPENING script (§3)

  5. if BUILD conditions hold (§5): BUILD castle      # ~0.25% of actions;
                                                      # never in first 50

  6. if enemy general remembered (post-sight):
       if tip army < TIP_AT_SIGHT_FLOOR (~10; operating ~23):
         gather into tip (full leave-1)
       else FULL move tip toward remembered general   # toward_frac ~0.93
       # no new castle projects after sight (INFERRED)

  7. otherwise MOVE by phase (PHASE = t % 50), one chain:
       source:
         a. head of active chain (prev dst) if army >= 2   # ~78% overall;
            ~86% after own/enemy-target moves
         b. else new chain: usually general (leave-1,
            prefer odd ticks) or a fresh interior stack;
            resume abandoned chain only ~7%
       destination:
         wave  (PHASE in [28,49] ∪ [0,9]): step along route to OBJECTIVE;
            capture neutrals/enemies on-route; attack when moved > defender
         gather (PHASE in [10,27]): step through own territory toward
            muster; drain castles passed (leave 1)
       split: always all-but-one (~99%)

  OBJECTIVE:
    before enemy contact: fog-frontier target away from own general
       (best local rules 83–92%; exact scorer UNKNOWN — § open Q)
    after contact, before general sight: believed enemy-general direction
       (pre-sight enemy captures reduce true distance in ~91–92% vs
       64–76% baseline; mechanism UNKNOWN)
    after sight: remembered general cell; near-BFS march (median path
       overhead 11–20%); first wave kills ~85%; zero passes in last 15
       ticks of every resolvable win
```

Contact overlay (does not replace the mod-50 clock): first orthogonal enemy
contact median tick **82**; army ratio at contact median **~1.05** (gate ≥ 1.0
holdout-confirmed on wins); next 50 ticks still ~**63%** neutral gains — do
not hard-pivot off expansion.

---

## 3. Opening (t = 1 to ~50) — MEASURED

| ticks | behavior |
| --- | --- |
| 1–2 | forced pass (general at 1; +1 lands so t=3 is first legal move) |
| 3 | first move: general → neutral, moved=1 (~98.5%; holdout 38/39 clean) |
| 3–13 | trickle on production cycle (t=3 ~100%, t=6 ~95%, t=9 ~75%); land ≈ 4 by t=10 |
| ~14–26 | staging: 1-unit sends every other tick down a short corridor onto a staging tip; near-zero net captures |
| ~27 (mode; ±1 in ~94%) | big run from tip (army mode ~7–10) snakes ~1 neutral/tick |
| ~37–50 | wave train; capture rate high; general re-launches at decreasing army |
| 50 | **land = 24** (IQR ~23–25; holdout band 20–25 also holds); army ≈ **50** |

Spawn-independent for land@50 (corner/edge/interior, board sizes) — MEASURED
(fable). Land stays in own half through t=50 (enemy-half frac median 0) —
MEASURED (grok). **No real castle in first 50** (spend drop ≥30); EventLog
`castle_built`@~10 is **false positive**.

**First-step direction (UNRESOLVED scorer).** Fable: shortest known path to
nearest unseen cell in 92.5% of multi-option decisions. Grok: open neighbour
minimizing Manhattan to **map center** (hit 0.909). Treat as one unrecovered
planner; do not average. Relay moves repeat the source cell’s previous
out-direction (~96–99%) and ignore adjacent capturable neutrals in ~92% of
relay ticks.

**Staging tip distance (soft disagreement).** Fable: Manhattan **2** from
general in 233/341. Grok: median **1** during pulse (tip often on general).
Use “tip stays near general through the pulse, then floods from ~27.”

---

## 4. Economy: mod-50 cycle — MEASURED (fable primary)

- Gather residues ~[10,27]: own-move share peaks 0.86–0.90 at residues 14–18.
- Wave residues [28,49] ∪ [0,9]: capture/attack share peaks 40–49;
  exposure-normalized neutral capture 0.538/tick vs 0.122 at 10–19 (~4.4×).
  Wave vs non-wave capture rate ratio ~2.6× fit / ~3.3× holdout.
- Expansion never fully stops: last neutral capture at median ~86% of game
  length; post-contact capture 0.152/tick vs 0.503 pre-contact.
- Land plateaus at ≈ 0.24 of playable cells (median; p90 0.31).
- Big-carry departures (stack ≥ 15 parked ≥ 5 ticks): army median 21, parked
  median 38 ticks; spike at `t%50 == 1` (15.3% vs 2% uniform).
- Pass before bulk growth (`t % 50 == 49`) is **not** elevated — do not idle
  for +1.

Single contiguous component through expansion end — MEASURED (grok H3).

---

## 5. Castles — MEASURED (spend-detector)

- 0 real builds before t≈116 in the corpus (fable min 116; grok: none in
  first 50, first real median **138**).
- First-build timing band: fable median **122** (cluster 116–134; secondary
  ~153–177); grok spend-detector median **138**. Use **never before 116**;
  operating first build in **116–150**. Exact tick gate vs emergent growth is
  UNKNOWN.
- Cost: exactly **35** in ~82% (never > 45); army_before median ~34–37.
- Spacing: fable mode Manhattan **7** to nearest own structure; grok median
  **9** to general. Place on frontier (distance 1–2 to non-own ~81%), slightly
  toward the enemy.
- Cadence: ~1 per 50-tick cycle; ≤ 4 per game; interval median ~44.
- Build tick often in gather phase (`t%50` median ~18).
- Contested predictor (partial): army ratio at t=120 ≤ 1.10 → game contains a
  build (acc ~0.78 / holdout ~0.81). “Behind always builds” fails on lag game
  24189 only among observed games.
- ~27% of builds fire while far ahead; that trigger is UNKNOWN. Some wins
  never show a real spend (skip condition UNKNOWN).
- Castles are drained like the general (no standing garrison).

---

## 6. Attack — MEASURED

- Engagement is mostly routing. Attack source = largest stack adjacent to a
  visible enemy cell (~94.5–95.6%). Destination = enemy neighbour minimizing
  BFS distance to the (remembered / believed) enemy general (~93% post-sight /
  ~87% pre-sight).
- Margin: `moved > defender` in ~95–97%; P(attack | negative margin) ≈ 1%;
  median delay 0 when a winning margin appears.
- Full send on attacks ~99.8%.
- Front class after contact (grok): push ~76% / trade ~13% /
  gather_then_push ~10%. Compatible with continued neutral grabs if push and
  expand interleave; exclusive-mode reading is open.
- Commit when held ≥ 10: sent ≥ held (surplus median ~+14.5) — INFERRED.
- Enemy castles: optional / on-path (visible→captured median 2 ticks when
  on-path; many visible enemy castles never taken; ~25% of wins capture one).
- Kill: **0 kills without prior sight** in resolvable wins. Sight→kill median
  **20** (fable) / **24** (grok); use **~20–24**. Tip@sight median **23**
  (floor ≥ 10). Killing chain often departed before sight (median −10.5).
  Marches through fog at unchanged distance-closing rate. Zero passes in last
  15 ticks of every resolvable win.

---

## 7. Defense — present but rare (**MERGED**)

Default (common path, fable-measured):

- No standing garrison; general and castles drain to 1–2 on the gather
  schedule even with distant threats.
- Drain rate with a visible enemy within 6 of the general differs from clear
  ticks by ≤ ~0.4–0.4 pp (holdout 6.2% vs 5.8%).
- Routine homeward-chain rate is nearly flat in threat distance; naive
  “response rate” equals the no-threat baseline for ordinary homeward moves.
- Wins keep the enemy far from home (min dist median **7**; ≤ 1 in only
  ~3.8% of fit wins vs 100% of losses).

Rare override (grok-measured when the conflict exists):

- When an away tip faces an enemy tile near home, redirect tip home in a large
  share of those samples (~92% of away-stack samples in the defense skim);
  latency median **~5** ticks.
- Exact `RECALL_PROX_D` is **UNKNOWN** (under-sampled in wins). Prefer a
  small integer (candidate band around 3) and keep it named for tuning.
- Priority when defense and wave both fire: **defense wins** (this merge).
  Wins rarely create the conflict.

Exploit surface (INFERRED): walk a stack ≥ ~1.2× Kubic’s *current* general
army (often 1–8 in a drain trough) to the general while Kubic’s tip is far;
race losses (5) match this pattern.

---

## 8. Tempo and serialization — MEASURED

- Move whenever legal: ~98.4% clean move rate; voluntary passes ≈ 0 outside
  lag. Forced passes at t∈{1,2} (and rare early starvation); never after t=50
  in clean games. Pass rate median ~0.011; gate &lt; 0.05 holdout-confirmed.
- One chain at a time; on break, ~93% start a new source, ~7% resume prior.
- Structure departures ~73–75% on odd ticks (explicit `t%2` vs emergent
  refresh: UNKNOWN).
- Reaction to new sightings: median 1 tick is largely an artifact of the
  advance that produced the sighting (fable). Soft react gate ≤ 3 ticks is
  holdout-WEAK (grok H10); prefer ≤ 3, accept ≤ 10.
- ≥ 1 gather wave every win (median **4**; ~72% pre-sight).

---

## 9. Named constants

| constant | value | confidence | source |
| --- | --- | --- | --- |
| `PASS_TICKS_FORCE` | {1, 2} (0-based: {0, 1}) | high | both |
| `OPEN_FIRST_GAIN_TICK` | 3 | high | both |
| `OPEN_PULSE_TICKS` | {3, 6, 9} then pause | high | both |
| `OPEN_FLOOD_START` | ≈ 27 (±1 operating; holdout ±1 WEAK under coarse detector) | high | both |
| `OPEN_TILES_T50` | 24 (band 20–25) | high | both |
| `OPEN_ARMY_T50` | ≈ 50 | high | grok |
| `GATHER_PHASE` | t%50 ∈ ~[10, 27] | high | fable |
| `WAVE_PHASE` | t%50 ∈ [28, 49] ∪ [0, 9] | high | fable |
| `SPLIT_FULL` | all-but-one on ~99% of moves | high | both |
| `GENERAL_PULL_LEAVE` | leave 1 (~95–96%); prefer odd ticks (~75%) | high | fable |
| `SNAKE_CONTINUE` | src = prev dst ~78% | high | fable |
| `RELAY_DIR_REPEAT` | ~96–99% (opening) | high | fable |
| `ATTACK_SRC_LARGEST` | ~95% | high | fable |
| `ATTACK_DST_MIN_BFS` | ~92–93% post-sight | high | fable |
| `ATTACK_MARGIN` | moved > defender ~95–97% | high | fable |
| `CONTACT_TICK` | median 82 | high | grok |
| `CONTACT_ARMY_RATIO_MIN` | 1.0 (median ~1.05) | high | grok |
| `EXPAND_COMPONENTS` | 1 through expansion end | high | grok |
| `POST_CONTACT_NEUTRAL_SHARE` | median ~0.63 of gains in next 50 | medium | grok |
| `CASTLE_EARLIEST` | ≥ 116; none in first 50 | high | both (spend) |
| `CASTLE_FIRST_OPERATING` | median band 122–138 | medium | both |
| `CASTLE_COST` | 35 typical; ≤ 45 | high | both |
| `CASTLE_DIST` | ~7 to nearest own structure (mode) / ~9 to general (median) | medium | both |
| `TIP_AT_SIGHT` | median 23; floor 10 | high | grok |
| `SIGHT_TO_KILL` | median 20–24; gate ≤ 120 | high | both |
| `TOWARD_AFTER_SIGHT` | median 0.93; gate ≥ 0.70 | high | grok |
| `GATHER_WAVES_MIN` | ≥ 1 per win; median 4 | high | grok |
| `PASS_RATE_MAX` | &lt; 0.05 | high | both |
| `RECALL_PROX_D` | small integer; candidate ~3 | **low / UNKNOWN** | grok; merge |
| `RECALL_LATENCY` | median ~5 | low | grok |
| `HOME_BANK_PEAK` | median ~27 | medium | grok |

---

## 10. State across ticks

1. **Active chain head** — last move destination; per-cell **last out-direction**.
2. **Persistent objective** — fog-frontier pre-contact; believed then
   remembered enemy general after.
3. **Remembered enemy-general cell** after first sight (generals do not move).
4. **Tick counter** — mod-2 (pull phasing) and mod-50 (gather/wave, build window).
5. **Own/opponent army+land totals** (protocol scalars) — at least for build
   contest.
6. **Home threat distance** — Manhattan from visible/known enemy tile to own
   general; used only by the rare recall gate (**MERGED**).
7. Built castles via **spend**, not raw EventLog early stamps.

**UNKNOWN:** section priors, mountain-pocket tables, MCTS trees, full belief
distribution over unseen general cells (only post-sight latch is strong),
fog terrain model beyond omniscient-BFS proxy.

---

## 11. Failure modes

- **Lag/stall (bist 24184–24189, and similar):** hundreds of voluntary
  passes, pass_rate ≥ 0.30, economic collapse. Policy when acting is
  unchanged — infrastructure failure, not a strategy branch. Holdout
  anomalies (late first move, land@50 collapse, behind-without-building)
  concentrate on 24189.
- **Race over defense (≈5 losses):** tip far (median ~12–22 steps from home)
  while a killer walks in; general often still has some army. Matches the
  rare-recall model: gate did not fire in time, or tip could not return.
- **Never sight (9/11 losses):** strike never latches.
- **Enemy reaches general (11/11 losses):** home distance 0/1.
- **Draw 22221:** sealed map, no contact; 1,100+ stack shuttles forever —
  no deathtouch sample with contact; fog exploration beyond frontier
  objective is weak.

---

## Holdout (both pipelines)

Fable R-rules and grok H-rules both confirm the shared core: early passes,
first move t=3, full send, land@50 band, snake/relay, attack source/dest/
margin, mod-50 capture ratio, castle spend not in first 50, sight before
kill, pass rate low.

Treat as soft, not hard asserts:

- Grok **H10** react ≤ 3 (holdout 0.784).
- Grok **H16** flood 27±1 under coarse detector (0.622); prefer opening
  flood detector.
- Fable **R14b** “behind always builds” — refuted only by lag game 24189;
  holds for non-lag games.

Raw:

- `docs/research/measurements/fable-kubic-holdout-verification.json`
- `docs/research/measurements/grok-kubic-holdout-verification.{json,md}`

---

## Open questions

1. Exact frontier / first-step scorer (center vs nearest-unseen vs toward-egen).
2. Direction tie-break (no lex/clockwise fit).
3. Split trigger (1–3% of moves).
4. Big-run / big-carry launch predicate (timer vs army threshold).
5. Late-build-when-ahead and castle skip condition.
6. Odd-tick departure: explicit `t%2` vs emergent refresh.
7. Pre-sight belief mechanism (enemy-tile gradient vs spawn prior vs deepest fog).
8. Exact integer `RECALL_PROX_D` and vision gating.
9. Deathtouch behavior with contact (n too small).
10. Sight memory for non-general enemy cells.
11. Terrain under fog (pathing vs believed map).
12. Arbiter when adjacent enemy and adjacent neutral both legal on one tick.
13. True actions on grok `multi`-inferred ticks (use fable reconstruction).
14. Cause of bist lag session and unresolvable final frames on some wins.

---

## Source index

| Role | Path |
| --- | --- |
| This merge | `docs/research/strategies/kubic-behavior-spec.md` |
| Fable source spec | `docs/research/strategies/fable-kubic-behavior-spec.md` |
| Grok source spec | `docs/research/strategies/grok-kubic-behavior-spec.md` |
| Fable scripts | `scripts/fable-analyze/` |
| Grok scripts | `scripts/grok-analyze/` |
| Fable aggregates | `docs/research/measurements/fable-kubic-*.json` |
| Grok aggregates | `docs/research/measurements/grok-kubic-*.{json,md}` |

### Reproduction

```bash
.venv/bin/python scripts/fable-analyze/fable_kubic_split.py
.venv/bin/python scripts/fable-analyze/fable_kubic_common.py all
.venv/bin/python scripts/fable-analyze/fable_kubic_opening.py
.venv/bin/python scripts/fable-analyze/fable_kubic_holdout.py

.venv/bin/python scripts/grok-analyze/kubic_corpus.py
.venv/bin/python scripts/grok-analyze/analyze_kubic_opening.py
.venv/bin/python scripts/grok-analyze/verify_kubic_holdout.py
```
