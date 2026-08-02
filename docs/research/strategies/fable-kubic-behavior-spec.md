# Kubic — implementation-grade behavior specification

> Superseded for implementation reading by the merge
> [`kubic-behavior-spec.md`](kubic-behavior-spec.md) (defense =
> present-but-rare). Keep this file as the fable evidence trail.

Reverse-engineered from 390 playable scraped leaderboard replays
(`competition-replays/Kubic/`), competition rules (`RULES.md`). Derived on a
351-game fit set; every rule verified against a 39-game holdout (see
[Holdout verification](#holdout-verification)). This is a specification of
observed behavior, not an implementation.

**Corpus and provenance.** 396 replay files; 6 one-tick forfeits excluded.
Real outcomes (derived from the replay `winner` + Kubic's seat, never the
folder): 378 wins, 11 losses, 1 draw. Split: sorted by numeric match id,
every 10th to holdout → fit 351 (340 W / 10 L / 1 D), holdout 39 (38 W /
1 L). Manifest: `docs/research/measurements/fable-kubic-split.json`
(`scripts/fable-analyze/fable_kubic_split.py`).

**Method.** Replays store per-tick state only. `scripts/fable-analyze/fable_kubic_common.py`
reconstructs both players' exact actions per tick by forward-simulating the
engine (builds → move-order rule → combat → growth) over candidate action
pairs and accepting the pair that reproduces the next frame bit-for-bit:
**4 unresolved ticks out of 92,160** across all 390 games (0.004%), ~0.5%
ambiguous. Six dimension analyses (opening, expansion, army, attack, defense,
tempo) each have a deterministic script `scripts/fable_kubic_<dim>.py` and raw
aggregates `docs/research/measurements/fable-kubic-<dim>.json`. Holdout:
`scripts/fable-analyze/fable_kubic_holdout.py` →
`docs/research/measurements/fable-kubic-holdout-verification.json`.

Evidence tags: **MEASURED** (n + rate), **INFERRED** (reasoning stated),
**UNKNOWN**. Percentages are fit-set unless marked `holdout`.

---

## 1. Architecture in one paragraph

Kubic is a **single-task, feed-forward conveyor bot on a 50-tick economic
clock, with no defensive module**. Every tick with a legal move, it moves
(voluntary passes ≈ 0). It runs exactly one stack task at a time: a chain of
all-but-one moves where each move starts on the previous move's destination.
On a mod-50 schedule it alternates a **gather phase** (collect army through
own territory, drain general and castles to 1) and a **wave phase** (roll the
accumulated stack outward, capturing neutral and enemy cells along a
near-shortest path toward its current objective). Attacks are a side effect
of routing, not a separate decision; defense does not exist — the general is
drained on schedule even with an enemy stack at the doorstep, and survival
rests on passive production plus the tie-keeps-defender rule. Castles are a
mid-game economy move: roughly one per 50-tick cycle in contested games,
placed on the frontier at minimum price.

## 2. Decision procedure (priority order)

Pseudocode a reimplementation should follow. Where a step's selection rule is
only partially recovered, the agreement rate is given and §6 lists the gap.

```
each tick t (state = frame t-1):
  1. if no owned cell has army >= 2 and a passable orthogonal neighbor:
       PASS                                            # the only pass Kubic makes
  2. if t <= ~50: run OPENING script (§3)              # fixed choreography
  3. if BUILD conditions hold (§5): BUILD castle       # ~0.25% of actions
  4. otherwise MOVE, chosen as:
       source:
         a. head of the active chain (previous move's dst) if army >= 2   # 78% overall,
            86% after own/enemy-target moves; chains die on exhaustion (army 1)
         b. else start a new chain: usually the general (leave-1 pull,
            preferentially on odd ticks) or a fresh interior stack; returns to
            a previously abandoned chain only ~7%
       destination, by phase (PHASE = t % 50):
         wave  (PHASE in [28,49] or [0,9]): step along the route to OBJECTIVE;
            capture neutral/enemy cells on-route; attack executes the tick a
            winning margin exists (moved > defender in 95-97% of attacks)
         gather (PHASE in [10,27]): step through own territory, merging army
            toward the route/muster; drain castles passed (leave 1)
       split: always all-but-one (99% of moves; exceptions §6)
  OBJECTIVE:
    before enemy contact: a fog-frontier target away from own general
       (best local rules 83-92% agreement; exact scoring unrecovered, §6)
    after contact, before general sight: believed enemy-general direction —
       pre-sight enemy captures reduce true distance to the unseen general
       in 91-92% of cases vs 64-76% baseline
    after sight: the remembered general cell; march near-BFS-shortest
       (median 11-20% path overhead), straight through fog, strike on
       arrival, repeat waves until captured (first wave kills 85%)
  NO defensive branch exists (§7): no recall, no garrison, no threat gate
     on any of the above.
```

## 3. Opening script (t = 1 to ~50) — MEASURED, highly stereotyped

| ticks | behavior |
| --- | --- |
| 1–2 | forced pass (general at 1 army; +1 lands at t=2) |
| 3 | first move: general → neutral, moved=1 (351/351 fit; 38/39 holdout — the exception is the lag-session loss 24189) |
| 3–13 | trickle: capture on the production cycle (t=3 100%, t=6 95%, t=9 75%); land ≈ 4 by t=10 |
| ~14–26 | staging: general emits 1-unit sends every other tick down a fixed 2-step corridor onto a **staging tip** (Manhattan 2 from general in 233/341), which accumulates 6–12 army; near-zero captures; a near-universal regather at t=37–38 exists later in the wave train |
| ~27 (mode; range 19–29) | big run launches from the tip (army mode 7) and snakes ~1 neutral/tick until exhausted |
| ~37–50 | wave train: capture rate ≥ 0.85/tick; general re-launches at army ≈ 7, then 5, then 3 |
| 50 | **land = 24 ± 1** (fit median 24, IQR 23–25; holdout median 24, 36/39 in 23–25) — the production-limited maximum before the first mod-50 growth |

Spawn-independent: land@50 identical for corner/edge/interior spawns and all
board sizes (MEASURED). Direction choice: the chosen capture direction starts
a shortest known-path to the nearest unseen cell in **92.5%** of multi-option
decisions; straight-continuation is preferred when optimal (77.8%) but the
final tie-break is not deterministic from local state (lookup-table ceiling
84.4%, §6). Relay moves follow the source cell's previous out-direction
(96.1% fit, 98.6% holdout) and ignore adjacent capturable neutrals in 92% of
relay ticks — a path-follower, not a greedy expander.

## 4. Economy: the mod-50 cycle — MEASURED

- Action mix by `t % 50`: own-move (gather) share peaks 0.86–0.90 at residues
  14–18; capture/attack share peaks at residues 40–49 (exposure-normalized
  neutral-capture rate 0.538/tick at residues 40–49 vs 0.122 at 10–19; 4.4×).
  Wave predicate `t%50 ∈ [28,49] ∪ [0,9]`: capture rate 0.450 vs 0.171
  outside (fit); holdout ratio 3.27. Land is maximized immediately before
  every all-cell +1.
- Expansion never stops: last neutral capture at median 86% of game length;
  post-contact capture rate 0.152/tick (vs 0.503 pre-contact).
- Land plateaus at ≈ 0.24 of playable cells (median; p90 0.31).
- Big-carry departures (stack ≥ 15 parked ≥ 5 ticks, n=621): army at
  departure median 21, parked median 38 ticks, enemy visible in 92.9%,
  disproportionately launched at `t%50 == 1` (15.3% vs 2% uniform) — the
  wave rides the fresh bulk growth.

## 5. Castles — MEASURED

- 0 builds before t=116 in all 390 games (fit min 116, holdout min 116).
  First build median t=122 (cluster 116–134; secondary ~153–177). Counts per
  game: 0×198, 1×92, 2×46, 3×3, 4×1 (fit wins); interval between builds
  median 44 ticks ≈ one cycle.
- Placement: **cost exactly 35 in 82%** (fit 161/197; holdout 19/23; never
  > 45); Manhattan distance to nearest own structure mode **7** (holdout mode
  7 too); ~7 from the general; frontier-adjacent (distance 1–2 to non-own in
  81%); slightly toward the enemy (0.78 of generals-separation).
- Mechanics: one stack (median 34 army) walks in over ~15 ticks; build leaves
  0–1 army; 78% on even ticks; build tick `t%50` median 18 (IQR 14–22) — in
  the gather phase (holdout: 20/23 in residues 9–28).
- Trigger (partial): contested games build — army ratio at t=120 ≤ 1.10
  predicts "game contains a build" at 0.776 (holdout 0.81). The stronger fit
  claim "no game behind (ratio < 0.95) ever fails to build" is **REFUTED on
  holdout**: 24189 (ratio 0.55, lag-session loss) never built. Restated:
  holds in all non-lag games observed (fit + holdout). 27% of builds fire
  while far ahead (t≈150–220); their trigger is unrecovered (§6).
- Castles are never garrisoned: drained to 1 like the general (median
  garrison at +50 ticks ≈ 10 < passive production), re-drained every ~23
  ticks at median 26 army.

## 6. Attack — MEASURED

- All engagement is routing. Attack source = the largest stack adjacent to a
  visible enemy cell (94.5% fit / 95.6% holdout when ≥ 2 candidates); attack
  continues the active chain (92%); destination = the enemy neighbor
  minimizing BFS distance to the (remembered) enemy general (93.1% fit /
  91.6% holdout post-sight; 86.6% pre-sight against the believed position).
- Margin discipline: `moved > defender` in 94.8% fit / 97.1% holdout;
  P(attack | negative margin) ≈ 1%; when a winning margin appears on the
  front stack it attacks that same tick (median delay 0). The 5% non-winning
  attacks are 1-army pokes and exact ties (defender zeroed, "disarm"), plus
  under-strength general shaves.
- Commitment: no army-lead or sight gate — a completed gather rolls at the
  objective (entry army median 19, smooth distribution, no "big commit"
  threshold; general seen in only 30% of commits).
- Enemy castles: no priority — capture is incidental to the route (visible→
  captured median 2 ticks when on-path; half of visible enemy castles are
  never taken).
- Kill: **0 kills without prior sight** (336/336 resolvable fit wins; 38/38
  holdout wins saw the general before the end). Sight→kill median 20 ticks;
  the killing chain departed before sight in most games (median −10.5) — the
  rolling stack discovers and finishes, with no post-sight gather. Marches
  to the remembered cell through fog (distance-closing rate identical when
  the general is fogged, 88.3% vs 88.8% visible). Zero passes in the last 15
  ticks of every win (fit 340/340, holdout 38/38).
- Pre-sight belief: pre-sight enemy captures reduce BFS distance to the
  true, unseen general in 92% (fit; holdout 90.8%) vs availability baselines
  of 64% (fit, global-option definition) / 75.5% (holdout, local-option
  definition). The fog-legal mechanism (enemy-tile gradient vs spawn prior)
  is UNKNOWN.

## 7. Defense — MEASURED absence

- **No threat-triggered recall**: homeward-chain start rate is flat in
  visible-threat distance (0.36–1.5%/tick everywhere, incl. no-threat 0.45%);
  the best threat-conditioned recall predicate reaches 8.9% precision. The
  apparent 91% "response rate" to incursions equals the no-threat baseline
  probability of a routine homeward move.
- **General drain ignores threats**: drain rate with a visible enemy within
  6 of the general differs from clear ticks by ≤ 0.4 pp (fit; holdout 6.2%
  vs 5.8%). The general is routinely left at 1–2 army.
- No castle garrisons, no path-blocking, no deliberate use of the
  chase-priority rule (3.6% of attacks, no elevation under threat).
- Survival in wins = passive general production + tie-keeps-defender +
  opponent error (in 19/28 close calls the passively accumulated general
  army covered the threat; 3 survived ties like 9-vs-9).
- Exploit surface (for opponents; INFERRED): walk a stack ≥ 1.2× Kubic's
  *current* general army (which cycles down to 1–8) to the general timed to
  the drain trough; Kubic will keep racing (5/5 race losses) rather than
  turn.

## 8. Tempo and scheduler — MEASURED

- Moves whenever legal: 98.4% move rate clean; voluntary passes ≈ 0.02%
  outside one degenerate session. Forced passes only at t∈{1,2,4} and (in 12
  games) exactly at the t≈50 starvation boundary; never after t=50.
- Feed-forward, not reactive: reaction "latency" to new sightings (median 1
  tick) is an artifact of the advance that produced the sighting; response to
  losing own cells is no faster than a shifted control (median 8 vs 6) and
  42% of lost cells are never retaken within 40 ticks.
- Strict serialization: one chain at a time; on a chain break, 92.7% start a
  new source, 7.3% resume the prior chain; no interleaving. Half of breaks
  are forced by exhaustion (abandoned stack ≤ 1).
- Structure departures 73–75% on odd ticks with even re-departure intervals
  (phase-locked to even-tick production; explicit `t%2` check vs emergent
  greed is UNKNOWN). General-origin chains have a length mode at exactly 12.
- No opponent-conditional behavior found (baseline bots vs humans identical
  once the lag session is excluded).

## 9. Named constants

| constant | value | evidence |
| --- | --- | --- |
| first move tick | 3 (earliest legal) | MEASURED 341/351 fit, 38/39 holdout |
| opening staging-tip distance | Manhattan 2 from general | MEASURED 233/341 |
| big-run launch | t≈27 (mode; 19–29), tip army 6–12 (mode 7) | MEASURED; trigger UNKNOWN |
| land at t=50 | 24 (IQR 23–25) | MEASURED fit+holdout |
| gather phase | t%50 ∈ ~[10,27] | MEASURED (action-mix cycle) |
| wave phase | t%50 ∈ [28,49] ∪ [0,9]; capture-rate ratio 2.6–3.3× | MEASURED fit+holdout |
| split usage | all-but-one on 98.8–99.9% of moves | MEASURED |
| general pull | leave exactly 1 (95–96%), odd ticks (75–76%) | MEASURED |
| snake continuation | src = prev dst on 77.8% (fit) / 77.9% (holdout) | MEASURED |
| relay direction repeat | 96.1% fit / 98.6% holdout | MEASURED |
| attack full-send | 99.8–99.9% | MEASURED |
| attack winning margin | moved > defender 94.8% / 97.1%; median margin +19 | MEASURED |
| attack source | largest enemy-adjacent stack 94.5% / 95.6% | MEASURED |
| attack destination | argmin BFS-to-general 93.1% / 91.6% post-sight | MEASURED |
| first castle | t ∈ [116,134] median 122; never < 116 | MEASURED fit+holdout |
| castle cost | exactly 35 in 82–83%; ≤ 45 always | MEASURED |
| castle spacing | Manhattan 7 to nearest own structure (mode) | MEASURED fit+holdout |
| castle cadence | ~1 per 50-tick cycle; ≤ 4 per game | MEASURED |
| build trigger | contested: army-ratio@120 ≤ 1.10 → builds (acc 0.78/0.81) | MEASURED, partial |
| big-carry departure | army med 21 after ~38 parked ticks; t%50==1 spike | MEASURED; trigger UNKNOWN |
| sight→kill | median 20 ticks; first wave kills 85% | MEASURED |
| kill-phase passes | 0 in last 15 ticks (378/378 resolvable wins) | MEASURED fit+holdout |
| recall trigger | none (threat-independent) | MEASURED |

## 10. State the bot tracks across ticks (INFERRED from behavior)

1. **Active chain head** — the cell its last move landed on (drives 78–86%
   of source choices) and, per cell, the **last out-direction** (96–99%
   relay repetition). Fog-independent.
2. **A persistent movement objective** — a fog-frontier target pre-contact,
   the believed then remembered enemy-general cell after; stable across many
   ticks (straight-line walks, consistent wave directions). The objective
   survives fog: the march continues at an unchanged distance-closing rate
   when the target region is not visible.
3. **Remembered enemy-general position** after first sight (generals never
   move, so memory equals truth). No evidence of memory for other enemy
   cells was obtainable (near-general threats were always currently visible).
4. **The tick counter** — mod-2 (production phasing of pulls) and mod-50
   (gather/wave cycle, build scheduling).
5. **Own/opponent army+land totals** (protocol scalars, fog-legal) — used at
   least by the build trigger.
6. No tracked threat state, no defensive memory.

## 11. Failure modes (all 11 losses + draw, fit + holdout)

- **Lag/stall session (6 losses: 24184–24189, all vs bist, consecutive
  ids)**: hundreds of voluntary passes, ~4-tick action cycle, economic
  collapse. Policy when acting is unchanged — an infrastructure failure
  mode, not a strategy branch. All holdout anomalies (first move t=8,
  land@50=6, behind-without-building) are game 24189 of this session.
- **Race chosen over defense (5 losses)**: general drained on schedule while
  a visible killer stack walked in (median fog-legal warning 4.5 ticks);
  Kubic's own stack was 16–22 steps from home at death, 2 steps from the
  enemy general in two games.
- **Draw 22221**: sealed map, enemy never contacted; Kubic shuttles a
  1,100+ army stack forever — no fog-exploration drive exists beyond the
  frontier objective, and no deathtouch behavior was observable anywhere
  (only that one game reached t=800, without contact).

## Holdout verification

39 games (38 W / 1 L), rules frozen before evaluation
(`scripts/fable-analyze/fable_kubic_holdout.py`; raw:
`docs/research/measurements/fable-kubic-holdout-verification.json`).

| rule | fit | holdout | verdict |
| --- | --- | --- | --- |
| R1 pass ⟺ no legal move (t≥11) | 100% clean | 0 violations in 38 clean games (233 in lag game) | CONFIRMED |
| R2 first move t=3 canonical | 97% | 38/39 | CONFIRMED |
| R3 full send (army ≥ 3) | 98.8% | 98.9% (n=7,032) | CONFIRMED |
| R4 general pull leaves 1 | 95.4% | 96.3% (n=789) | CONFIRMED |
| R5 general pulls on odd ticks | 75.5% | 75.9% | CONFIRMED |
| R6 snake src=prev dst | 77.8% | 77.9% (n=8,714) | CONFIRMED |
| R7 relay direction repeat (t≤50) | 96.1% | 98.6% | CONFIRMED |
| R8 attack src = largest adjacent | 94.5% | 95.6% (n=1,056) | CONFIRMED |
| R9 attack full send | 99.8% | 99.9% | CONFIRMED |
| R10 attack moved > defender | 94.8% | 97.1% (n=1,428) | CONFIRMED |
| R11 attack dst min-BFS-to-general (post-sight) | 93.1% | 91.6% (n=203) | CONFIRMED |
| R12 mod-50 wave/gather capture ratio | 2.63× | 3.27× | CONFIRMED |
| R13 castle cost/spacing/timing | 35 / mode 7 / ≥116 | 19/23 cost 35; mode 7; min t=116; ≤3 per game | CONFIRMED |
| R14 build trigger ratio ≤ 1.10 | acc 0.776 | acc 0.81 (n=37) | CONFIRMED |
| R14b "behind always builds" (ratio < 0.95) | 0 exceptions | **1 exception (24189, lag loss)** | **REFUTED as absolute; holds for non-lag games** |
| R15 drain threat-independent | gap ≤ 0.4 pp | 6.2% vs 5.8% | CONFIRMED |
| R16 kill: no pass last 15; sight precedes kill | 100% | 38/38 both | CONFIRMED |
| R17 land@50 = 24 (23–25) | median 24 | median 24; 36/39 in range (outliers incl. lag game) | CONFIRMED |
| R18 pre-sight march at true general | 92% vs 64% baseline | 90.8% vs 75.5% baseline (local-option defn) | CONFIRMED (baseline definitions differ; direction and gap robust) |
| R19 opening src = prev-dst else general | 86.2% | 85.3% (n=1,799) | CONFIRMED |

## Open questions / cannot be determined from the replays

1. **Exact frontier-target selection.** Best fog-legal local rules cap at
   83–92% agreement (BFS-to-nearest-unseen in the opening; away-from-own-
   general mid-game). The residual is consistent with a persistent global
   objective whose scoring was not recovered; identical local situations
   produce different choices (lookup ceiling 84.4%), so local state cannot
   decide it. The two analysts' best mid-game vs opening rules also differ —
   treated as one unrecovered planner, not two rules.
2. **Tie-break among equally scored directions** — no lexicographic,
   clockwise, or direction-order rule fits (pairwise ≈ 50/50); hidden state
   or RNG.
3. **Split trigger** (1–3% of moves): opening relay cluster and a pre-mod-50
   front-feeding cluster are described distributionally; no predicate ≥ 90%.
4. **Big-run/carry launch trigger** — tick-concentrated (t≈27 opening;
   t%50==1 later) but army varies; threshold vs timer confounded.
5. **The late-build trigger when far ahead** (27% of builds) and whether the
   t≈116–126 first-build cluster is tick-gated or emergent from the t=100
   growth.
6. **Odd-tick departure**: explicit `t%2` check vs emergent "leave when
   refreshed" — observationally identical.
7. **Pre-sight belief mechanism** (enemy-tile gradient vs spawn-symmetry
   prior vs deepest-fog): only its effect is measurable.
8. **Deathtouch behavior**: n=1 eligible game, with no contact — nothing
   observable.
9. **Sight memory for non-general cells**: no discriminating situations in
   the corpus.
10. **Terrain model under fog** (pathing vs believed map): omniscient-BFS
    proxy only (11–20% median path overhead).
11. **Cause of the bist lag session** (timeouts vs crash vs adversarial
    compute blowup) — behavioral fingerprint only; and the 4 wins ending on
    an unresolvable final frame at round tick counts (20580, 21537, 21760,
    21855) look like opponent disconnects, not Kubic actions.

## Reproduction

```bash
.venv/bin/python scripts/fable-analyze/fable_kubic_split.py        # census + 90/10 split
.venv/bin/python scripts/fable-analyze/fable_kubic_common.py all   # exact action reconstruction cache
.venv/bin/python scripts/fable-analyze/fable_kubic_opening.py      # + expansion / army / attack / defense / tempo
.venv/bin/python scripts/fable-analyze/fable_kubic_holdout.py      # holdout verification
```
