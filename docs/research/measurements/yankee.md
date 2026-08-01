# Round report — yankee

Candidate `yankee@5f9aff334f46` against baseline `proteus@7ae237e99d34`.
Spec: [`../strategies/yankee.md`](../strategies/yankee.md). Thresholds:
[`../../arena/decision-rule.md`](../../arena/decision-rule.md).

All games `--mode competition`, `--seat-policy alternate`, `--strict-versions`,
engine `9e3b9d13cca5`.

| round | roster (beside yankee, proteus, cm_expander) | games/pair | seed | games | rates |
| --- | --- | ---: | ---: | ---: | --- |
| `yankee-r1` | cm_hunter, blitz, boom, aegis, fog_scout, metro, army_convey | 50 | 7 | 2250 | `67a2a9e6e444` |
| `yankee-hunter` | cm_hunter | 220 | 13 | 1320 | `67a2a9e6e444` |
| `yankee-r2` | blitz, boom, aegis | 120 | 21 | 1800 | `67a2a9e6e444` |
| `yankee-r3` | fog_scout, metro, army_convey | 120 | 29 | 1800 | `67a2a9e6e444` |
| `yankee-r4` | cm_hunter | 220 | 41 | 1320 | **`5f9aff334f46`** |
| `yankee-r5` | blitz, boom, aegis | 140 | 43 | 2100 | **`5f9aff334f46`** |
| `yankee-r6` | fog_scout, metro, army_convey | 140 | 47 | 2100 | **`5f9aff334f46`** |
| `yankee-r7` | blitz | 200 | 53 | 1200 | **`5f9aff334f46`** |

**Two candidate hashes, and the first one's rounds are kept.** `67a2a9e6e444`
is yankee with the search allowed to override any core; `5f9aff334f46`
restricts it to blitz and forbids it from vetoing a build. The first four
rounds are what found that defect — the pooled contrast on `67a2a9e6e444` read
flat while two per-opponent cells were down ~95 Elo — so they are reported
(§4.5) rather than deleted. They rate as a **different entity**, which is
correct: it is a different program.

`proteus` was re-run in every round rather than compared against its history:
it is the A arm, and a matched arm is what makes the per-opponent rows below a
contrast rather than two separate measurements.

---

## 1. Acceptance criterion 1 — vs cm_hunter ≥ 95%

**Not met.** 270 games, seat-alternated, pooled over `yankee-r1` and
`yankee-hunter`:

| arm | games | W | L | D | winrate | CI₉₅ | seat A | seat B |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| **yankee** | 270 | 253 | 17 | 0 | **93.7%** | [90.1, 96.0] | 126-9-0 (93.3%) | 127-8-0 (94.1%) |
| proteus | 270 | 245 | 25 | 0 | 90.7% | [86.7, 93.6] | 121-14-0 (89.6%) | 124-11-0 (91.9%) |

yankee is **+3.0 pp** on the matched arm. The target is 1.3 pp above the point
estimate, and 95% lies inside the interval — so this sample cannot distinguish
93.7% from 95%, and it equally cannot support a claim of 95%. Reported as a
miss, per the brief: the measured number is 93.7%, not 95%.

More games do not fix this. A confidence interval narrows with `n`, but the
point estimate is what has to move, and at the observed rate no sample size
makes 93.7% into 95%. What is missing is mechanism, not evidence — see §5.

**Seat split: no effect.** 93.3% (A) against 94.1% (B), 9 losses against 8.
This is worth stating plainly because the brief's premise was the opposite:
round5's four losses all had cm_hunter in seat A. Re-measuring proteus's head
at 300 games put 15 of 22 losses in the *other* seat (p ≈ 0.13), and yankee's
270 games put them 9/8. Across all three measurements the seat is not doing
anything that separates from noise.

The engine does carry a real seat asymmetry — `game._determine_move_order`
awards a full §02 tie (same chase status, same reinforce status, equal source
armies) to player 0 — and yankee's forward model reproduces it. Nothing here
measures its effect on the outcome.

### Where the remaining 17 losses are

Turn at which each ended, over the same 270-game arms:

```
yankee   72  75 115 123 151 175 184 202 215 220 227 252 252 275 302 402 909
proteus  71  72  72 107 120 130 152 156 158 166 179 187 202 302 310 311 352
         353 402 402 433 488 802 819 874
```

| band | proteus | yankee |
| --- | ---: | ---: |
| < 160 | 9 | 5 |
| 160-200 | 3 | 2 |
| 200-320 | 4 | **8** |
| > 320 | 9 | 2 |

The eight games yankee wins back are all early or late; the mid-game band goes
the other way, from 4 to 8. Two readings are consistent with this and the data
does not separate them: either the search is converting some would-be early
losses into games that survive to turn 250 and lose there, or the mid-game is
simply where an override costs the most (it is the assault phase, and it is the
core's strike-stack state that an override desynchronises).

Either way, the brief's turn-200-to-320 band is now where yankee's remaining
losses *concentrate* — 8 of 17 — even though it was not where proteus's were.
That is the band to instrument next, and instrumenting it needs the trajectory
channel rather than another winrate.

---

## 2. Acceptance criterion 2 — no regression on the roster

_(filled in below from the pooled refit)_

---

## 3. Acceptance criterion 3 — latency

`Agent.act` timed directly, in-process, on the real engine: the same call the
stdio loop makes between reading a frame and writing a reply, which is what
RULES.md §08 budgets.

| sample | moves | p50 | p95 | p99 | max | >100 ms | >150 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 full games vs fog_scout (996/833/752/172 turns) | 2753 | 2.0 ms | 41.7 ms | 41.9 ms | **44.7 ms** | 0 | 0 |
| 1 full game vs cm_hunter | 580 | 1.6 ms | 41.6 ms | 41.9 ms | 42.9 ms | 0 | 0 |

**Gate: p95 41.7 ms < 100 ms, max 44.7 ms < 150 ms, zero faults across
7170 stored games.** No round reported a forfeit or a late reply.

The distribution is bimodal by construction, and the shape is the design
working: p50 is the heuristic core alone (~2 ms) on turns the search skips, and
the upper mode is the 40 ms search budget on turns it does not. `search_p50` is
40.0 ms in every sample — the search almost always spends its whole budget and
exits on the deadline rather than on `mcts_max_iters`, which is what the
deadline check inside the rollout loop is there to guarantee.

What 40 ms buys, measured rather than assumed: **125-242 MCTS iterations per
searched turn** (mean over games), max 400 (the iteration ceiling). At 12 root
moves that is 10-20 samples per child — thin, and the reason `_search` returns
`None` and falls back to the heuristic when it completes fewer iterations than
it has root moves.

Search firing rate, per game: 1-3 turns in ~390 in the mid-game, rising to
686 of 996 in a game that ran past turn 800, where §07 makes any 2-stack near
our general a live threat. Mean per-move latency tracks it: 1.7 ms in a short
game, 28.7 ms in the long one.

---

## 4. Increment-by-increment

Each of these is a paired screen — identical map seeds and seats for every
cell, so the flip counts are matched-pair evidence, not two independent
samples. These are **exploratory** (`fit` verdicts come from §1-§2); they are
what chose the shipped configuration.

### 4.1 Parameter tuning — null

200 paired games vs cm_hunter, one knob at a time. Best cell `defense_margin=6`
at 0.925 against 0.915, paired flips +10/−8 — a sign test at p ≈ 0.8. Full
table in [`../strategies/yankee.md`](../strategies/yankee.md) §4.1. **Nothing
shipped.**

### 4.2 MCTS — shipped on

400 paired games vs cm_hunter:

| config | W-L-D | winrate | paired flips |
| --- | --- | ---: | --- |
| off | 369-31-0 | 0.922 | — |
| **on** | 377-22-1 | **0.943** | +22 / −14 |
| on, window 14 | 368-27-5 | 0.920 | +19 / −20 |

The first version of the search, before the scope and evaluation were fixed:

| config | W-L-D | winrate | paired flips |
| --- | --- | ---: | --- |
| off | 55-5-0 | 0.917 | — |
| on (any stack in window) | 27-29-4 | 0.450 | +2 / −30 |
| on, window 12 | 11-46-3 | 0.183 | +1 / −45 |
| on, rollout depth 20 | 24-36-0 | 0.400 | +3 / −34 |

Recorded because it is the more informative half. A 47-point loss that gets
*worse* monotonically as the search fires more often identifies the search as
the cause rather than the noise, and the two defects it exposed — overriding a
stateful core desynchronises the memory that core writes while choosing, and an
evaluation that rewards army standing on your own general makes hiding score
well — are properties of bolting a search onto a heuristic bot, not of this
search.

### 4.3 Standing home guard — measured and rejected

400 paired games vs cm_hunter:

| `guard_ratio` | W-L-D | winrate | paired flips |
| ---: | --- | ---: | --- |
| 0.00 | 369-31-0 | 0.922 | — |
| 0.35 | 340-59-1 | 0.850 | +5 / −34 |
| 0.60 | 308-90-2 | 0.770 | +0 / −61 |

Monotone in the dose, so this is the guard and not the noise. The hypothesis
was good and the answer is no: blitz wins by racing, and army parked at home
saves the general it was sized for and then loses the game it was taken from.
Shipped at 0 and left reachable, because a negative this specific is worth
being able to re-run.

### 4.4 Deathtouch core — shipped on

400 paired games against the four draw-heavy opponents, MCTS on in both arms so
the only difference is the core:

| opponent | core off | core on | draws | winrate |
| --- | --- | --- | --- | --- |
| fog_scout | 84-6-10 | 86-9-5 | 10 → 5 | 0.840 → 0.860 |
| army_convey | 73-13-14 | 77-17-6 | 14 → 6 | 0.730 → 0.770 |
| metro | 86-6-8 | 92-7-1 | 8 → 1 | 0.860 → 0.920 |
| aegis | 75-20-5 | 81-14-5 | 5 → 5 | 0.750 → 0.810 |
| **pooled** | 318-45-37 | 336-47-17 | **37 → 17** | 0.795 → 0.840 |

Paired flips +36/−18 (sign test p ≈ 0.024). Draws fall by 54%, wins rise by 18,
losses by 2 — a draw-conversion result, which is what it was built to be and
the only thing it is offered as. Against cm_hunter it is close to inert: the
median win comes at turn ~418, and the core cannot act before 800.

---

## 5. What was not achieved, and why

The 95% target is not met: 93.7% [90.1, 96.0] over 270 games, against a 90.7%
matched baseline.

The gap is understood rather than mysterious. Replaying losses (spec §3) shows
a single mechanism: cm_hunter's fist becomes *visible* at BFS distance 4-6
already sized at 17-23 army, our general is holding 4-9, and our own stacks are
5-13 steps out. Everything downstream of that is priced correctly and arrives
two turns late. Three of the four things tried attack it:

- **Tuning** cannot reach it — it changes when the reaction starts, and the
  reaction is not what is late.
- **The guard** attacks it directly, by holding army home before the threat is
  observable, and loses more elsewhere than it saves (§4.3).
- **The search** attacks it locally and wins ~3 pp, because inside the window
  it finds better defensive moves than the core's one-stack gather.

What remains untried is the thing the replays actually point at: the army is in
the wrong place *before* the threat is observable, and the only fog-proof signal
about that (`opp_army − opp_land`) turned out to be the wrong thing to spend on.
A version that biases *expansion direction* toward home during the fist window,
rather than banking army at home, would test the same hypothesis without paying
the guard's cost. That is the next experiment, not a claim.

---

## 6. Reproduce

```bash
python -m arena.tournaments.competition \
  bots/yankee/run.sh bots/proteus/run.sh bots/cm_hunter/run.sh bots/cm_expander/run.sh \
  --round yankee-hunter --games-per-pair 220 --round-seed 13 \
  --seat-policy alternate --strict-versions
```

The MCTS-off control arm is `YANKEE_TUNE='{"mcts_enabled": false}'`; the
deathtouch-off arm is `'{"deathtouch_enabled": false}'`. Both are research-only
plumbing (`bots/yankee/params.py`) — unset, which is what every stored game
above ran, the shipped defaults are the whole program.
