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

**Not met.** Three measurements, all seat-alternated, all ≥200 games:

| # | arm | condition | games | W | L | D | winrate | CI₉₅ |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | yankee `67a2a9e6e444` | rounds r1+hunter | 270 | 253 | 17 | 0 | **93.7%** | [90.1, 96.0] |
| | proteus (matched) | same rounds | 270 | 245 | 25 | 0 | 90.7% | [86.7, 93.6] |
| 2 | yankee `5f9aff334f46` | paired screen, idle box | 200 | 189 | 10 | 1 | **94.5%** | [90.4, 96.9] |
| | same, search off | paired, same seeds | 200 | 178 | 22 | 0 | 89.0% | [83.9, 92.6] |
| 3 | yankee `5f9aff334f46` | round r4, contended box | 220 | 199 | 20 | 1 | 90.5% | [85.8, 93.7] |
| | proteus (matched) | same round | 220 | 200 | 19 | 1 | 90.9% | [86.4, 94.0] |

The best measurement of the shipped bot is **94.5%** (row 2), against 89.0%
for the same bot with the search off on the same 200 map seeds — a matched
+5.5 pp. Row 3 is the same program measured while the machine was 13×
oversubscribed; §3 shows the search ran at a fifth strength there, and the gain
duly disappears. Row 1 is the earlier candidate on a mostly idle box.

**The target is not reached under any of them.** 94.5% is 0.5 pp short and
93.7% is 1.3 pp short; 95% lies inside both intervals, so no sample here can
distinguish success from noise, and none supports a claim of 95%. Reported as a
miss: the measured number is 94.5%, not 95%.

More games narrow the interval; they do not move the point estimate. What is
missing is mechanism, not evidence — see §5.

**Seat split** (row 1, 270 games): 93.3% seat A against 94.1% seat B, 9 losses
against 8. Row 3: 98-12 against 101-8-1. No effect in either.

This is worth stating plainly because the brief's premise was the opposite:
round5's four losses all had cm_hunter in seat A. Re-measuring proteus's head
at 300 games put 15 of 22 losses in the *other* seat (p ≈ 0.13). Across every
measurement here the seat does not separate from noise.

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

Turn at which each ended, over row 1's 270-game arms:

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

**Verdict: `unproven`.** Not `regression` — which is the part the criterion
forbids — but not `improvement` or `no change` either, so the criterion is not
satisfied.

Pooled refit over `data/games/`, contrast read off `fit.delta`:

| candidate | yankee games | Δ (B−A) | SE | CI₉₅ | P(B>A) | verdict |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| `67a2a9e6e444` (superseded) | 2310 | −1.67 | 12.66 | [−26.48, +23.14] | 0.448 | unproven |
| `5f9aff334f46` (shipped) | 660+ | +8.03 | 23.46 | [−37.95, +54.01] | 0.634 | unproven |

The superseded candidate misses `no change (proven flat)` by **1.5 Elo on the
low side** — `CI₉₅` must lie inside ±25 and its low is −26.48.
`games_to_resolve(target_se=12.75)` for the shipped candidate is **≈1575 more
candidate games**; §3 explains why buying them on this machine would not have
been worth what the numbers would then mean.

### Per-opponent, superseded candidate (2310 games, largely uncontended)

This is the table that matters, because it is the one that caught the defect
the shipped candidate fixes.

| opponent | yankee | proteus | Δ pp |
| --- | ---: | ---: | ---: |
| cm_expander (anchor) | 0.992 | 0.998 | −0.6 |
| cm_hunter | **0.937** | 0.907 | **+3.0** |
| blitz | 0.482 | 0.500 | −1.8 |
| boom | 0.694 | 0.624 | +7.0 |
| aegis | 0.753 | 0.776 | −2.3 |
| fog_scout | 0.824 | 0.894 | **−7.0** |
| metro | 0.859 | 0.912 | **−5.3** |
| army_convey | 0.794 | 0.794 | 0.0 |

fog_scout and metro are ≈ −95 Elo each. The pooled contrast was flat because
cm_hunter, boom and aegis paid for them. **A flat pooled Δ is not evidence of
no regression** — that is what the per-opponent gate exists for, and it earned
its keep here.

Note also how much noise a 50-game cell carries: in `yankee-r1` alone, aegis
read +16 pp for the candidate and boom +18 pp. At 170 and 170 games the same
cells read −2.3 and +7.0. Nothing under ~150 games per cell was worth acting
on.

### Per-opponent, shipped candidate

The fix was verified before the rounds, on an idle machine, 200 paired games
per cell (identical seeds and seats, so these are matched pairs):

| opponent | search off | search unrestricted | search restricted (shipped) |
| --- | ---: | ---: | ---: |
| fog_scout | 0.850 | 0.845 | **0.870** |
| metro | 0.925 | 0.900 | **0.925** |
| cm_hunter | 0.890 | 0.945 | **0.945** |

Both regressions are gone and the cm_hunter gain is intact. The registered
rounds that would turn this into a rated contrast (`yankee-r4` … `yankee-r6`)
ran under the contention described in §3 and read `cm_hunter` at 0.905 against
proteus's 0.909 — consistent with a search running at a fifth strength, and
**not** a measurement of the bot the paired screen measured. The clean rerun is
the outstanding work.

---

## 3. Acceptance criterion 3 — latency

`Agent.act` timed directly, in-process, on the real engine: the same call the
stdio loop makes between reading a frame and writing a reply, which is what
RULES.md §08 budgets.

**On an idle machine — the §08 condition, one dedicated core:**

| sample | moves | p50 | p95 | p99 | max | >100 ms | >150 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 games vs fog_scout | 1280 | 1.8 ms | 41.8 ms | 41.9 ms | **50.6 ms** | 0 | 0 |
| 3 games vs cm_hunter | 1145 | 1.6 ms | 2.2 ms | 2.4 ms | 42.1 ms | 0 | 0 |

**Gate met: p95 41.8 ms < 100 ms, max 50.6 ms < 150 ms, zero faults across
15 210 stored games.** No round reported a forfeit or a late reply.

The distribution is bimodal by construction, and the shape is the design
working: p50 is the heuristic core alone on turns the search skips, and the
upper mode is the 40 ms budget on turns it does not. `search_p50` is 40.0 ms in
every sample — the search almost always spends its whole budget and exits on
the deadline rather than on `mcts_max_iters`.

What 40 ms buys, measured rather than assumed: **144-242 MCTS iterations per
searched turn**, capped at 400. At 12 root moves that is 12-20 samples per
child — thin, and the reason `_search` returns `None` and falls back to the
heuristic when it completes fewer iterations than it has root moves.

### The cap is only as hard as the scheduler

Repeating the fog_scout sample while an unrelated 6-worker tournament shared
the machine (load average 140 on 11 physical cores):

| | idle | 13× oversubscribed |
| --- | ---: | ---: |
| iterations per searched turn | 144 | **31** |
| p50 move | 1.8 ms | 24.1 ms |
| p95 move | 41.8 ms | 100.2 ms |
| max move | 50.6 ms | **272.2 ms** |
| moves > 150 ms | 0 | **4** |
| longest single search | 40.3 ms | **177.3 ms** |

The deadline is checked inside the rollout loop, but a check can only fire when
the process is running. Descheduled mid-rollout for 130 ms, the search returns
137 ms late having *executed* perhaps twenty more microseconds of work. **No
user-space wall-clock cap can survive arbitrary descheduling** — the guarantee
is conditional on §08's dedicated core, and that condition should be stated
rather than assumed.

Two consequences, both real:

1. Under contention yankee degrades *gracefully but not silently*: it still
   returns a legal move (the heuristic is computed first and the search only
   ever replaces it), but it can return late, and 50 late replies forfeit.
2. **Rounds `yankee-r4` … `yankee-r7` were run under this contention.** They
   therefore measure a yankee whose search ran at roughly a fifth strength.
   They are a *conservative* bound on the candidate, not a neutral one, and
   §4 quotes the idle-machine paired screens separately for that reason.

The named next experiment, not shipped here because it is unmeasured: track a
running estimate of rollout cost and refuse to *start* a rollout that cannot
fit in the remaining budget. That removes the systematic overshoot; it cannot
remove the scheduler.

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

400 paired games vs cm_hunter, at the point the search could still override
any core:

| config | W-L-D | winrate | paired flips |
| --- | --- | ---: | --- |
| off | 369-31-0 | 0.922 | — |
| **on** | 377-22-1 | **0.943** | +22 / −14 |
| on, window 14 | 368-27-5 | 0.920 | +19 / −20 |

After restricting it to the blitz core and forbidding build vetoes (§4.5), 200
paired games: off 178-22-0 = 0.890, **on 189-10-1 = 0.945**.

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

### 4.5 Restricting the search — the fix the panel forced

The pooled contrast on `67a2a9e6e444` read flat while fog_scout and metro were
each down ≈95 Elo (§2). A four-arm paired screen, 200 games per cell, split the
two shipped increments apart:

| | fog_scout | metro |
| --- | ---: | ---: |
| both off | 0.850 | 0.925 |
| MCTS only | **0.795** | **0.825** |
| deathtouch only | 0.865 | 0.915 |
| both (as shipped then) | 0.830 | 0.905 |

The search, not the endgame core. Both are opponents yankee answers with
**boom**, whose `build_target` / `build_stack` / `striking` are latches set
while choosing and trusted afterwards — and whose castle build is the last move
of a multi-turn plan the search models not at all, so vetoing it cost every
castle rather than one. blitz survives an override because `ensure_stack` and
the chain-head check re-derive from the board every turn.

`mcts_cores = "blitz"` and `mcts_skip_builds = True`. Re-measured (§2).

---

## 5. What was not achieved, and why

The 95% target is not met: the best measurement of the shipped bot is 94.5%
[90.4, 96.9] over 200 paired games, against 89.0% for the identical bot with
the search off on the identical seeds.

The gap is understood rather than mysterious. Replaying losses (spec §3) shows
a single mechanism: cm_hunter's fist becomes *visible* at BFS distance 4-6
already sized at 17-23 army, our general is holding 4-9, and our own stacks are
5-13 steps out. Everything downstream of that is priced correctly and arrives
two turns late. Three of the four things tried attack it:

- **Tuning** cannot reach it — it changes when the reaction starts, and the
  reaction is not what is late.
- **The guard** attacks it directly, by holding army home before the threat is
  observable, and loses more elsewhere than it saves (§4.3).
- **The search** attacks it locally and wins 5.5 pp, because inside the window
  it finds better defensive moves than the core's one-stack gather.

What remains untried is the thing the replays actually point at: the army is in
the wrong place *before* the threat is observable, and the only fog-proof signal
about that (`opp_army − opp_land`) turned out to be the wrong thing to spend on.
A version that biases *expansion direction* toward home during the fist window,
rather than banking army at home, would test the same hypothesis without paying
the guard's cost. That is the next experiment, not a claim.

## 5b. Outstanding work, stated as such

1. **A clean rerun of `yankee-r4` … `yankee-r7`.** They ran while an unrelated
   6-worker tournament shared the box (§3). The search is wall-clock budgeted,
   so those rounds measure a weakened candidate. Until they are re-run on an
   idle machine, the rated contrast for `5f9aff334f46` is not the number the
   bot deserves — and `yankee-r7` was not run at all.
2. **≈1575 more candidate games** to bring the pooled contrast to
   `SE(Δ) = 12.75`, which is what "proven flat" costs.
3. **A per-opponent Elo contrast**, rather than the per-opponent winrates in
   §2. The gate as written asks for `CI₉₅.high > −10` per opponent; the fit
   exposes a whole-entity contrast, so the per-opponent check here is a
   winrate difference with a Wilson interval. That is weaker than the gate
   asks for and should be built.

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
