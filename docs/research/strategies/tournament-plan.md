# Tournament plan — the 13-competitor pool

How to rank the pool, what to measure when games draw, and what "healthy"
looks like. Follows [`experiment-protocol.md`](../experiment-protocol.md);
runs through `arena/tournament.py`.

Measured cost basis: one full-length 1200-turn competition match takes
**≈ 3.9 s** wall clock (`expand_plus` vs `general_hunter`, seed 0, measured
2026-07-31). Decisive games are shorter. Every estimate below uses 4 s.

---

## 1. Roster

Twelve local bots plus the engine reference agent. All twelve
`bots/<name>/run.sh` paths, plus
`competition-module/competition/agents/expander_python/run.sh`.

| Bot | Idea | Spec | Role |
| --- | --- | --- | --- |
| `smoke` | minimal expander | [`docs/bots/smoke.md`](../../bots/smoke.md) | frozen weak anchor — do not optimize (see [`optimize-existing.md`](optimize-existing.md) §4.1) |
| `expander_python` | engine reference | submodule | frozen external anchor |
| `expand_plus` | expand + frontier march | [`docs/bots/expand-plus.md`](../../bots/expand-plus.md) | expansion baseline |
| `castle_builder` | castle economy | [`docs/bots/castle-builder.md`](../../bots/castle-builder.md) | economy baseline |
| `general_hunter` | deathtouch beeline | [`docs/bots/general-hunter.md`](../../bots/general-hunter.md) | aggression baseline |
| `army_convey` | funnel interior army forward | [`army_convey.md`](army_convey.md) | new |
| `fog_scout` | probe fog systematically | [`fog_scout.md`](fog_scout.md) | new |
| `garrison` | defend the general first | [`garrison.md`](garrison.md) | new |
| `late_rush` | mass one stack, commit at 650–800 | [`late_rush.md`](late_rush.md) | new |
| `splitter` | `split=1` where it beats `split=0` | [`splitter.md`](splitter.md) | new |
| `choke_control` | hold corridors | [`choke_control.md`](choke_control.md) | new |
| `phase_switch` | explicit early/mid/late phases | [`phase_switch.md`](phase_switch.md) | new |
| `castle_rush` | aggressive build parameters | [`castle_rush.md`](castle_rush.md) | new |

Two frozen anchors (`smoke`, `expander_python`) are what make Elo movement
interpretable. Never change them inside a measurement season.

Round 1 standing, provisional — one game per pair, seed 0, no side swap
([`../measurements/round1.md`](../measurements/round1.md)):
`army_convey` 1616.4, `fog_scout` 1580.4, `late_rush` 1576.0,
`castle_builder` 1495.2, `castle_rush` 1473.8, `phase_switch` 1473.7,
`expand_plus` 1464.7, `choke_control` 1457.2, `splitter` 1455.5,
`garrison` 1453.7, `smoke` 1453.5. `general_hunter` and `expander_python`
played no games.

### Entry gate

No bot enters the grid until it has, on its own:

1. finished one `--mode competition` match (the gate in root `AGENTS.md`);
2. produced no crash and no malformed action line;
3. stayed inside the 150 ms per-move budget on a 21×21 board.

A bot that fails the gate is excluded, not carried. A broken bot in a round
robin corrupts 12 other bots' ratings.

---

## 2. Seed grid and pairings

### Why sides must be swapped

Seat order is not cosmetic. `game._determine_move_order` breaks a chasing /
reinforcing / smaller-army tie by seat, and player 0 is polled first each
tick. Any claim from single-side games is confounded by seat. Always pass
`--swap-sides`, and treat a pair as one paired sample of two games per seed.

### Why the seeds must be fixed, and some held out

`matchup.py --seed` is reproducible locally, but the competition
infrastructure never reuses seeds (`RULES.md` §01). Tuning on seeds 0–2 and
reporting on seeds 0–2 measures memorization of three maps. Split the grid:

| Grid | Seeds | Use |
| --- | --- | --- |
| Development | 0–9 | iterate, tune thresholds, compare versions |
| Holdout | 100–111 | **never** used for tuning; run once per season for the published leaderboard |

Report both. A large development-to-holdout gap means the thresholds are
fitted to specific maps.

### Staged schedule

Full round robin with swapped sides on 10 seeds is
`C(13,2) × 2 × 10 = 1560` matches ≈ 1 h 44 m. That is affordable but wasteful
while bots are still changing. Stage it:

| Stage | Field | Seeds | Sides | Matches | Est. time | Purpose |
| --- | --- | --- | --- | --- | --- | --- |
| 0 — gate | each bot vs `smoke` | 0 | one | 12 | ~1 min | catch broken bots |
| 1 — screening | all 13 | 0–4 | one | 390 | ~26 min | first ranking, find the dead bots |
| 2 — full | all 13 | 0–9 | both | 1560 | ~1 h 45 m | the development leaderboard |
| 3 — final | top 6 by stage 2 | 100–111 | both | 360 | ~24 min | the published holdout leaderboard |

Stage 2 can be skipped for a single-bot change; use the pairwise A/B in
`.cursor/skills/evaluate-bot-change/` instead.

### Parallel workers (do not shell-shard)

`arena/tournament.py` runs an in-process worker pool. Default `--jobs` equals
physical CPU cores and is capped there. Workers pin JAX/OpenMP to one thread
each. Elo rebuilds once after the pool (omit `--no-ratings` for that, or pass
`--no-ratings` and run `scripts/leaderboard.py` yourself).

```bash
source .venv/bin/activate
BOTS="$(ls -d bots/*/run.sh) competition-module/competition/agents/expander_python/run.sh"

python arena/tournament.py $BOTS \
  --round stage2-dev \
  --games-per-pair 50 \
  --round-seed 1 \
  --timeout 120
```

`--timeout 120` is ~30× the measured match time, so it only fires on a hung
bot. Without it one deadlocked bot stalls the whole grid.

Do not raise `--jobs` past physical cores: a bot starved of CPU can exceed the
150 ms budget and record faults that are the harness's fault, not the bot's.
Do not launch multiple tournament processes that update ratings at once.

---

## 3. Success metrics beyond "everything drew"

### The problem with the current metric set

Written when the store held 31 games, 31 draws, and every bot at exactly 1500
Elo. Elo is degenerate under an all-draw pool: it is mathematically correct and
carries no information. Two things fix this: a primary health metric, and a
tiebreak that ranks draws.

**Round 1 resolved the degenerate case.** 58 games, 24 decisive, decisive rate
41.4%, round Elo spread 1616.4 down to 1453.5
([`../measurements/round1.md`](../measurements/round1.md)). Elo is now the
primary ranking signal. The shadow leaderboard below drops to a secondary role
and is discarded once the decisive rate holds above 30% across a full round.

### Primary: decisive rate

```
decisive_rate = games with terminated == true / total games
```

This is the health metric for the whole season. Targets:

| decisive_rate | Reading |
| --- | --- |
| < 10% | the pool is broken; stop tuning and fix the root causes in [`optimize-existing.md`](optimize-existing.md) §1 |
| 10–30% | the win conditions work but scouting is too slow |
| 30–70% | **healthy** — rank with Elo, differences are real |
| > 90% | one bot dominates or a general-snipe exploit is unanswered; check that reserves (§3.1 there) are in place |

Report it per bot as well as per season. A bot with a 0% decisive rate across
24 games either cannot win or cannot lose, and either way is not being
measured.

### Secondary: rank the draws by territory

A 1200-turn draw at 90 land against 20 land is not the same game as 55 against
55, and the current store cannot tell them apart. Once the schema fields in
[`optimize-existing.md`](optimize-existing.md) §5.2 land, compute:

```
land_diff  = final_land_a - final_land_b
army_diff  = final_army_a - final_army_b
```

Publish a **shadow leaderboard** alongside the real one: identical Elo
computation, except a draw is scored as a win for the side with more final
land (a true tie stays a draw). Rules for it:

- It is analysis only. It never overwrites `data/ratings/leaderboard.json`;
  write it to `data/ratings/leaderboard-shadow.json` and label it clearly.
- It is the ranking signal *while* the pool is draw-heavy, and it is discarded
  once `decisive_rate` clears 30%.
- Never quote a shadow rating as a bot's rating.

This is what makes an all-draw season rankable at all, and it is the cheapest
way to see whether a strategy change helped before it starts winning outright.

### Full metric set per season

| Metric | Source | Question it answers |
| --- | --- | --- |
| W-L-D, Elo, Elo delta | `arena/ratings.py` | who is better |
| decisive_rate | `terminated` | is the season informative |
| mean / median turns | `turns` | are wins early or at the buzzer |
| land_diff, army_diff at truncation | schema §5.2 | who was winning a drawn game |
| castles built per side | `[matchup] castles built:` line, already printed | did the economy bots actually build |
| turn of first enemy-general sighting | bot telemetry | the scouting signal — the deciding skill |
| deathtouch attempts, chases | bot telemetry | did the endgame logic ever fire |
| faults, timeouts, crashes | matchup stderr | is the bot legal |

The sighting turn is the most diagnostic single number in the pool. Deathtouch
makes the endgame army-blind, so the bot that finds the enemy general first
wins the endgame. If no bot ever sights a general, the season will draw no
matter how the other knobs move.

---

## 4. Reading the results without fooling ourselves

### Sample size

Ten games per pair (5 seeds × 2 sides) detects only large effects. Rough
95%-confidence thresholds for "this pair is not a coin flip":

| Games per pair | Record needed |
| --- | --- |
| 10 | 9–1 |
| 20 | 15–5 |
| 40 | 26–14 |

So: do not report a 6–4 pairwise result as a finding. Pooled Elo across 12
opponents is far stronger evidence than any single pair, which is the reason
to run the round robin rather than a handful of favourite matchups.

### Paired comparisons for a bot change

Seeds are fixed, so a before/after comparison is *paired*: for each seed and
side, the outcome either flipped or it did not. Use the count of flips (a sign
test), not the difference of two winrates. A change that flips 7 of 10 games
in one direction and 0 in the other is strong; 5 up and 4 down is nothing.

### Confounds to state in every note

- **Seat order** — mitigated by `--swap-sides`; unmitigated results are void.
- **Shared code lineage** — most bots fork `expand_plus`, so their errors are
  correlated. Beating a pool of near-copies is weak evidence of generality.
  The two frozen anchors partly control for this.
- **Map memorization** — see the holdout grid in §2.
- **Draw inflation** — a bot can climb the shadow leaderboard by turtling and
  accumulating land while never contesting anything. Cross-check the shadow
  rank against the decisive rate before believing it.

---

## 5. Expected matchup structure

From the specs, before any data. Round 1 scored four of the five predictions;
its round robin was one game per pair on seed 0 with no side swap, so every
verdict is provisional.

| Prediction | Falsified if | Round 1 verdict |
| --- | --- | --- |
| `garrison` beats `late_rush` and `general_hunter` | it loses to a deathtouch runner — its intercept logic does not work | **Falsified.** `garrison` lost to `late_rush` at turn 477 and finished 0-3-7. |
| `late_rush` and `general_hunter` beat the pure economy bots (`castle_builder`, `castle_rush`) | the economy bots survive to 1200 — the rush never finds a general | **Half confirmed.** `late_rush` beat `castle_rush` at turn 769. `general_hunter` played no games. |
| `fog_scout` has the earliest first-sighting turn | a bot that does not scout finds generals sooner, which would mean sightings are accidental | **Untested.** No bot reports a sighting turn. |
| The economy bots beat the pure expanders in long games | draws at similar land, meaning castle income never converts | **Falsified.** Castle count and result run opposite: 4 castles → 0 wins, 3 → 0 wins, 2 → 1 win, and all three winning bots built 0. |
| `smoke` and `expander_python` finish last | they do not, which means the new bots are not actually stronger | **Confirmed for `smoke`** — last at 1453.5. `expander_python` did not play. |

If `smoke` does not finish last, stop and investigate before reading anything
else in the table. It is the control.

---

## 6. Round 2

Round 1 was a screening grid: 58 games, one side, seed 0 for the round robin,
and two competitors missing. Round 2 makes the ranking trustworthy and scores
the revisions in
[`optimize-existing.md`](optimize-existing.md#parameter-revision-1).

### 6.1 What round 1 leaves open

| Gap | Effect on round 1 | Round 2 answer |
| --- | --- | --- |
| No side swap | Every round-robin result is confounded by seat order (§2). | `--swap-sides` on every stage. |
| One game per pair | §4 needs 9–1 over 10 games to call a pair. Round 1 called pairs on a single game. | 5 seeds × 2 sides = 10 games per pair. |
| `general_hunter` absent | The aggression baseline has never been measured. | Full entry, after defect H2 is fixed. |
| `expander_python` absent | The external anchor is missing, so cross-round Elo is not comparable. | Full entry, frozen. |
| Unequal games per bot | 4 for `castle_builder`, 16 for `smoke`. Ratings are not comparable. | Complete round robin, equal games for all. |
| No land, army, or sighting telemetry | 34 drawn games carry no ranking information. | Land §5.2 telemetry first; it is a blocker, not a nice-to-have. |

### 6.2 Entry gate for round 2

The §1 gate still applies, plus three conditions that come from round 1:

1. **The four revised bots must pass their own A/B before entering.** Steps 4–6
   of [`optimize-existing.md`](optimize-existing.md) R1.6 are paired
   comparisons against the round 1 version, not part of the round robin.
2. **`general_hunter` must fire its probe path at least once per game.** Defect
   H2 disables it today. A bot whose differentiator never executes contributes
   a rating but no information.
3. **Per-move time must stay under 150 ms on a 21×21 board.** The probe search
   runs one BFS per candidate cell. Round 2 is the first grid where that path
   runs at all, so measure it before the grid, not during it.

`smoke` and `expander_python` enter frozen. Changing either inside round 2
voids the round.

### 6.3 Schedule

Thirteen competitors. Measured cost from round 1: 208 s of wall clock for 58
games, so 3.6 s per game, consistent with the 4 s estimate in §2.

| Stage | Field | Seeds | Sides | Matches | Est. time | Purpose |
| --- | --- | --- | --- | --- | --- | --- |
| 2.0 — gate | the 4 revised bots + `general_hunter` + `expander_python`, each vs `smoke` | 0–1 | both | 24 | ~2 min | catch a bot broken by revision 1 |
| 2.1 — A/B | each revised bot vs its round 1 version | 0–9 | both | 80 | ~5 min | score revision 1 by the sign test in §4 |
| 2.2 — full | all 13 | 0–4 | both | 780 | ~47 min, ~16 min over 3 shards | the round 2 development leaderboard |
| 2.3 — holdout | top 6 from 2.2 | 100–111 | both | 360 | ~22 min | the published leaderboard |

Stage 2.2 is `C(13,2) × 2 × 5 = 780`. Shard by seed with `--no-ratings` and
rebuild once, exactly as §2 describes. Do not run more than three shards: a
match uses about 1.8 cores, and a CPU-starved bot records faults that belong to
the harness.

### 6.4 Extra pairings round 2 must include

The round robin covers these, but report them separately — each answers an open
question from [`../measurements/round1.md`](../measurements/round1.md).

| Pairing | Question | Reading |
| --- | --- | --- |
| `army_convey` vs `fog_scout` | Round 1's only unresolved top-3 pair; it drew at 1200. | 10 games decide the top of the table. |
| `castle_builder` vs `castle_rush` vs `phase_switch` | Do the economy bots separate on Elo at all? | If the three stay inside 25 points over 10 games each, the economy cluster is one bot in three copies. Apply [`diversity-constraints.md`](diversity-constraints.md) §4. |
| Each economy bot vs `army_convey`, `fog_scout`, `late_rush` | Round 1 never tested `castle_builder` against a bot that wins. | The economy axis stands or falls here. |
| `general_hunter` vs `late_rush` | Both end in a deathtouch execution and differ only in targeting. | If the two produce the same action trace, delete one ([`optimize-existing.md`](optimize-existing.md) §3.7). |
| `garrison` vs `army_convey` | Round 1: `garrison` lost at turn 461 to the bot that masses army. | Tests whether a reserve can hold against a conveyed stack at all. |

### 6.5 Health targets for round 2

| Metric | Round 1 | Round 2 target | If missed |
| --- | --- | --- | --- |
| Decisive rate | 41.4% | 30–70% | Below 30%: the revisions made the pool passive. Above 70%: check that one bot is not simply sniping undefended generals. |
| Elo spread, top to bottom | 163 points | ≥ 100 points over ≥ 60 games each | A narrower spread means the roster is converging; open [`diversity-constraints.md`](diversity-constraints.md) §4. |
| Bots at 0% decisive rate | 1 (`castle_builder`) | 0 | A bot that can neither win nor lose is not being measured. |
| Mean turn of a decisive game | 558 | report, no target | This number sets every turn threshold in the roster. Re-derive `APPROACH_START`, `SENTRY_FROM` and `MAX_TURN_TO_BUILD` from it after each round. |
| Games per bot | 4 to 16 | equal for all 13 | Unequal samples make Elo incomparable. |

### 6.6 Deliverables

1. `docs/research/measurements/round2.md` and `round2.json`, same shape as
   round 1, plus land, army and first-sighting turn.
2. A rebuilt `data/ratings/leaderboard.md` from the full store.
3. One experiment note per strategy change from R1.6, with the three
   diversity tests recorded
   ([`diversity-constraints.md`](diversity-constraints.md) §5).
4. `Parameter revision 2` appended to
   [`optimize-existing.md`](optimize-existing.md).

---

## 7. Runbook

```bash
source .venv/bin/activate

# Stage 2.0 — gate every bot against the anchor
for b in bots/*/run.sh; do
  python arena/run_match.py "$b" bots/smoke/run.sh --seed 0 --timeout 120
done

# Stage 2.2 — full round robin via worker pool (Rule C)
BOTS="$(ls -d bots/*/run.sh) competition-module/competition/agents/expander_python/run.sh"
python arena/tournament.py $BOTS \
  --round stage2-dev \
  --games-per-pair 50 \
  --round-seed 1 \
  --timeout 120

# Stage 2.3 — holdout, top 6 only (fixed seeds if desired)
python arena/tournament.py <six run.sh paths> \
  --round stage2-holdout \
  --seeds 100-111 \
  --timeout 120
```

Default rounds use Rule C (random map seeds, no seat swap). Pass `--seeds` for
a fixed holdout list. Pass `--swap-sides` only when you want both seat orders
on a fixed-seed A/B grid.

Store every game before rating — `arena/tournament.py` stores under
`data/games/<round>/` and rebuilds Elo once after the pool. Ratings are
rebuildable from `data/games/` at any time, so the store is the source of
truth and `data/ratings/` is a derived snapshot.

## 8. Relation to the Phase 3 remote goal

This plan ranks the roster under the **competition** ruleset. It is not the
path to the Phase 3 end goal, which is **95 wins in 100 logged games against
human opponents** on classic generals.io. Classic removes the build action,
adds pre-placed neutral cities, and removes deathtouch, so a competition rank
does not carry over.

The two tracks share the bots and the measurement discipline, and nothing else.
Remote games never enter `data/games/` or `data/ratings/`. See
[`human-95-plan.md`](human-95-plan.md).

## 9. Related

- [`../measurements/round1.md`](../measurements/round1.md) — the round 1 result
  set this plan is built on
- [`diversity-constraints.md`](diversity-constraints.md) — the rules that stop
  the roster collapsing into one policy between rounds
- [`optimize-existing.md`](optimize-existing.md) — root causes of the all-draw
  season, `Parameter revision 1`, and the schema fields this plan depends on
- [`experiment-protocol.md`](../experiment-protocol.md) — per-change protocol
- [`docs/arena/tournament.md`](../../arena/tournament.md) — runner reference
- [`docs/arena/ratings.md`](../../arena/ratings.md) — Elo implementation
- [`docs/engine/remote-eval-heuristics.md`](../../engine/remote-eval-heuristics.md)
  — the separate, non-Elo evaluation path against humans
