# unclejoe shadow measurements

What the tactics layer *would* have done, measured while it did nothing. The
milestones that end in a shadow report write their sections here:
[tactics-plan.md](tactics-plan.md) §7 — U2 (triggers) below, U3 (what the
searches behind those triggers proved) after it, U4 (the afterstate re-rank)
last.

**U4's verdict is a no-go on the re-rank.** Read
[§ U4](#u4--what-the-afterstate-values-said-2026-08-16) before building on
it, and do not rate the U4 build: it runs past 150 ms.

## U2 — trigger fire rates (2026-08-15)

Build: `bots/unclejoe/` at the U2 code, artifact `4f2ab8703559` (the
step-23500 joe checkpoint, byte-identical to joe-rs's). Host: dev arm64
macOS. Triggers per [tactics-plan.md](tactics-plan.md) §3 with
`SEARCH_DEPTH = 3`.

Reproduce a gate row with the gate command in
[fork-plan.md](fork-plan.md) §5 plus `--seed <n>`, and read the
`tactics shadow` line off stderr. The corpus row needs the first 1,140
frames of `data/joe/joe-rs-parity/games/synthetic-long.in.log` piped through
`unclejoe bench` — see the note under the table for why the whole file is
not one game.

Two sets of numbers, because the defense trigger changed inside U2: it now
requires **first contact** — see [the change](#the-defense-trigger-waits-for-first-contact)
below. Everything else is unchanged between the two columns.

| Game | Turns | Result (unclejoe is player 0) | First contact | Enemy general known | Kill fires | Defense fires, ungated | Defense fires, gated |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gate seed 0 | 770 | loss | turn 92 | 0.0% | 0 (0.0%) | 770 (100.0%) | 678 (88.1%) |
| gate seed 1 | 659 | loss | turn 182 | 0.0% | 0 (0.0%) | 101 (15.3%) | 3 (0.5%) |
| gate seed 2 | 624 | win | turn 85 | 28.5% | 29 (4.6%) | 47 (7.5%) | 0 (0.0%) |
| gate seed 3 | 847 | win | turn 77 | 53.5% | 59 (7.0%) | 54 (6.4%) | 5 (0.6%) |
| corpus replay | 1,140 | n/a | turn 76 | 73.9% | 50 (4.4%) | 112 (9.8%) | 36 (3.2%) |

Every gate match reached a normal end — a general capture, no faults, no
timeouts — and the four outcomes are identical across both builds, as a
shadow layer's must be.

The corpus row is **one pass** of `synthetic-long`. That fixture is the
longest natural game's frames played twice (`make_synthetic_long.py`, to
cross the 512-turn window), so `bench` over the whole file replays one game
through one seat twice and carries memory across the seam: its counters
double the game and arm the second pass from the first. The row above is the
first pass alone, which is a real game.

Fire rates are sane in the sense U2 asked for — with one exception that the
contact gate did not fix. Findings, in the order they matter.

### The kill trigger is gated on ever seeing the enemy general

It cannot fire before the general is located, and that is correct rather than
a gap — no kill is provable against an unlocated general. But the sighting
rate is the real cap on H1's upside, and it splits by outcome: in both games
unclejoe lost, it never saw the enemy general at all, so the kill trigger was
dead weight for the whole game; in both games it won, it had the general
located for 29–54% of turns. The enemy general is a thing you see because you
are winning.

The general is *in view* on only 3–5% of turns even once located, so the army
bound the trigger compares against is usually stale, and by a lot: over the
corpus game's 50 kill fires the sighting was a median **375 turns** old
(max 399). The proof search inherits that — it will be proving kills against
a lower bound that aged, which is the direction that declines rather than the
direction that plays a bad move.

### The defense trigger waits for first contact

**Changed in U2**, after the ungated column was measured. The defense trigger
now returns nothing until `Memory::first_contact_turn` is set — the turn we
first see any enemy cell, latched, so losing sight of them again does not
disarm it. The other two arms need a visible enemy cell anyway, so this is
the fog arm's gate in practice.

The ungated arm fired on **every turn before contact, in every game
measured**: turn 0 onwards, 76 of the corpus game's 112 defense fires, all 92
pre-contact turns of gate seed 0. The arithmetic makes that unavoidable —
with nothing of the opponent ever seen, `hidden` is their entire army, and
the arm reduces to "is there an unseen cell within `D`".

Through turn 13 those fires are *provably* empty: generals spawn ≥17 BFS
steps apart, all enemy army starts on their general, and army moves one step
per turn, so no enemy cell can be within `D = 3` of ours before turn 14.
Between turn 14 and contact the gate is an **assumption**, and the only one
in the trigger layer: under the pessimistic fog model a stack could have
reached our neighbourhood without ever crossing our vision. We take that as
not worth a search. It is recorded here rather than buried because it is the
one place a trigger stopped being a superset.

Cost: nothing else changed. The gate removed exactly the pre-contact fires —
112 − 36 = 76 in the corpus game, matching its 76 pre-contact turns
one for one — and no post-contact fire moved.

### The fog arm still pins at 100% after contact in one game

The gate emptied the fog arm in three of four gate matches (98, 47 and 49
fires → 0), which says those fog fires were *all* pre-contact: by the time
they met the opponent, the bot owned enough ground around its general to
light the whole neighbourhood. Seed 0 is the counter-example and it is
untouched — 674 fog fires on the 678 turns after contact. Its general sits
beside a fog pocket within three moves for the whole game, and the arm's
other half is always satisfied: joe pushes its general's army out and keeps
the garrison small (3–19 in the sampled fires) while the opponent's hidden
total runs 58–232.

So the arm is bimodal rather than merely loose — roughly 0% or roughly 100%,
depending on whether a fog pocket sits inside `D` of our general. The
consequence for U3 is a **budget** question, not a correctness one: a 60 ms
proof search that declines still leaves the re-rank ~45 ms, which is the
design the plan already commits to. But in a seed-0-shaped game the defense
search runs every turn and declines every turn, so its decline path is the
hot path. Tightening the arm further — a hidden-army bound per *reachable*
fogged cell rather than the global budget at every cell — is strategist work
against this data.

### The layer costs nothing measurable

`bench` over the synthetic-long log, two interleaved pairs on the same host
(ms):

| Pair | joe-rs p50 / p99 / max | unclejoe p50 / p99 / max |
| --- | --- | --- |
| 1 | 21.90 / 24.24 / 31.78 | 22.07 / 27.87 / 39.94 |
| 2 | 22.15 / 25.49 / 42.99 | 21.91 / 23.29 / 24.66 |

p50 agrees to 0.2 ms, and the tails move more between two runs of the *same*
binary than between the two binaries — this dev host cannot resolve the
trigger pass at all, which is the expected result for two O(cells) scans and
a bounded BFS against a ~22 ms forward. The number that counts is U5's, on
the Modal one-core x86-64-v3 proxy.

### What did not change

The wire-replay test still shows reply-for-reply equality with the recorded
corpus — run against the final, contact-gated build — which is the guard
that the shadow layer chose nothing: the replies are the network's argmax,
exactly as joe-rs's are. The four gate matches back that up from the other
side: same seeds, same ending turns, before and after the trigger change.

The gate at seed 0 ends on turn 770 where [fork-plan.md](fork-plan.md) §7
recorded turn 385 for U1. That is the step-23500 checkpoint sync, not the
tactics layer — both bots moved to the new artifact together, and the match
is deterministic (two runs, same turn).

## U3 — what the searches proved (2026-08-16)

Build: `bots/unclejoe/` at the U3 code, same artifact `4f2ab8703559`, same dev
arm64 macOS host, same four gate seeds. The triggers are unchanged from U2, so
their columns repeat the table above; what is new is the column after each —
what the proof search behind the trigger did with the fire.

| Game | Turns | Result | Kill fires | Kill proved | Defense fires | argmax refuted | Defense proved | Search nodes | Slowest search |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gate seed 0 | 770 | loss | 0 | 0 | 678 | 1 | 0 | 34 | < 0.05 ms |
| gate seed 1 | 659 | loss | 0 | 0 | 3 | 1 | 0 | 72 | < 0.05 ms |
| gate seed 2 | 624 | win | 29 | 0 | 0 | 0 | 0 | 92,464 | 26.4 ms |
| gate seed 3 | 847 | win | 59 | 1 | 5 | 0 | 0 | 120,091 | 10.7 ms |

2,900 turns. The kill trigger fired 88 times and every fire ran a proof search;
the defense trigger fired 686 times and **two** of them got past the refutation
gate to a search. One override came out of the 90, and no cap tripped in any of
them. Every match reached a normal end — a general capture, no faults, no
timeouts — on the same turn as its U2 run, with the same winner.

### The one override changed nothing

Seed 3, turn 846: `override kill depth 1 nodes 18`, a deathtouch touch onto the
general at (1, 19) from (1, 18). It is the move that won the game.

"Fired once" counts the turns where the layer produced the reply instead of the
network — not the turns where the two disagreed. On this evidence they did not
disagree at all. Re-running the seed with `UNCLEJOE_OVERRIDE=0` gives the same
847-turn game, the same capture by the same player, and an equal value and
trigger line on every turn including 846 — so the two runs are the same game up
to that decision, and the network won on that turn without the override. Which
action the argmax named is not in the log; that it also captured the general on
turn 846 follows from the game ending where it did.

So the honest reading of U3's four matches is: the override fired once in 2,900
turns, and no game moved.

That is a finding rather than a defect, and it is the one H1 was written to be
falsifiable about ([the spec](../../research/strategies/unclejoe.md) §1): a
proof-gated override can only pay where the network is wrong *and* the
position is provable, and this sample found no such position. It also sets the
expectation for U6 — an override-only contrast at this rate cannot move a
rating, so the decomposition arm that matters is the re-rank's.

### The kill search finishes; the caps never bind

The kill trigger fired 88 times across the two won games and the search
declined 87 of them, each time by exhausting its tree rather than by running
out: `kill capped 0` everywhere. Cost per fire is ~3,200 nodes (seed 2) and
~2,000 (seed 3) against a 300,000-node ceiling, and the slowest single search
was 26.4 ms against a 60 ms deadline.

So in real positions the depth-3 proof search is not budget-limited. The reason
is the one the plan predicted: the enemy general is *in view* on 3–5% of turns,
and when it is not, the pessimistic bound puts the whole hidden army on its
cell and every neighbour of it — a position that refutes at depth 1 and costs
almost nothing to refute. The binding constraint is knowledge, not compute.

Worst-case turn cost is therefore ~49 ms — a 22 ms forward plus a 26 ms search
— against a 150 ms limit. U5 re-measures on the Modal proxy, where it matters.

### The defense override is a last-ply net, and both catches were already lost

681 defense trigger fires produced **two** refutations of the network's move,
and no defense proof. That gap is the design working as intended in one place
and running out of room in another.

Working as intended: `refutes` asks whether a *visible* reply takes the general
one ply from now, which needs an enemy stack already adjacent to it. 679 of the
681 fires never had one — they are the fog arm firing on a pocket three moves
out (§ *The fog arm still pins at 100% after contact in one game*, above), and
each ended on the cheap check without a search.

Running out of room: both refutations landed on the turn the game ended, and
neither had a defense to find. By the time a stack that beats the garrison is
adjacent, the ply that could have saved the general is two or three turns back
— and that is exactly the horizon the fog bound makes unprovable, since a
fogged cell two steps from our general is credited with the opponent's whole
unaccounted army. So the defense override is structurally a one-ply net, which
[tactics-plan.md](tactics-plan.md) §7 now records as `DEFENSE_DEPTH = 1` rather
than leaving it implied.

### What did not change

Wire-replay equality still holds with `UNCLEJOE_TACTICS=0`: 14 games, 7,092
turns, every reply equal to Python joe's recording. With the master switch off
the layer is not merely quiet — `decide` returns before the triggers run — so
that test proves the U3 code cannot reach the reply, and the four gate matches
prove that when it can, it did not.

## U4 — what the afterstate values said (2026-08-16)

Build: `bots/unclejoe/` at the U4 code, same artifact `4f2ab8703559`, same dev
arm64 macOS host, same four gate seeds and the same corpus prefix. New in this
milestone: `search/afterstate.rs` renders the position a candidate leaves back
into a frame, `tactics/filters.rs` masks candidates, and `Seat::act` runs one
forward per surviving candidate and logs what a live re-rank *would* have
played. The reply is the argmax on every turn, as U3 left it.

Reproduce a gate row with the gate command in [fork-plan.md](fork-plan.md) §5
plus `--seed <n>`; reproduce the corpus row with the first 1,140 frames of
`data/joe/joe-rs-parity/games/synthetic-long.in.log` piped through
`unclejoe bench`. Read the `tactics rerank` line off stderr. Constants per
[tactics-plan.md](tactics-plan.md) §4: `TOP_K = 4`,
`RERANK_RESERVE_FLOOR_MS = 25`, `CASTLE_SURCHARGE_CAP = 8`,
`CASTLE_LATE_TURN = 650`.

The milestone shipped its constants twice. The first pass ran `TOP_K = 5`
against a **fixed** 25 ms reserve and overran the turn limit; the reserve is
now measured and `TOP_K` is 4. Everything below is the second pass, and
[§ The reserve was a guess](#the-reserve-was-a-guess-and-the-guess-was-wrong)
records what the first one found, because it is the reason the constants
changed.

| Game | Turns | Re-ranked | Forwards/turn | Would change | Gap p50 | p90 | p99 | max | Masked (argmax) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gate seed 0 | 770 | 770 | 3.96 | 381 (49.5%) | 0.0000 | 0.0142 | 0.0446 | 0.0849 | 6 (2) |
| gate seed 1 | 659 | 659 | 3.90 | 375 (56.9%) | 0.0006 | 0.0195 | 0.1198 | 0.2959 | 25 (6) |
| gate seed 2 | 624 | 624 | 3.45 | 258 (41.3%) | 0.0000 | 0.0045 | 0.0235 | 0.0690 | 0 (0) |
| gate seed 3 | 847 | 846 | 2.96 | 357 (42.2%) | 0.0000 | 0.0071 | 0.0328 | 0.2157 | 0 (0) |
| corpus replay | 1,140 | 1,140 | 3.06 | 520 (45.6%) | 0.0000 | 0.0142 | 0.0784 | 0.3935 | 6 (1) |

4,039 re-ranked turns, 13,770 afterstate forwards. Every gate match reached a
normal end on the **same turn with the same winner** as its U2 and U3 run,
which is what a shadow mechanism must produce. Seed 3 re-ranked 846 of 847
turns: the missing one is turn 846, where the U3 kill proof overrode and the
turn ended before a slate was built.

### The go/no-go: no-go, and the reason is a unit

The plan's bar is that "live re-rank requires candidate value gaps that stand
clear of the near-zero mass of the gap distribution". There is a natural unit
to measure that in, and it decides the question.

joe's value head is a distribution over **128 bins spanning [−1, +1]**
(`bin_centers` in the artifact), so one bin is **0.0157** wide. A value
difference below that is not the head preferring one position to another; it
is the softmax interpolating between two adjacent bins.

| Sample | Turns | Would change | Gap ≥ 1 bin | Gap ≥ 2 bins |
| --- | --- | --- | --- | --- |
| gate seed 0 | 770 | 49.5% | 8.6% | 3.0% |
| gate seed 1 | 659 | 56.9% | 13.7% | 7.0% |
| gate seed 2 | 624 | 41.3% | 2.7% | 0.6% |
| gate seed 3 | 846 | 42.2% | 4.7% | 1.4% |
| corpus replay | 1,140 | 45.6% | 8.9% | 3.7% |
| **all** | **4,039** | **46.8%** | **7.8%** | **3.1%** |

So the re-rank would take the move away from the network on **nearly half of
all turns**, and on **83% of those** it would do so on a value difference
smaller than one bin of the head's own output. The median gap is 0.0000 and
the p90 is 0.005–0.020 — about one bin. Only the p99 tail (0.024–0.12, up to
0.39) is a difference the head could actually resolve.

That is the failure the spec named in advance and priced at one milestone
rather than one round
([unclejoe.md](../../research/strategies/unclejoe.md) §1: "if the shadow
re-rank shows candidate value gaps sitting inside the head's noise, the
re-rank never goes live and the negative is recorded without spending a
round"). **H2 is not supported.** The re-rank does not go live as designed.

It is a no-go on *this* design, not a proof that the value head is useless
here. The 7.8% of turns clearing one bin are a different population from the
46.8%, and a re-rank gated on a **minimum gap** would only ever act on them.
That is the strategist revision this data argues for, and it is cheap to
try: the machinery is built, the switch exists, and the threshold is one
constant.

### The reserve was a guess, and the guess was wrong

The first U4 build reserved a fixed `RERANK_RESERVE_MS = 25` ms before
starting an evaluation. An evaluation is a whole forward pass — 22 ms typical
on this host and 48 ms at joe-rs's own max — so the loop routinely started
work it could not finish. The anytime structure bounded where it stopped
*starting* work, not where it stopped working, and the measured cost was
**p99 143 ms, max 181 ms** against a 150 ms limit.

The fix is to stop declaring the number and measure it. The loop now keeps
the cost of the slowest evaluation it has run this game and starts no other
with less than that left; the warmup forward `Seat::new` already pays for
seeds it before the first frame, so the constant that remains is a floor the
measurement is normally above. This is what makes the bound
host-independent — a slower machine measures a bigger reserve and runs fewer
candidates, which is a degradation of an anytime loop and not a fault.

`bench` over the corpus prefix, same host, one pass each (ms):

| Build | p50 | p90 | p99 | max |
| --- | --- | --- | --- | --- |
| joe-rs | 22.26 | 22.48 | 23.55 | 26.34 |
| unclejoe, `UNCLEJOE_RERANK=0` | 22.53 | 24.09 | 28.92 | 47.57 |
| unclejoe, U4 **first** pass (fixed 25 ms reserve, `TOP_K = 5`) | 114.84 | 122.97 | 143.26 | 181.49 |
| unclejoe, U4 **shipped** (measured reserve, `TOP_K = 4`) | 88.77 | 102.96 | 112.49 | **130.91** |

The max lands on 130.91 ms against an internal deadline of 130, which is the
structure doing exactly what it claims: the loop never starts work past the
deadline minus a reserve that covers it, so the turn ends at the deadline plus
one reply emit, and the 20 ms of slack to 150 is still slack.

`TOP_K` changed for the same reason. Five afterstates plus the live forward is
~138 ms before any rendering, so the first pass cut itself off on 4,008 of
4,037 turns and averaged 3.9 — the anytime cutoff was quietly supplying the
real constant. Four is written down instead.

Four is still not always reached: the five runs average **3.4** forwards, and
the reserve settled between 40 and 55 ms where a typical evaluation costs ~23.
That is the reserve ratcheting on host jitter — it keeps the worst evaluation
ever seen and never decays, so one stall pins it for the rest of the game.
Deliberately so, in the conservative direction; a reserve set to the typical
cost is precisely the bug this section is about. Whether a decaying bound buys
the fourth candidate back without giving back the guarantee is a U5 question,
and it belongs on the Modal one-core x86-64-v3 proxy, where joe-rs's own tail
is 34.5 ms rather than this host's 26–48.

**The gate matches still say nothing about timing.** `matchup.py` has no
per-move time limit; the judge does (RULES.md §08). The bench above is the
evidence, and U5's proxy run is what a rated round waits on.

### The masks are cheap, correct, and almost never load-bearing

Across 4,039 turns the filters removed **37** candidates: 24 crowding, 1
lateness, 12 refutation. They removed the *argmax* — the only case where a
mask can change the played move — on **9** turns, 0.22% of them. Three turns
filtered every candidate and fell back to the argmax, which is the intended
degradation.

The distribution is lumpy in the way castle play is: seed 1 built 9 castles
and took 21 of the 24 crowding masks; seeds 2 and 3 built 2 and 0 and took
none. So the mechanism is real but rare, and a contrast on it alone would need
far more games than §6's ~1,150 per arm.

The refutation veto fired 4 times per game in the two games unclejoe lost and
never in the two it won — the same shape U3 found for `refutes` as an override
gate, and for the same reason: a refutation needs an enemy stack already
adjacent to our general, which is a position you reach when you are losing.

One afterstate in seed 2 came back `Wins` — a candidate that takes the enemy
general with the opponent standing still, which the U3 kill proof had declined
because the opponent does not have to stand still. It cost no forward and the
value head never had to rank it.

### What did not change

Wire-replay equality still holds with `UNCLEJOE_TACTICS=0`: 14 games, 7,092
turns, every reply equal to Python joe's recording. That covers this
milestone's one mechanical edit to a copied file — `pub` on
`board::obs::BUILD_BASE_COST` ([tactics-plan.md](tactics-plan.md) §1) — and
the four gate matches cover the rest: same seeds, same ending turns, same
winners as U2 and U3.

The Rust suite is 101 tests, all passing — 27 new: 10 on the afterstate
renderer, 8 on the filters, and 9 on the slate, the score order, the anytime
guard and the measured reserve. The Python suite is untouched at 13.6 s.
