# unclejoe shadow measurements

> **The bot was removed on 2026-08-20.** `bots/unclejoe/`, `fork-plan.md`, and
> `tactics-plan.md` exist only in git history from that date, so the links to
> those two pages below are dead on disk. This page stays because it is the
> evidence of rounds that ran: read it as a record, not as a description of a
> bot in the tree.

What the tactics layer *would* have done, measured while it did nothing. The
milestones that end in a shadow report write their sections here:
[tactics-plan.md](tactics-plan.md) §7 — U2 (triggers) below, U3 (what the
searches behind those triggers proved) after it, U4 (the afterstate re-rank)
last.

**U4's verdict: no on the re-rank as specified, yes on the gated one.** Playing
the best afterstate value would overrule the policy on nearly half of all
turns, almost always on a difference below the value head's own training
scale. With a one-bin minimum gap it acts on one turn in fifteen instead. Read
[§ U4](#u4--what-the-afterstate-values-said-2026-08-16) before building on it.

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
`RERANK_RESERVE_FLOOR_MS = 25`, `RERANK_MIN_GAP = 2/127`,
`CASTLE_SURCHARGE_CAP = 8`, `CASTLE_LATE_TURN = 650`.

**The milestone measured its own constants wrong twice and fixed both.** The
first build ran `TOP_K = 5` against a *fixed* 25 ms reserve and overran the
turn limit; the reserve is now measured and `TOP_K` is 4
([§ The reserve was a guess](#the-reserve-was-a-guess-and-the-guess-was-wrong)).
The second had no minimum gap and would have overruled the policy on half of
all turns; it now has one
([§ The threshold](#the-threshold-and-what-it-changes)). Numbers below say
which build they came from, because the intermediate ones are the evidence for
the constants that replaced them.

Across all three builds: every gate match reached a normal end on the **same
turn with the same winner** as its U2 and U3 run, which is what a shadow
mechanism must produce. Seed 3 re-ranks 846 of its 847 turns — the missing one
is turn 846, where the U3 kill proof overrode and the turn ended before a slate
was built.

### The go/no-go: no on the design as written, yes on the gated one

The plan's bar is that "live re-rank requires candidate value gaps that stand
clear of the near-zero mass of the gap distribution". There is a natural unit
to measure that in, and it decides the question.

joe's value head is a distribution over **128 bins spanning [−1, +1]**
(`bin_centers` in the artifact), so one bin is **0.0157** wide. The head's
output is a continuous expectation over that softmax, so a smaller difference
is representable — what it is not is *evidence*. The head was trained against
a 128-bin target, so bin width is the scale at which it learned to separate
one outcome from another, and a gap an order of magnitude below that is the
softmax leaning between two adjacent bins rather than a preference between two
positions.

Measured against that unit, the re-rank **as originally specified** — play the
best value, no threshold — fails. From the second build, `TOP_K = 4` with the
measured reserve:

| Sample | Turns | Rival preferred | Gap ≥ 1 bin | Gap ≥ 2 bins |
| --- | --- | --- | --- | --- |
| gate seed 0 | 770 | 49.5% | 8.6% | 3.0% |
| gate seed 1 | 659 | 56.9% | 13.7% | 7.0% |
| gate seed 2 | 624 | 41.3% | 2.7% | 0.6% |
| gate seed 3 | 846 | 42.2% | 4.7% | 1.4% |
| corpus replay | 1,140 | 45.6% | 8.9% | 3.7% |
| **all** | **4,039** | **46.8%** | **7.8%** | **3.1%** |

It would take the move away from the network on nearly half of all turns, and
on **83% of those** on a difference below the head's own training scale. H2 as
written is **not supported**.

### The threshold, and what it changes

So the layer now carries one: `RERANK_MIN_GAP = 2/127`, one bin, and a rival
takes the turn off the policy only by clearing it. Everything short of that is
a tie, and a tie belongs to the policy — which is right far more often than a
value comparison at this scale. Re-measured with the threshold live:

| Game | Turns | Forwards/turn | Rival preferred | Gated change | Gap p50 | p90 | p99 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gate seed 0 | 770 | 2.93 | 312 (40.5%) | 47 (6.1%) | 0.0000 | 0.0110 | 0.0340 | 0.0849 |
| gate seed 1 | 659 | 3.00 | 337 (51.1%) | 80 (12.1%) | 0.0001 | 0.0187 | 0.1088 | 0.2959 |
| gate seed 2 | 624 | 3.14 | 251 (40.2%) | 15 (2.4%) | 0.0000 | 0.0045 | 0.0235 | 0.0690 |
| gate seed 3 | 846 | 2.95 | 358 (42.3%) | 40 (4.7%) | 0.0000 | 0.0072 | 0.0328 | 0.2157 |
| corpus replay | 1,138 | 2.53 | 436 (38.3%) | 83 (7.3%) | 0.0000 | 0.0122 | 0.0677 | 0.3935 |
| **all** | **4,037** | **2.87** | **1,694 (42.0%)** | **265 (6.6%)** | | | | |

The threshold refuses **84%** of the value head's preferences. What survives is
a mechanism that acts on one turn in fifteen, each time on a gap the head's
training scale supports, with a tail running to 2–25 bins. The gap column is
deliberately still the *ungated* one — best rival against the argmax, recorded
whether or not it cleared — so the distribution the threshold is drawn from
keeps being measured now that the threshold exists.

Three repeat runs of the corpus give 45.4%, 45.4%, 45.8% preferred and 8.8%,
8.9%, 8.9% gated: the behaviour is reproducible even where the timing is not.

**The verdict is a conditional go.** The gated re-rank clears the plan's bar —
there is mass standing clear of the near-zero pile, and the mechanism now only
touches it. What the shadow data cannot say is whether it *helps*: a gap of one
bin means the head distinguishes those two positions, not that it distinguishes
them correctly. That is a rating question and it belongs to U6, as an arm
behind `UNCLEJOE_RERANK` rather than as part of the headline contrast.

One coupling to carry forward: **the reserve caps how often the threshold can
act.** It ratchets on host jitter (below), and a run that settles at 77 ms
evaluates 2.5 candidates per turn where one settling at 46 ms evaluates 3.1.
Fewer candidates evaluated is fewer chances for one to clear a bin — 6.1% to
12.1% across the five runs. The mechanism's firing rate is therefore a property
of the host as much as of the position, which is an argument for measuring it
on the proxy before U6 sizes anything.

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

`bench` over the corpus prefix, three interleaved passes on the same host
(ms). Interleaved because this host's tail moves more between two runs of the
*same* binary than between two binaries, and a single draw would report that
jitter as a result:

| Pair | joe-rs p99 / max | `UNCLEJOE_RERANK=0` p99 / max | re-rank live p99 / max |
| --- | --- | --- | --- |
| 1 | 25.02 / 50.49 | 41.70 / 57.73 | 100.53 / 143.34 |
| 2 | 28.09 / 44.75 | 25.38 / 45.36 | 111.03 / 146.50 |
| 3 | 24.86 / 44.12 | 24.55 / 29.00 | 113.67 / 120.89 |

For comparison, the **first** U4 build — fixed 25 ms reserve, `TOP_K = 5` —
measured p99 143.26, max 181.49 on the same corpus.

Every pass is inside 150 ms, and the p50 sits at 89 ms in all three: the loop
ends where the internal deadline says it should. The 121–147 ms spread in the
max column is this host's own jitter, not the mechanism — joe-rs alone swings
44 to 50 ms on a 22 ms median in the same window, so roughly 25 ms of tail
belongs to the machine and rides on top of whatever the bot does. That eats
most of the 20 ms of slack the plan set aside between the 130 ms internal
deadline and the judge's 150.

Which is why the proxy measurement is not a formality. On the Modal one-core
x86-64-v3 proxy joe-rs measures p99 23.7 / max 34.5 — about 11 ms of tail
against this host's 25 — so the same loop should land with the slack intact
there. U5 has to show that rather than assume it.

`TOP_K` changed for the same reason the reserve did. Five afterstates plus the
live forward is ~138 ms before any rendering, so the first pass cut itself off
on 4,008 of 4,037 turns and averaged 3.9 — the anytime cutoff was quietly
supplying the real constant. Four is written down instead.

Four is still not always reached: these runs average **2.9** forwards, and the
reserve settled between 46 and 77 ms where a typical evaluation costs ~23. That
is the reserve ratcheting on host jitter — it keeps the worst evaluation ever
seen and never decays, so one stall pins it for the rest of the game.
Deliberately so, in the conservative direction; a reserve set to the typical
cost is precisely the bug this section is about. But it has a behavioural
price, noted above: fewer evaluations means fewer chances for a rival to clear
the threshold. Whether a decaying bound buys the candidates back without giving
back the guarantee is a U5 question, and it belongs on the proxy, where the
stalls are smaller.

**The gate matches still say nothing about timing.** `matchup.py` has no
per-move time limit; the judge does (RULES.md §08). The bench above is the
evidence, and U5's proxy run is what a rated round waits on.

### The masks are cheap, correct, and almost never load-bearing

Across 4,037 turns the filters removed **37** candidates: 24 crowding, 1
lateness, 12 refutation. They removed the *argmax* — the only case where a
mask can change the played move — on **9** turns, 0.22% of them. Three turns
filtered every candidate and fell back to the argmax, which is the intended
degradation. The counts are identical across all three builds, as they must
be: a filter reads the frame, and the frames did not change.

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

The Rust suite is 106 tests, all passing — 32 new: 10 on the afterstate
renderer, 8 on the filters, and 14 on the slate, the score order, the anytime
guard, the measured reserve and the minimum-gap threshold. The Python suite is
untouched at 15.2 s.
