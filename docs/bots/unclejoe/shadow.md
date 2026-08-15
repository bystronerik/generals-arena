# unclejoe shadow measurements

What the tactics layer *would* have done, measured while it did nothing. The
milestones that end in a shadow report write their sections here:
[tactics-plan.md](tactics-plan.md) §7 — U2 (triggers) below, U3 (what the
searches behind those triggers proved) after it, U4 (afterstate re-rank)
later.

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
