# 052 — sosipolis vs macaria: two gates that outrank the attack

Bot: [`bots/sosipolis/`](../../../bots/sosipolis/) — follows 051.

Prompted by a GUI observation on seed 1: the bot gathers an army, starts an
attack, then walks back over its own one-army cells without ever reaching the
enemy — and is not gathering a bigger stack while it does. It "looks
disoriented". Both halves of that turned out to be one shape: **the priority
shell has several rungs above the assault that arm on presence and never
release**, so the turn is spent walking the biggest stack backwards.

## Baseline correctness note

The 20-seed grid is **deterministic**: two consecutive runs of the committed
bot produced the same winner *and* the same turn count on all 20 seeds. The
baseline is **5/20**, not the 7/20 recorded in 051 — that figure came from a
different session and does not reproduce. Every number below is the same
2-minute grid, so a changed win set is signal, not noise.

## What was measured

Per-turn branch mix, sosipolis as seat a, turns past t50:

| seed | result | `recall` | `castle` | `mcts` |
| --- | --- | --- | --- | --- |
| 1 | loss t340 | 3 | **86** | 147 |
| 5 | loss t619 | **124** | **119** | 304 |
| 7 | loss t268 | **148 of 218** | 4 | 64 |

Seed 7: `recall` fires at t121 and then on **every remaining turn** — 148
consecutive. Seed 5's longest run is 86.

### Gate 1 — `recall_armed` armed on presence

It returned true whenever *any* visible enemy tile sat within `RECALL_PROX_D`
(3, manhattan) of our general. Enemy land near our general is permanent — they
do not leave — so the gate latched the first time macaria captured a cell there
and never released. `recall_move` then walks the tip home and shuffles there.
It also measured only the **nearest** tile, so a one-army captured cell masks a
40-stack one step further out.

Fixed by arming on what the stack can do: the biggest enemy stack within the
radius must satisfy `army - 1 >= general_army` (attacking spends one unit to
leave). The truly adjacent lethal case is already handled one rung higher by
`imminent_loss_move`.

### Gate 2 — the castle programme outranked the hunt

`Economy.decide` sits above both the tip feed and the march, and funds a site
with `gather_toward`, which scores `army * 100 - dist` — mass-first, so the
stack it reaches for is **always the assault tip by construction**.

Seed 1, turns 217–224: eight consecutive `castle` turns walked a **47-army tip
from 8 cells off the objective back to 15, arriving at 27**. Two castles built,
the first at t223, game lost at t340. That is precisely the reported symptom.

A castle costs 35–47 army and repays at one unit every *other* turn, so it needs
~80 quiet turns to break even. Fixed by making castles a pre-contact investment
only: `Economy.decide` returns `None` unless `phase == "search"`.

## Results (20-seed grid, one game per seed)

| build | wins /20 | seeds |
| --- | --- | --- |
| committed baseline | 5 | 3 9 13 14 19 |
| recall fix alone | **3** | 3 8 13 |
| castle gather skips the tip | 4 | 8 9 14 18 |
| `CASTLE_MAX = 0` (castles off entirely) | 6 | 0 3 4 6 9 15 |
| **recall fix + castle phase gate** | **7** | 4 6 8 9 14 16 19 |

`CASTLE_MAX = 0` and the phase gate both score 7 combined with the recall fix;
the phase gate ships because it keeps the subsystem and its tests.

## The finding worth keeping: the gates interact

**The recall fix alone is −2. Combined with the castle gate it is +2.**

The fix did exactly what it was designed to do in isolation — recall went 148 →
10 on seed 7 and 124 → 6 on seed 5 — and the bot lost *faster* (t268 → t220,
t619 → t330). The trace says why: with recall disarmed, `castle` took 80 of 247
contact turns on seed 5. The freed turns went straight into the other rung that
does not attack, and the bot had given up the defence it was getting for them.

So a rung that outranks the assault cannot be evaluated on its own. This also
explains a 051 result: rows 9–11 all tried to make defence or tip-holding
*better* and all lost, because the problem was never the quality of those rungs
— it was that there are several of them and they are all above the attack.

## Where the failure moved

Post-fix traces (contact turns, share of moves advancing on the objective):

| seed | contact turns | toward objective | sighted general |
| --- | --- | --- | --- |
| 1 | 488 | 51% | **no** |
| 5 | 248 | 85% | yes |
| 7 | 236 | 78% | yes |
| 13 | 326 | 60% | yes |
| 18 | 148 | 86% | **no** |

The branch mix is now dominated by `mcts` and the bot moves at its objective on
most turns. The live defect is different: **seed 1 spends 488 turns in contact
and never sights the general**, with a single objective latched the whole time.
That is a belief/hunt failure, not a priority failure, and it is the next thing
to look at.

## Not yet addressed from the same report

Two other observations from the same seed-1 session are untested:

- Our opening expands as a disk; Kubic runs several long lines and extends the
  ones that find something. `capture_move` sorts by walk-from-general, which
  builds a disk by construction. (051 row 12 tried a related idea — biasing the
  same sort toward the objective — and got a wash.)
- Kubic mostly stops exploring after contact and hunts, taking enemy land and
  castles. Our enemy-capture share already matches Kubic's (18% vs 20%, 051),
  so the difference is more likely *where* we hunt than whether we do.

## Reproduction

```bash
for s in $(seq 0 19); do
  python competition-module/competition/matchup.py \
    bots/sosipolis/run.sh bots/macaria/run.sh --mode competition --seed $s
done
```
