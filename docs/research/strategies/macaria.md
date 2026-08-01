# macaria — blitz's policy, with a move search allowed to overrule it

Spec, written before the search. Bot doc: [`../../bots/macaria.md`](../../bots/macaria.md).
Round report: [`../measurements/macaria-r1.md`](../measurements/macaria-r1.md).

## 1. The claim

`bots/blitz` is currently the strongest entity in the pool
(`blitz@b4a69aad6389`, 2499.5 [2458, 2540] over 1883 games). macaria takes
blitz's policy **unchanged** and adds one thing: on the turns where a
heuristic ladder is most likely to be locally wrong, spend some of the move
budget searching for a better move, and play it instead.

> **r2 amendment.** "Unchanged" describes r1 and the r1 measurement only. The
> vendored core now also carries the general hunt — see §7 — which is a change
> to blitz's targeting, not to the search. Read every claim below about the
> core being blitz as scoped to r1.

The hypothesis is therefore narrow and falsifiable:

> A heuristic that picks moves by priority ladder gives up measurable strength
> at the *tactical* moments — the ones where the right move depends on how a
> two-sided exchange resolves over the next handful of plies — and a search
> that fires only there recovers part of it, without touching the strategic
> policy that makes blitz strong in the first place.

If that is wrong, the failure is visible and specific: macaria and blitz come
out flat, and the probe says the search fired and overrode moves. That is a
different result from "the search never ran", which is why the probe reports
`searched` and `overrode` as a pair.

## 2. Why the core is vendored, not imported

`bots/macaria/blitz_core.py` is a copy of `bots/blitz/agent.py`, taken at
content hash `b4a69aad6389`, and verified move-for-move identical to it
(6 seeds against `cm_expander`, same capture turn on every one).

blitz is the **A arm of the contrast that decides this bot**. Editing it —
even to add a hook, even to make a constant configurable — moves its content
hash, and the rated entity `blitz@b4a69aad6389` that the baseline refers to
stops being the program the baseline was measured on
([`../../arena/decision-rule.md`](../../arena/decision-rule.md), *connectivity*).
A vendored copy makes that mistake unreachable: nothing macaria does can
touch the baseline.

The cost is honest duplication: a future fix to blitz does not reach macaria.
That is the correct trade here, because the two are being *compared*, not
composed.

## 3. Every constant in one place

`bots/macaria/params.py` holds every behavioural constant of both halves in
one frozen dataclass, including the ones upstream blitz kept as module
constants (`DEATHTOUCH_TURN`, `CHASE_DEFEND_FROM`, the never-reserve strategy
context, the `or 50` wave period, the `// 2` general regen). `blitz_config()`
translates back into the core's own config type, so the core keeps its own
vocabulary.

`MACARIA_TUNE` (JSON, environment) overrides fields at construction, so a
sweep varies one knob without forking a content hash per cell. It is
research-only: unset — which is what the competition runner and every stored
game does — the shipped defaults are the whole program. Malformed input,
non-dict input and unknown keys all fall back to the defaults rather than
raising, because RULES.md §08 forfeits a bot that crashes and a sweep typo
must not be able to spend a game.

## 4. The turn budget

RULES.md §08 allows 150 ms per move and forfeits at 50 late or malformed
replies. macaria spends at most **100 ms on the whole move** and keeps 50 ms
as reserve.

- The heuristic core runs first and is not interruptible.
- The search's deadline is what is left of the 100 ms cap after the core has
  been paid, further capped by the search's own budget — so a slow core
  shrinks the search instead of pushing the turn over.
- The deadline is an absolute `time.monotonic()` timestamp, checked **inside
  the rollout loop**, not only between iterations: one long rollout is enough
  to miss a turn.
- On expiry the search returns the core's move, which is already computed and
  already legal. A legal reply always exists.

The local `matchup.py` does **not** enforce §08's timeout — it blocks on
`readline` — so "no faults locally" proves nothing about latency. Latency is
therefore measured directly, per move, from the probe channel.

## 5. What the search is not allowed to be

- **Not opponent-specific.** No parameter is fitted to one opponent's
  behaviour.
- **Not a second strategy.** It re-picks a move inside the core's plan; it
  does not decide to expand instead of attack.
- **Not free of the core's state.** The core mutates its own memory *while*
  choosing (it records the strike stack's destination before returning the
  push). An override that ignores that leaves the core planning from a game
  that did not happen — see `search.py`.

## 6. How it will be decided

Per [`../../arena/decision-rule.md`](../../arena/decision-rule.md), from the
pairwise rating contrast after a refit — never from rank.

| | |
| --- | --- |
| baseline `A` | `blitz@b4a69aad6389` |
| candidate `B` | `macaria@<hash>` |
| panel | `cm_expander` (anchor, 1500), `cm_hunter` (1925), `metro` (2027), `aegis` (2195), `boom` (2343) |
| grid | 50 games/pair, `--round-seed 7`, `--seat-policy alternate`, `--strict-versions` |
| per arm | 300 games, 50 per opponent |

The panel spans ~1000 Elo and shares games with both arms, so the two arms are
connected by construction.

Reported alongside: the direct macaria-vs-blitz head-to-head with its interval,
and each panel opponent separately, so a pooled gain cannot hide a regression
against one of them.

**Control arm.** `MACARIA_TUNE='{"mcts_enabled": false}'` is macaria with the
search off — at r1 that was the vendored core alone, i.e. blitz. Run on the
same map seeds as the live arm, it separates the search's contribution from the
core's. It is stored **outside `data/games/`** and rated by winrate only, never
pooled: it plays under macaria's content hash while being a different program,
and pooling it would merge two programs into one rated entity.

## 7. r2 — the general hunt

The claim in §1 is about *tactics*. This is a separate, strategic defect,
measured on 318 scraped leaderboard games rather than argued from the design:

> In **97 of 127 losses macaria never once had vision of the opponent's
> general.** Median closest approach of any owned tile to it: 3 in losses
> against 0 in wins; median peak stack 49 against 67. In the canonical game
> (match 15932) its closest approach was 8 steps, reached at tick 36 during the
> opening and never again, and all five of its gather waves were aimed at
> incidentally visible enemy tiles.

Blitz's targeting reads "where the enemy lives" off `visible_enemy_tiles`. In a
game it is losing, that quantity sits inside its *own* half — the opponent's
forward tiles — so `enemy_anchor` (which drives the expansion bias) and
`contact_target` (which drives the wave) both point home, and the bot orbits
its own territory. The mechanism, the four call sites changed and the knobs are
in [`../../bots/macaria.md#the-general-hunt`](../../bots/macaria.md).

Falsifiable the same way §1 is, and on the same axis the defect was measured
on: `blitz_hunt_enabled=false, blitz_expand_bias_first=false` is upstream
targeting under the same content hash, so the contrast is the fraction of games
in which the general is ever sighted with the game still live, not just the
winrate. If the hunt is wrong, that fraction moves and the winrate does not —
which is a different failure from "the hunt never fired".

### Verdict: the third case, which the prediction above did not name

2,624 games against `macaria_base@862ac0189a1a`: **+18.82 ± 12.41 Elo,
P(B>A)=0.935 — unproven.** Vision moved (net +16/+28/+34 paired games newly
sighted with 20/50/100 turns left) and the winrate did not, which is the
"hunt is wrong" branch above. But the reason is the one nobody wrote down:
**vision is not the lever here.** Only 10% of arena games go unsighted against
49% of leaderboard losses; in the unsighted ones the *old* estimator was
already a median 6–8 hops from the true general; and those games are even
grinds (land margin ≈ 0) where the frontier stalls 5 hops out against a
defended general, not games lost to looking in the wrong place.

The generals.bot sample that motivated §7 is a different population — a
different opponent pool, and losses only. Its 76%-unsighted headline is not a
property of macaria that this panel reproduces.

Numbers, the per-rung replay analysis, and the one r2 revision it justified
(`hunt_drives_anchor`, default off) are in
[`../../bots/macaria.md#what-the-arena-said-about-it`](../../bots/macaria.md).

Not re-run as a measurement round after r2: the sanity sweeps in the
implementing session are **not** the contrast. §6 still governs.
