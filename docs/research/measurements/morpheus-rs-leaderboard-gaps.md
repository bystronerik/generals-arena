# Morpheus-rs vs the leaderboard winners — behavioral gaps

> Verdict: **morpheus-rs already has the machinery the leaderboard winners
> use — economy-first play with no scripted phases, directed search pressure,
> a belief over the hidden general — but two of the three winner behaviors
> are unverified in the replay-analysis vocabulary, and the one measurable
> proxy (time-to-close) points the wrong way: its arena wins take 323–510
> ticks against a panel it dominates, where leaderboard winners close in
> 252–324 against real opposition.** Ranked gaps below, each tied to replay
> evidence.

Date: 2026-08-11. Companion to
[leaderboard-top-loss-analysis.md](leaderboard-top-loss-analysis.md) (with
per-player profiles for [Kubic](leaderboard-losses-kubic.md),
[bca](leaderboard-losses-bca.md), [thor](leaderboard-losses-thor.md)).
No bot code changed for this analysis.

## What morpheus-rs is, behaviorally

From [search-and-tactics.md](../../bots/morpheus-rs/search-and-tactics.md),
[inference.md](../../bots/morpheus-rs/inference.md),
[belief.md](../../bots/morpheus-rs/belief.md), and the morpheus design docs
it is a parity port of:

- Belief-aware IS-MCTS, ~16–20 simulations per turn at M7 knobs; no fixed
  expansion/castle/strike phases — the search compares all legal actions.
- Mandatory candidate prefix: general captures, enemy-source interactions,
  enemy attacks, frontier expands are always in the matrix.
- 8-particle filter over hidden enemy state with three-stage recovery; the
  *learned* enemy proposal is ablated to uniform
  ([belief-ablation-macaria.md](belief-ablation-macaria.md): no measurable
  benefit, −13 ms/turn).
- Latency is a solved problem relative to the Python sibling: belief update
  plus root inference complete on 100% of turns vs 18.4%, 0/21,000 late
  moves on the x86 host. Current best strength estimate vs Python morpheus
  is **+138 Elo** (M7 round; M6's +425 headline is
  [retracted](morpheus-rs-m6-replication.md)).

Arena evidence (round `morpheus-rs-m6-strength`, 1,152 games against five
heuristic panel bots plus Python morpheus): 979–142–31. Weakest matchup by far is **macaria at 0.557**
(107–85) — the one panel bot whose defining feature is a belief-driven
*general hunt*, itself built from the same leaderboard finding this analysis
reconfirms ([macaria.md](../../bots/macaria.md)). Median win length by
opponent: 323–510 ticks.

## Mapping the winner behaviors

| winning-opponent behavior (evidence) | morpheus-rs status |
| --- | --- |
| Sustained expansion through contact; land keeps rising during fights (ResBot 141→166 across bca's 784-tick stall, bca 153167; stall fires for the loser in 29/36 sampled losses) | **Unmeasured.** No land-curve numbers exist for morpheus-rs; the design has no expansion phase to guarantee it. |
| Straight, directed pressure: toward-fraction median 0.86 vs the losers' 0.80, 90–95% in the cleanest grinds (ResBot 95% in bca 153167; thor 90% in Kubic 152219) | **Plausibly present** via mandatory interaction candidates + search, but toward-fraction never measured. |
| Blind conversion: drive at the general region before sighting it; sight and capture 2–3 ticks apart (ResBot sight t1129/capture t1131, bca 153167; nanomena dgen 9 forty ticks before sight, thor 153055) | **Partially present.** The belief filter tracks exactly this, but nothing measures whether the search cashes it in, and the learned proposal earned nothing vs macaria. The 0.557 macaria score says the hunt contest is our closest fight. |
| Fast close once ahead (win medians 252–324 vs the losers' 348–577) | **Gap.** Panel win medians 323–510 against far weaker opposition. |
| Early-castle discipline: castles ~t120–140, not before contact, sited depth ~7 from own general (resbot-evidence) | **Unmeasured.** morpheus-rs builds median 2 castles/game; siting and timing never analyzed. |

## Ranked behavioral gaps

1. **Slow conversion of a won position.** Proxy is measurable today: median
   win 323–510 ticks vs a dominated panel, where every leaderboard winner
   closes in 252–324 against peers. Long wins are where the losers' comeback
   windows live (bca's grinds), and they cap self-play throughput. Evidence:
   corpus win-length table in the hub file; `data/games/morpheus-rs-m6-strength/`.
2. **Unverified anti-stall expansion.** The single most universal loss
   mechanism in 456 losses (expansion stall while the opponent gains —
   Kubic 153084, 178483; bca 153167, 162256; thor 152274, all 12/12 thor
   samples) is exactly the behavior we have never measured on ourselves.
   ResBot's 25±1 tiles at turn 50 is the benchmark curve.
3. **Blind general-hunt conversion.** 75% of top-bot losses are blind
   (78/104, 98/135, 162/217), and the winners win *without* needing sight.
   We carry a belief posterior the leaderboard bots can only approximate —
   and our worst panel matchup is the one bot that hunts by belief
   (macaria, 0.557). Whether our search converts the posterior into
   approach pressure is unknown; the ablation says the learned half of the
   belief currently buys nothing.
4. **Directed-pressure efficiency.** Winners' largest-stack toward-fraction
   median 0.86 vs losers' 0.80, with the top grinders at 90–95%; ours is
   unmeasured. (Wave end-distance turned out *not* to separate winners from
   losers in the sample — measure toward-fraction, not wave landing.) The
   punch losses show the failure at its starkest: the fast loser is the side
   at 0.67–0.75 while the winner runs 0.90+ (Kubic 152219, 96219).
5. **Contested-castle discipline.** bca lost 153167 partly by re-buying one
   central castle eight times; thor's fresh frontier castle became
   nanomena's forward base in 153055 (built t126, captured t132). We build
   ~2 castles/game; whether we defend them, and whether we walk away from
   sunk-cost castle fights, is unmeasured.

## Loss patterns we likely share — and one we must not

- **Share (risk):** the grind. Nothing in the design guarantees land keeps
  rising through contact, and the slow closes suggest surplus army is not
  being spent forward. Against ResBot-class pressure that is the 1.24–1.48×
  land drift that decides games by t300.
- **Share (risk):** the blind stall — big stacks pointed at the nearest
  visible tile. The mandatory-candidate prefix biases toward *visible*
  interactions, the same target class the losers chase.
- **Must not share:** thor's 4 no-contact, no-op losses. morpheus-rs
  deliberately refuses to start on artifact mismatch and degrades to a
  slow-but-correct bot on a non-FMA build
  ([inference.md](../../bots/morpheus-rs/inference.md)); both failure shapes
  would look exactly like thor 112499 from the outside. The guards exist;
  keep them.

## What to measure next (no bot changes)

Run morpheus-rs vs the panel with `--record`, and reduce the trajectories to
the same vocabulary as `scripts/replay.py`
([trajectories.md](../../arena/trajectories.md)): land at turns 25/50/100,
expansion-stall incidence, largest-stack toward-fraction, wave end-distance,
and dgen-vs-time against the *believed* general. That turns gaps 2–4 from
plausible into numbers, against benchmarks this analysis has already fixed
(ResBot 7/25/55 land curve; winner toward-fraction 0.86+, 0.90+ when
closing; close within ~25 ticks of a decisive army lead).
