# How the top leaderboard bots lose — Kubic, bca, thor

> Verdict: **all three lose the same way, at different speeds.** The winner
> establishes a land-and-army lead by mid-game, the loser enters a permanent
> `expansion_stall` while the winner keeps gaining, and the general falls out
> of the attrition — usually without the loser ever sighting the winner's
> general (75% of losses) and usually without the winner needing to sight the
> loser's until the final blow. The winners do not gather more or aim at the
> general more; they expand harder and press somewhat straighter
> (largest-stack toward-fraction median 0.86 vs the losers' 0.80 across the
> 36-game drill-down, reaching 90–95% in the cleanest grinds).

Date: 2026-08-11. Observational analysis of scraped `generals.bot` replays
under `competition-replays/` — **read-only; nothing here enters `data/games/`,
`data/ratings/`, or any rating fit** (see
[leaderboard-replays.md](../../engine/leaderboard-replays.md)).

Per-player profiles:
[Kubic](leaderboard-losses-kubic.md) ·
[bca](leaderboard-losses-bca.md) ·
[thor](leaderboard-losses-thor.md).
Morpheus-rs mapping: [morpheus-rs-leaderboard-gaps.md](morpheus-rs-leaderboard-gaps.md).

## Method

```bash
python scripts/replay.py batch <player> --outcome lose --json   # flaw aggregates
python scripts/replay.py batch <player> --outcome win  --json   # win contrast
python scripts/replay.py full  <player> <id> --every 8 --json   # per-game drill
```

Outcomes are derived from `Replay.outcome` (replay `winner` + name-resolved
seat), never the folder. No forfeits appeared in these three players' batches;
4 thor losses ended without contact and are treated as bot failures, not play.
The drill-down sample is 12 losses per player, evenly spaced across the
game-length distribution (ids listed in the per-player files) — 36 games total.

## Corpus (batch-derived, played games)

| player | wins | losses | loss median ticks | win median ticks | never saw enemy general in losses |
| --- | ---: | ---: | ---: | ---: | ---: |
| Kubic | 3,536 | 104 | 348 | 252 | 78/104 (75%) |
| bca | 919 | 135 | 577 | 279 | 98/135 (73%) |
| thor | 1,342 | 217 | 440 | 324 | 162/217 (75%) |

("Saw general" is 100% in wins by construction — capturing it puts it in
vision — so only the loss-side number carries information.)

## The pecking order

70–81% of each player's losses come from a six-bot cluster; the tail almost
never beats them. Head-to-head (rows lose to columns):

| W–L | ResBot | Kubic | nanomena | bca | thor | Mattz |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Kubic | 25–44 | — | 166–7 | 166–12 | 181–7 | 75–3 |
| bca | 9–33 | 2–21 | 24–27 | — | 21–9 | 25–5 |
| thor | 6–54 | 3–37 | 41–49 | 29–25 | — | 19–10 |

ResBot sits on top (99.3% played win rate in its own corpus —
[resbot-evidence.md](../../bots/morpheus/resbot-evidence.md)), then Kubic,
then nanomena, then thor/bca.

## Two loss archetypes

- **The grind** (Kubic 41%, bca 72%, thor 49% of losses at ≥450 ticks, mostly
  vs ResBot): both sides gather constantly, but the winner holds a modest,
  *sustained* land lead — median 1.24–1.48×, army 1.41–1.83× at the last
  contested frame — and keeps expanding through the fighting while the loser
  stalls. `expansion_stall` fires for the loser in 29 of the 36 sampled
  losses; the extreme is bca stuck ≤94 tiles for **784 ticks** while ResBot
  grew (bca 153167).
- **The punch** (Kubic 31%, bca 14%, thor 24% at ≤250 ticks, mostly vs
  nanomena/Kubic): parity at first contact, then the winner lands one bigger,
  earlier gather wave, breaks the line, and drives at the general *before
  sighting it* — thor 153055 goes from tile parity at t96 to general captured
  at t171.

## When the loss becomes irreversible

Across the 36 sampled losses: first contact at median t76; the sustained
1.25× land divergence starts at median t300; the last tick the loser held
≥80% of the winner's army is median **t352** of a median 475-tick game. The
final quarter (and in ResBot grinds, the final several hundred ticks) is
already decided. No sampled loss shows a successful comeback after a
land-stall plus 1.5× army deficit.

## What the winners do differently (and don't)

Same in both seats: gather-wave counts (median 6.5–7.5 loser vs 7.5 winner in
the sample), castle counts and first-castle timing (~t120–140 both sides).
Different:

1. **Sustained expansion through contact.** Winner land keeps rising during
   fights; the loser's flatlines. ResBot's pre-contact curve is metronomic
   (25 tiles at turn 50, p10–p90 24–25).
2. **Straighter pressure.** Winner largest-stack toward-fraction median 0.86
   (p25–p75 0.80–0.90) vs loser 0.80 (0.75–0.84) across the sample; 95% for
   ResBot in bca 153167, 90% for thor in Kubic 152219. A modest but
   consistent edge — and a negative result worth keeping: **gather-wave
   end-distance does not separate the seats** (winner median 4, loser 2), so
   "waves that land on their target" is not what distinguishes winners here.
3. **No dependence on general intel.** Winners sight the losing general at
   median 0.4–0.6 of game length — often 2–3 ticks before capture (ResBot in
   bca 153167: first sight t1129, capture t1131). Economy wins the game;
   the sighting is a formality. Conversely the *losers'* blindness is real:
   their closest approach is median 3–4 steps and they never convert.

## Sampling caveats

- Win and loss folders face **different opponent pools**: wins are dominated
  by tail bots, losses by the top cluster. Win-vs-loss behavioral contrasts
  confound opponent strength with behavior; the honest contrast is the
  within-cluster one (per-player files).
- The list endpoint serves a recent window with no pagination; the corpus
  accumulates only by re-scraping, so it over-represents recent form.
- Replays store state, not actions — every event is inferred from frame
  diffs ([replay-analysis.md](../../engine/replay-analysis.md)).
