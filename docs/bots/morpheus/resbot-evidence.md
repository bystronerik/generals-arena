# ResBot replay evidence

This file contains observations only. A regularity is a target or evaluation
signal for Morpheus. It is not evidence about ResBot's implementation.

## Corpus

The current directory has 1,158 replay files, not the expected 2,140. The
replay tool skips 6 one-tick forfeits and reports 1,152 played games:

- 1,144 ResBot wins;
- 8 ResBot losses;
- 0 draws.

The played win rate is 99.3%. Only the 8 available losses are analyzed. Claims
about 14 losses cannot be made from the current files.

Whole-corpus command and sample:

```bash
python scripts/replay.py batch ResBot --outcome all --json
```

`n=1,152` played games, with derived outcomes rather than folder names.

Detailed commands:

```bash
python scripts/replay.py summarize ResBot <match_id> --every 1 --json
python scripts/replay.py events ResBot <match_id> --json
python scripts/replay.py flow ResBot <match_id> --json
```

The detailed sample is 96 wins at evenly spaced positions in the sorted list
of 1,144 wins, plus all 8 losses.

## Expansion

The 96-win sample has this median land curve:

- turn 25: 7 tiles, with p10-p90 also 7-7;
- turn 50: 25 tiles, with p10-p90 24-25;
- turn 100: 55 tiles, with p10-p90 47-60.

First contact occurs at median turn 82 in the detailed wins. ResBot has median
44 tiles, an 8-tile lead, a 3-army lead, and a largest stack of 14 at contact.

The whole 1,144-win batch gives median contact at turn 83. The narrow turn-50
range is the strongest expansion regularity in this sample.

## Commitment and contact

In 88 detailed wins with a detected gather wave, the first wave starts at
median turn 73. The first wave starts:

- before contact in 45 games;
- after contact but before general sight in 41 games;
- after general sight in 2 games.

All 96 detailed wins make an enemy capture after contact. The median delay is
9 turns, with p25-p75 of 1-21 turns.

The whole win corpus has median 3 gather waves and median peak stack 100.
The detector finds no gather wave in 103 of 1,144 wins, so a gather wave is
common but not required by this measurement.

## Castle timing and siting

The event command infers castles from repeated production. It stamps the first
production, not the build action.

In the 96 detailed wins:

- 39 games have at least one ResBot castle candidate;
- the tool finds 54 candidates;
- first production is at median turn 126, with p25-p75 of 122-134;
- first-site distance from ResBot's general is median 7, with p25-p90 of 7-7;
- only 2 first candidates occur before contact.

These are production-based candidates. The build action can be one or two
turns earlier. State-only replays and combat can make an exact spend turn
ambiguous.

## Fog contact and conversion

The whole batch reports enemy-general sight in all 1,144 wins. Seven of 8
losses never sight the enemy general.

In the 96 detailed wins, first sight is at median turn 317. The game ends a
median 2 turns after sight. In the 67 wins with a directed post-sight largest
stack move, the median fraction toward the known general is 94%.

The two-turn result does not mean ResBot usually starts a new attack after
sight. A move can reveal and capture the general at almost the same time.

## Loss cases

The 8 losses have the same median early land curve as the win sample:
7 tiles at turn 25 and 25 at turn 50. They contact at median turn 80 with 45
tiles, an 11-tile lead, and a 3-army lead.

The loss batch has median 1 gather wave and peak stack 51, compared with 3 and
100 in all wins. Seven losses never sight the enemy general. The one sighting
occurs at turn 350, but ResBot does not finish during the next 191 turns.

The loss sample is too small for stable thresholds. It supports two failure
cases only: failure to locate the general, and failure to convert one late
sighting.

## Limits

- The leaderboard window is selected, recent, and not paginated.
- Opponent strength is unknown and uneven.
- Six forfeits are excluded from played behavior.
- Events infer actions from state changes; the replay has no action log.
- `toward_fraction` targets visible enemy land before general sight.
- Final captured-state land totals are not useful; batch uses the last
  contested frame.
- The samples cannot identify a network, search method, reward, or other
  ResBot internal.
- These replays never enter arena games or ratings. See
  [leaderboard replay limits](../../engine/leaderboard-replays.md).
