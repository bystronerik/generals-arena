# Training

## Decision

Morpheus uses AlphaZero-style generalized policy iteration under the exact
competition rules. Self-play search produces policy targets, and terminal game
outcomes produce value targets.

The game reward is:

```text
nonterminal = 0
win = +1
draw = 0
loss = -1
discount = 1
```

Land, army, castles, sight, and game length are not reward terms.

## Sparse-reward solution

The old learned-bot placeholder records 31 games and 31 turn-1200 draws. Raw
WDL from full starts would give almost no early learning signal.

Morpheus keeps WDL unchanged and uses a backward curriculum of **reachable**
states:

1. States one tactical sequence from a general capture or defense.
2. States after enemy-general sight.
3. Contact states with both generals alive.
4. Pre-contact states from later and later distances.
5. Full competition starts.

Every curriculum state comes from a legal trajectory produced by
`GeneralsEnv(mode="competition")`. Training does not invent board states,
remove fog, shorten the board, or change deathtouch.

Each curriculum item is a seed plus an action prefix, not a bare state grid.
Training replays the prefix from turn 0 to reconstruct both seats' observations,
persistent memories, particle beliefs, and action history before it starts the
sampled continuation.

The scheduler samples state classes that currently produce both wins and
losses. It moves weight toward earlier classes only after later classes keep a
non-degenerate WDL target. The exact promotion interval is an
[open question](open-questions.md), not an invented threshold.

## Opponent mixture

The initial game mixture is:

- 70% checkpoint league;
- 30% fixed existing-bot panel.

This split is an **initial guess**.

The checkpoint league contains the current learner, the promoted best
checkpoint, recent snapshots, and exploiters selected for high loss rate.
Snapshots are immutable during one training epoch.

The fixed panel contains competition-mode heuristic and research bots from
`bots/`. Sampling gives more weight to uncertain matchups and opponents that
produce decisive games. It excludes `classic_duel` and every remote classic
result.

Seat assignment and map generation are random. Training includes all 18-21
rectangles and both seat orientations.

## Targets

For each played state:

- policy target: normalized root average strategy `S_A`;
- value target: final WDL from that seat's perspective;
- hidden owner, army, general, and castle targets: engine truth;
- final margin and termination targets: engine truth.

Auxiliary targets train representation and particle proposals. They never
enter reward or arena scoring.

Recorded arena trajectories remain actions plus seed. Dense states are
materialized on demand through the existing trajectory path. Training adds no
field to the game-record schema.

## ResBot evidence

ResBot data is not behavior-cloning data. The replays store states rather than
unambiguous actions, and the opponents are selected with unknown strength.

The measured expansion, contact, gather, castle, and sight distributions in
[resbot-evidence.md](resbot-evidence.md) serve two purposes:

- behavioral diagnostics on held-out leaderboard replays;
- curriculum coverage alarms that increase sampling of a weak state class.

They do not add reward, policy labels, value labels, or hidden-state labels.
Morpheus can depart from every ResBot range when arena results improve.

Direct shaping is rejected because it can optimize leaderboard correlations
instead of wins. It can also freeze ResBot's unknown opponent mix into the
policy and punish a better strategy that uses different timing.

## Search noise and exploration

Training adds root prior noise and samples from the root average strategy.
Rated play uses neither root noise nor action temperature.

Noise concentration, temperature, and the turn at which sampling becomes
deterministic are deliberately deferred to measured training stability.

## Checkpoint promotion

A candidate first passes WDL, legality, belief calibration, and runtime checks
on held-out seeds. It then plays stored competition arena games against a
frozen panel in both seat orientations.

Promotion uses the arena pairwise rating contrast and interval, not training
loss, ResBot similarity, raw rank, or one self-play win rate.

## Alternatives rejected

Permanent land or army shaping is rejected because a turn-1200 score is still
a draw. Perfect-information pretraining is rejected because it can teach the
policy to act on unavailable enemy state.

Training only against the current checkpoint is rejected because simultaneous
games can cycle. Training only against the heuristic roster is rejected
because it caps adaptation at a fixed opponent distribution.

## Failure mode

The curriculum can overrepresent short tactical states and fail from full
starts. Full-start arena games remain the promotion authority. Auxiliary
hidden-state losses can also dominate policy learning, so their weights require
an ablation before they become fixed values.
