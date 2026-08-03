# Morpheus design

Morpheus is a competition-mode research bot. An RL-trained neural network
guides a belief-aware Monte Carlo tree search.

This directory is a strategy specification only. It contains no bot code,
training code, implementation plan, task list, or schedule.

## Relationship to the learned-bot plan

Morpheus **extends** the
[learned-bot plan](../../research/learned-bot-plan.md). That file remains the
general pointer for learned-bot work. These files resolve the strategic choices
that the placeholder left open.

## Objective

Morpheus maximizes expected game outcome: win probability minus loss
probability. It treats a simultaneous capture as a draw and models deathtouch
from turn 800.

The policy has no fixed expansion, castle, or strike phase. The network and
search compare all legal actions on each turn. Hard rules apply only to action
legality, the competition transition, and the reply deadline.

## Design

- [ResBot evidence](resbot-evidence.md) keeps observations separate from design.
- [Observation tensor](observation-tensor.md) defines the exact network input.
- [Belief state](belief-state.md) defines hidden-state inference under fog.
- [Action space](action-space.md) defines policy logits and legal masks.
- [Network](network.md) defines the model family and heads.
- [Search](search.md) defines simultaneous information-set MCTS.
- [Runtime](runtime.md) defines deadline control and degraded operation.
- [Training](training.md) defines reward, curriculum, and opponent sampling.
- [Evaluation](evaluation.md) defines artifact identity and arena entry.
- [Diversity](diversity.md) defines the research axis.
- [Open questions](open-questions.md) lists only deliberately deferred choices.

## Sources

The rules source is [RULES.md](../../../RULES.md). The wire source is the
[competition protocol](../../competition/protocol.md). The arena contract is
the [unified bot API](../../engine/unified-bot-api.md).

Scraped leaderboard games are observational evidence only. They never enter
`data/games/`, `data/ratings/`, or `data/remote_games/`.
