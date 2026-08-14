"""
Arena ratings: a batch Bradley-Terry + Davidson + seat fit, not sequential Elo.

The system exists to answer one question — **did this bot change make the bot
stronger?** — and everything here follows from what that needs:

- **Order independence.** The likelihood reads games only through integer
  counts over ordered pairs, and the penalized objective is strictly convex, so
  the answer does not depend on the order matches finished in. Sequential Elo
  moved ratings by up to 250 Elo across shuffles of the *same* games.
- **Identity is the content hash.** Entities are `bot_id@content_hash`, so two
  revisions of one bot are as unrelated to the fit as two different bots. That
  is what makes a lineage delta a measurement rather than a restatement.
- **Uncertainty is first class.** The covariance matrix is an output, not an
  afterthought: `P(B > A)`, `CI95`, and `games_to_resolve` all come from it.
- **Draws are modelled.** A third of all games are 1200-turn truncations.
  Splitting them half-and-half compresses the draw-heavy tail ~3x, exactly
  among the bots that draw with each other.

- **A round is the unit.** Each round gets its own independent fit; there is no
  pooled table. Two byte-identical programs measured in different rounds fitted
  46 Elo apart, which is more than the decision rule's own thresholds, so a
  pooled column reported drift as strength.

There is no incremental update. Every write refits every round.

Module map:

    policy.py    eligibility rules, Policy and Prior
    counts.py    GameRecord[] -> canonical integer CountTable
    model.py     the likelihood, gradient and Hessian (pure math)
    fit.py       damped Newton -> RatingFit (estimates + covariance)
    rounds.py    one fit per round -> RoundResult / RoundFits
    scan.py      walk data/games/ -> one RoundCounts per round
    io.py        fits/<round>.json / leaderboard.{json,md}
    cli.py       python -m arena.records.ratings
"""

from __future__ import annotations

from arena.records.ratings.counts import Cell, CountTable, count_table, merge
from arena.records.ratings.fit import (
    Delta,
    Estimate,
    LoadedFit,
    RatingFit,
    fit_ratings,
)
from arena.records.ratings.io import (
    RATINGS_DIR,
    LeaderboardRow,
    leaderboard_markdown,
    leaderboard_rows,
    load_fit,
    split_entity,
    stored_counts_digest,
    stored_rounds_digest,
    write_all,
    write_leaderboard,
    write_round_fits,
)
from arena.records.ratings.model import ELO_SCALE
from arena.records.ratings.policy import Policy, Prior, eligible, entity_key, rejection_reason
from arena.records.ratings.rounds import (
    RoundFits,
    RoundResult,
    fit_rounds,
    resolve_round_anchor,
    scale_id,
)

__all__ = [
    "Cell",
    "CountTable",
    "Delta",
    "ELO_SCALE",
    "Estimate",
    "LeaderboardRow",
    "LoadedFit",
    "Policy",
    "Prior",
    "RATINGS_DIR",
    "RatingFit",
    "RoundFits",
    "RoundResult",
    "count_table",
    "eligible",
    "entity_key",
    "fit_ratings",
    "fit_rounds",
    "leaderboard_markdown",
    "leaderboard_rows",
    "load_fit",
    "merge",
    "rejection_reason",
    "resolve_round_anchor",
    "scale_id",
    "split_entity",
    "stored_counts_digest",
    "stored_rounds_digest",
    "write_all",
    "write_leaderboard",
    "write_round_fits",
]
