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

There is no incremental update. Every write refits.

Module map:

    policy.py    eligibility rules, Policy and Prior
    counts.py    GameRecord[] -> canonical integer CountTable
    model.py     the likelihood, gradient and Hessian (pure math)
    fit.py       damped Newton -> RatingFit (estimates + covariance)
    lineage.py   registry steps -> per-step deltas
    cache.py     per-round count caches
    io.py        fit.json / leaderboard.{json,md}
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
    write_all,
    write_fit,
    write_leaderboard,
)
from arena.records.ratings.lineage import (
    StepDelta,
    lineage_deltas,
    lineage_table_lines,
    steps,
)
from arena.records.ratings.model import ELO_SCALE
from arena.records.ratings.policy import Policy, Prior, eligible, entity_key, rejection_reason

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
    "StepDelta",
    "count_table",
    "eligible",
    "entity_key",
    "fit_ratings",
    "leaderboard_markdown",
    "leaderboard_rows",
    "lineage_deltas",
    "lineage_table_lines",
    "load_fit",
    "merge",
    "rejection_reason",
    "split_entity",
    "steps",
    "stored_counts_digest",
    "write_all",
    "write_fit",
    "write_leaderboard",
]
