"""
One independent fit per round. There is no pooled fit.

A round is a design: one roster, one seed list, one seat policy, played in one
sitting. Pooling rounds treats their games as exchangeable, and they are not.
Two byte-identical `macaria` programs measured in different rounds fitted
**+46.07 ± 22.01 Elo, P = 0.982** apart, and the M6 `morpheus-rs` round
disagreed with four re-measurements by **7.4 sigma**. Both numbers are larger
than the thresholds `docs/arena/decision-rule.md` decides on, so a pooled
column presented drift as strength. Fitting each round on its own removes the
column rather than annotating it.

What that costs, and what pays for it:

- **Cross-round contrasts stop having an answer.** `fits["round"].delta(a, b)`
  needs both arms inside one round. That is the rule the decision rule already
  states in prose; here the data model cannot express the other thing.
- **Each round has its own scale.** The anchor fixes an additive constant, not
  the conditions the games were played under, so a shared anchor does not make
  two tables comparable. `scale_id` carries the round name inside its digest, so
  two rounds can never share a token — *same scale, comparable; different
  scale, not*.
- **13 Newton solves cost 0.077 s** against 0.002 s for one pooled solve, on a
  9,150-game store. Nothing is bought by pooling.

The `_root` bucket — loose `data/games/*.json` — is excluded here, in one named
line. Those records carry a `round` field that no directory backs (49 distinct
values over 102 files), so they are not a round and cannot be fitted as one.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping

from arena.records.ratings.cache import (
    CacheStats,
    RoundCounts,
    round_count_tables,
)
from arena.records.ratings.counts import CountTable, component_index
from arena.records.ratings.fit import Delta, Estimate, RatingFit, fit_ratings
from arena.records.ratings.policy import ENGINE_MISMATCH, Policy, Prior
from arena.records.registry import Registry

RATED = "rated"
UNRATED = "unrated"

# How a round's anchor was resolved. Printed in every section header, because
# which program pins the scale is the first thing that makes a table readable.
ANCHOR_GLOBAL = "global"
ANCHOR_ROUND_LOCAL = "round_local"

# Why a round could not be fitted. Part of the on-disk format.
NO_ELIGIBLE_GAMES = "no_eligible_games"
NO_GAMES_IN_ERA = "no_games_in_era"
SOLVER_FAILED = "solver_failed"

UNRATED_REASONS = (NO_ELIGIBLE_GAMES, NO_GAMES_IN_ERA, SOLVER_FAILED)

# Reasons render as prose next to the counts, and every one has a remediation.
# They live here rather than in the renderer so the reporting layer carries no
# knowledge of what a rating is.
REASON_TEXT = {
    NO_ELIGIBLE_GAMES: "no eligible games",
    NO_GAMES_IN_ERA: "no games in this engine era",
    SOLVER_FAILED: "the solver failed",
}
REMEDIATION = {
    NO_ELIGIBLE_GAMES: (
        "Register the roster (`python -m arena.records.registry --verify`) and refit."
    ),
    NO_GAMES_IN_ERA: (
        "Re-run the round on the current engine era, or read the old one with "
        "`--era <version>`."
    ),
    SOLVER_FAILED: (
        "Inspect the round's counts with `python -m arena.records.ratings "
        "--list-rounds`."
    ),
}


def scale_id(
    *, round_name: str, anchor: str, anchor_kind: str, counts_digest: str
) -> str:
    """
    The token that says which scale a table is on.

    The round name is inside the digest deliberately: two rounds with
    byte-identical games still get different tokens, so a consumer that joins
    two rows on rating without checking this has no excuse.
    """
    sha = hashlib.sha256()
    sha.update(
        "\0".join((round_name, anchor, anchor_kind, counts_digest)).encode("utf-8")
    )
    return f"scale:{sha.hexdigest()[:12]}"


def resolve_round_anchor(
    table: CountTable, global_anchor: str
) -> tuple[str | None, str | None]:
    """
    `(anchor entity, anchor kind)` for one round, in three tiers.

    | Tier | Condition | Anchor | kind |
    | --- | --- | --- | --- |
    | 1 | the global anchor played in this round | it | `global` |
    | 2 | it did not, but the round has games | the most-played entity | `round_local` |
    | 3 | the round has no eligible games | `(None, None)` | — |

    **The global anchor is never forced into a round that did not play it.**
    Forcing it in via `count_table(extra_entities=[anchor])` makes it a zero-game
    singleton component; the round's real entities then form a second component
    whose location is set by the prior alone. Measured on a 10-game round where
    `b` beat `c` 9–1, that moved the `b → c` contrast from −151.5 ± 157.2 to
    −220.0 ± 191.3 — the prior doing work an anchor should be doing. See
    `tests/test_ratings_connectivity.py`'s isolated-anchor case for the shape.

    Tier 2's tiebreak is `max(sorted(entities), key=games)`, so an appearance tie
    resolves to the lowest entity key — the same deterministic rule
    `scripts/measure_heuristics.py` uses for its round-local snippet.
    """
    if not table.entities:
        return None, None
    # An entity is in the table only because it played, so membership *is* the
    # "played at least one eligible game in this round" test.
    if global_anchor in table.entities:
        return global_anchor, ANCHOR_GLOBAL
    ordered = sorted(table.entities)
    return max(ordered, key=lambda e: sum(table.record_for(e))), ANCHOR_ROUND_LOCAL


def unrated_reason(table: CountTable) -> str:
    """Why a round with no eligible games has none."""
    excluded = dict(table.excluded)
    if excluded and set(excluded) == {ENGINE_MISMATCH}:
        return NO_GAMES_IN_ERA
    return NO_ELIGIBLE_GAMES


@dataclass(frozen=True)
class RoundResult:
    """
    One round's published state: a fit, or a status saying why there is none.

    An unrated round is never dropped. A round that disappears from the report is
    indistinguishable from a round nobody ran, and the actionable content is
    exactly the part that survives — the stored-game count and the `excluded`
    breakdown, which for today's `morpheus-cadence-*` rounds say "register these
    hashes and refit".
    """

    round: str
    counts: CountTable
    engine_versions: tuple[str, ...]
    stored_games: int
    status: str
    reason: str | None = None
    fit: RatingFit | None = None
    anchor: str | None = None
    anchor_kind: str | None = None
    scale_id: str | None = None

    # -- what a round is --

    @property
    def rated(self) -> bool:
        return self.status == RATED

    @property
    def rated_games(self) -> int:
        return self.counts.games

    @property
    def excluded(self) -> Mapping[str, int]:
        return self.counts.excluded

    @property
    def counts_digest(self) -> str:
        return self.counts.digest

    @property
    def era_split(self) -> bool:
        """Whether the round's stored games span more than one engine era."""
        return len(self.engine_versions) > 1

    @property
    def scale_token(self) -> str:
        """The `scale_id` without its prefix, for rendering."""
        return "" if self.scale_id is None else self.scale_id.split(":", 1)[-1]

    @property
    def reason_text(self) -> str | None:
        return None if self.reason is None else REASON_TEXT.get(self.reason, self.reason)

    @property
    def remediation(self) -> str | None:
        return None if self.reason is None else REMEDIATION.get(self.reason)

    # -- the fit, when there is one --

    @property
    def anchor_rating(self) -> float | None:
        return None if self.fit is None else self.fit.spec.anchor_rating

    @property
    def anchor_component(self) -> int | None:
        """
        Which connectivity group holds the anchor.

        When a round splits and the anchor lands in a minority group, most rows
        are prior-located rather than measured, and the warning has to say which
        group is the measured one.
        """
        if self.fit is None or self.anchor is None:
            return None
        return component_index(self.counts)[self.anchor]

    @property
    def components(self) -> tuple[tuple[str, ...], ...]:
        return () if self.fit is None else self.fit.components

    @property
    def connected(self) -> bool:
        return self.fit is not None and self.fit.connected

    @property
    def entities(self) -> tuple[str, ...]:
        return self.counts.entities

    @property
    def seat_advantage(self) -> Estimate | None:
        return None if self.fit is None else self.fit.seat_advantage

    @property
    def draw_log_nu(self) -> Estimate | None:
        return None if self.fit is None else self.fit.draw_log_nu

    @property
    def min_games_display(self) -> int | None:
        return None if self.fit is None else self.fit.policy.min_games_display

    @property
    def solver(self):
        """The round's `SolverReport`, or None when there is no fit."""
        return None if self.fit is None else self.fit.solver

    @property
    def ranked_count(self) -> int:
        return 0 if self.fit is None else len(self.fit.ranked())

    def delta(self, a: str, b: str) -> Delta:
        """`theta_b - theta_a` inside this round. Unrated rounds have no answer."""
        if self.fit is None:
            raise ValueError(
                f"round {self.round!r} is unrated ({self.reason}), so it supports "
                f"no contrast; store eligible games and refit"
            )
        return self.fit.delta(a, b)


class RoundFits:
    """
    Every round's result, ordered by round name ascending.

    Indexing gives the `RatingFit` — `fits["joe-r4"].delta(a, b)` is the contrast
    a decision is made from — and iterating gives `RoundResult`s, which is what
    reporting needs. Asking for an unrated round raises and names the rated ones,
    rather than handing back something that reads like a fit over no games.
    """

    def __init__(
        self,
        results: list[RoundResult] | tuple[RoundResult, ...],
        *,
        policy: Policy,
        prior: Prior,
        cache_stats: CacheStats | None = None,
    ) -> None:
        # Sorted here rather than trusted from the caller: the order is part of
        # the published format, and directory discovery order is not a contract.
        self.results = tuple(sorted(results, key=lambda r: r.round))
        self.policy = policy
        self.prior = prior
        self.cache_stats = cache_stats
        self._by_name = {result.round: result for result in self.results}

    def __iter__(self) -> Iterator[RoundResult]:
        return iter(self.results)

    def __len__(self) -> int:
        return len(self.results)

    def __contains__(self, round_name: object) -> bool:
        return round_name in self._by_name

    def __getitem__(self, round_name: str) -> RatingFit:
        result = self._by_name.get(round_name)
        if result is None:
            raise KeyError(
                f"no round {round_name!r}; rounds are: {', '.join(self.names) or 'none'}"
            )
        if result.fit is None:
            raise KeyError(
                f"round {round_name!r} is unrated ({result.reason}); rated rounds "
                f"are: {', '.join(r.round for r in self.rated) or 'none'}"
            )
        return result.fit

    def result(self, round_name: str) -> RoundResult:
        result = self._by_name.get(round_name)
        if result is None:
            raise KeyError(
                f"no round {round_name!r}; rounds are: {', '.join(self.names) or 'none'}"
            )
        return result

    def get(self, round_name: str) -> RoundResult | None:
        return self._by_name.get(round_name)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(result.round for result in self.results)

    @property
    def rated(self) -> tuple[RoundResult, ...]:
        return tuple(result for result in self.results if result.rated)

    @property
    def unrated(self) -> tuple[RoundResult, ...]:
        return tuple(result for result in self.results if not result.rated)

    @property
    def rated_games(self) -> int:
        return sum(result.rated_games for result in self.results)

    @property
    def stored_games(self) -> int:
        return sum(result.stored_games for result in self.results)

    @property
    def rounds_digest(self) -> str:
        """
        One read that answers "does anything need refitting?".

        Replaces the pooled `counts_digest`. Per-round digests still answer the
        same question per round; this is the whole-store form.
        """
        sha = hashlib.sha256()
        for result in self.results:
            sha.update(
                f"{result.round}\0{result.status}\0{result.counts_digest}\n".encode(
                    "utf-8"
                )
            )
        return f"sha256:{sha.hexdigest()}"


def fit_round(
    counts: RoundCounts,
    *,
    global_anchor: str,
    policy: Policy,
    prior: Prior | None = None,
) -> RoundResult:
    """Fit one round, or say why it cannot be fitted."""
    table = counts.table
    common = {
        "round": counts.round,
        "counts": table,
        "engine_versions": counts.engine_versions,
        "stored_games": counts.stored_games,
    }
    anchor, kind = resolve_round_anchor(table, global_anchor)
    if anchor is None or kind is None:
        return RoundResult(**common, status=UNRATED, reason=unrated_reason(table))

    token = scale_id(
        round_name=counts.round,
        anchor=anchor,
        anchor_kind=kind,
        counts_digest=table.digest,
    )
    try:
        fit = fit_ratings(
            table,
            prior=prior,
            anchor=anchor,
            policy=policy,
            round_name=counts.round,
            anchor_kind=kind,
            scale_id=token,
        )
    except Exception:
        # The prior makes the Hessian invertible, so this is near-impossible —
        # but one round that cannot be solved must not take the others down with
        # it. A non-converged solve is *not* this case: it publishes, with
        # `solver.converged` false and a warning, the same as the pooled fit did.
        return RoundResult(**common, status=UNRATED, reason=SOLVER_FAILED)

    return RoundResult(
        **common,
        status=RATED,
        fit=fit,
        anchor=anchor,
        anchor_kind=kind,
        scale_id=token,
    )


def fit_rounds(
    *,
    games_dir: Path | None = None,
    cache_dir: Path,
    policy: Policy,
    prior: Prior | None = None,
    registry: Registry | None = None,
    global_anchor: str,
    use_cache: bool = True,
) -> RoundFits:
    """
    Fit every round under `games_dir`, independently.

    `include_root=False` is the whole `_root` exclusion: one line, named, and
    testable. Loose games contribute to no published number.
    """
    rounds, stats = round_count_tables(
        games_dir=games_dir,
        cache_dir=cache_dir,
        policy=policy,
        registry=registry,
        use_cache=use_cache,
        include_root=False,
    )
    results = [
        fit_round(counts, global_anchor=global_anchor, policy=policy, prior=prior)
        for counts in rounds
    ]
    return RoundFits(
        results,
        policy=policy,
        prior=prior or Prior(),
        cache_stats=stats,
    )


__all__ = [
    "ANCHOR_GLOBAL",
    "ANCHOR_ROUND_LOCAL",
    "NO_ELIGIBLE_GAMES",
    "NO_GAMES_IN_ERA",
    "RATED",
    "SOLVER_FAILED",
    "UNRATED",
    "UNRATED_REASONS",
    "RoundFits",
    "RoundResult",
    "fit_round",
    "fit_rounds",
    "resolve_round_anchor",
    "scale_id",
    "unrated_reason",
]
