"""
The fit: count table in, estimates and covariance out.

`RatingFit` is the object every decision is made from. A leaderboard rank is
never the basis for a keep/revert call — the pairwise contrast is, and that
needs the covariance, not just the point estimates:

    Var(theta_B - theta_A) = Sigma_AA + Sigma_BB - 2 Sigma_AB
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from arena.records.ratings.counts import CountTable
from arena.records.ratings.model import (
    ELO_SCALE,
    ModelSpec,
    SolverReport,
    per_game_information,
    probabilities,
    solve,
    unpack,
)
from arena.records.ratings.policy import Policy, Prior

# Standard normal CDF, so the package needs no scipy.
def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


@dataclass(frozen=True)
class Estimate:
    value: float
    se: float

    def interval(self, level: float = 0.95) -> tuple[float, float]:
        z = _z_for(level)
        return (self.value - z * self.se, self.value + z * self.se)

    def to_dict(self) -> dict[str, float]:
        return {"value": self.value, "se": self.se}


@dataclass(frozen=True)
class Delta:
    """One contrast, with everything a decision needs."""

    a: str
    b: str
    value: float
    se: float

    def interval(self, level: float = 0.95) -> tuple[float, float]:
        z = _z_for(level)
        return (self.value - z * self.se, self.value + z * self.se)

    @property
    def ci(self) -> tuple[float, float]:
        return self.interval()

    @property
    def p_stronger(self) -> float:
        """P(b stronger than a). Zero SE means the two are the same entity."""
        if self.se <= 0.0:
            return 0.5
        return _phi(self.value / self.se)


@lru_cache(maxsize=None)
def _z_for(level: float) -> float:
    if not 0.0 < level < 1.0:
        raise ValueError(f"confidence level must be in (0, 1), got {level}")
    # Inverse normal CDF by bisection: exact enough at any level worth naming,
    # and it keeps scipy out of the runtime dependencies.
    lo, hi = 0.0, 40.0
    target = 0.5 + level / 2.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if _phi(mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


class RatingFit:
    """Fitted strengths, nuisance parameters, and their joint covariance."""

    def __init__(
        self,
        *,
        counts: CountTable,
        spec: ModelSpec,
        free: np.ndarray,
        covariance: np.ndarray,
        report: SolverReport,
        prior: Prior,
        policy: Policy,
    ) -> None:
        self.counts = counts
        self.entities = list(counts.entities)
        self.spec = spec
        self.anchor = counts.entities[spec.anchor]
        self.prior = prior
        self.policy = policy
        self.solver = report
        # Set by `cli.refit` when the per-round cache was consulted. Reporting
        # only — it says nothing about the fit, which is identical either way.
        self.cache_stats: Any | None = None

        theta, beta, kappa = unpack(free, spec)
        self._theta = theta
        # Expand the free-parameter covariance back over every entity. The
        # anchor's row and column are zero: it is the reference point, so it
        # carries no uncertainty *by construction*, and a contrast against it
        # has variance Sigma_ee exactly.
        size = spec.n_entities
        full = np.zeros((spec.n_full, spec.n_full))
        keep = [i for i in range(spec.n_full) if i != spec.anchor]
        full[np.ix_(keep, keep)] = covariance
        self._covariance = full[:size, :size]
        self._full_covariance = full

        seat_se = (
            math.sqrt(full[spec.seat_index, spec.seat_index]) if spec.fit_seat else 0.0
        )
        draw_se = (
            math.sqrt(full[spec.draw_index, spec.draw_index]) if spec.fit_draws else 0.0
        )
        self.seat_advantage = Estimate(value=beta, se=seat_se)
        self.draw_log_nu = Estimate(
            value=(kappa if spec.fit_draws else -math.inf), se=draw_se
        )

        self._index = {name: i for i, name in enumerate(self.entities)}
        self._records = {name: counts.record_for(name) for name in self.entities}

    # -- per-entity readouts --

    def _at(self, entity: str) -> int:
        try:
            return self._index[entity]
        except KeyError:
            raise KeyError(f"{entity!r} is not in this fit") from None

    def rating(self, entity: str) -> float:
        return float(self._theta[self._at(entity)])

    def se(self, entity: str) -> float:
        return math.sqrt(float(self._covariance[self._at(entity), self._at(entity)]))

    def estimate(self, entity: str) -> Estimate:
        return Estimate(value=self.rating(entity), se=self.se(entity))

    def interval(self, entity: str, level: float = 0.95) -> tuple[float, float]:
        return self.estimate(entity).interval(level)

    def record(self, entity: str) -> tuple[int, int, int]:
        self._at(entity)  # raise a clear KeyError for an unknown entity
        return self._records[entity]

    def games(self, entity: str) -> int:
        return sum(self.record(entity))

    def provisional(self, entity: str) -> bool:
        """
        Below the display threshold.

        Provisional entities still **participate in the fit** — dropping them
        would change every other rating and break the "pure function of the
        game set" contract. They are only held out of the ranked block and
        barred from being a decision baseline.
        """
        return self.games(entity) < self.policy.min_games_display

    def ranked(self) -> list[str]:
        """Non-provisional entities, strongest first, ties broken by name."""
        names = [e for e in self.entities if not self.provisional(e)]
        return sorted(names, key=lambda e: (-self.rating(e), e))

    def provisional_entities(self) -> list[str]:
        names = [e for e in self.entities if self.provisional(e)]
        return sorted(names, key=lambda e: (-self.rating(e), e))

    # -- contrasts, which is what decisions are made from --

    def delta(self, a: str, b: str) -> Delta:
        """`theta_b - theta_a`, with the SE taken from the joint covariance."""
        i, j = self._at(a), self._at(b)
        variance = (
            self._covariance[i, i] + self._covariance[j, j] - 2.0 * self._covariance[i, j]
        )
        return Delta(a=a, b=b, value=self.rating(b) - self.rating(a), se=math.sqrt(max(variance, 0.0)))

    def p_stronger(self, b: str, a: str) -> float:
        """P(b stronger than a) = Phi(Delta / SE(Delta))."""
        return self.delta(a, b).p_stronger

    def expected(self, a: str, b: str, *, seat_a: bool = True) -> tuple[float, float, float]:
        """`(P(a wins), P(b wins), P(draw))` with `a` in seat A unless told otherwise."""
        first, second = (a, b) if seat_a else (b, a)
        arrays = _single_cell(self, first, second)
        _, _, p_a, p_b, p_draw = probabilities(
            self._theta,
            self.seat_advantage.value,
            self.draw_log_nu.value,
            arrays,
            fit_draws=self.spec.fit_draws,
        )
        if seat_a:
            return float(p_a[0]), float(p_b[0]), float(p_draw[0])
        return float(p_b[0]), float(p_a[0]), float(p_draw[0])

    def games_to_resolve(self, a: str, b: str, *, target_se: float) -> int:
        """
        Extra head-to-head games needed to shrink `SE(Delta)` to `target_se`.

        Inverts the model's own per-game Fisher information for the contrast,
        starting from the precision already in hand. Seat-neutral: it assumes
        the balanced design a decision arm actually runs.
        """
        if target_se <= 0.0:
            raise ValueError("target_se must be positive")
        current = self.delta(a, b).se
        if current > 0.0 and current <= target_se:
            return 0
        p_a, p_b, _ = _seat_neutral_probabilities(self, a, b)
        information = per_game_information(p_a, p_b)
        if information <= 0.0:
            return 0
        have = 1.0 / current**2 if current > 0.0 else 0.0
        need = 1.0 / target_se**2
        return max(0, math.ceil((need - have) / information))

    # -- serialization --

    def to_state(self) -> dict[str, Any]:
        from arena.records.ratings.io import fit_payload

        return fit_payload(self)

    @classmethod
    def from_state(cls, data: dict[str, Any]) -> "LoadedFit":
        from arena.records.ratings.io import fit_from_payload

        return fit_from_payload(data)


def _single_cell(fit: RatingFit, seat_a: str, seat_b: str):
    from arena.records.ratings.model import CountArrays

    return CountArrays(
        seat_a=np.array([fit._at(seat_a)], dtype=np.intp),
        seat_b=np.array([fit._at(seat_b)], dtype=np.intp),
        wins_a=np.array([0.0]),
        wins_b=np.array([0.0]),
        draws=np.array([0.0]),
    )


def _seat_neutral_probabilities(fit: RatingFit, a: str, b: str) -> tuple[float, float, float]:
    """Outcome probabilities with the seat term zeroed out."""
    arrays = _single_cell(fit, a, b)
    _, _, p_a, p_b, p_draw = probabilities(
        fit._theta, 0.0, fit.draw_log_nu.value, arrays, fit_draws=fit.spec.fit_draws
    )
    return float(p_a[0]), float(p_b[0]), float(p_draw[0])


def fit_ratings(
    counts: CountTable,
    *,
    prior: Prior | None = None,
    anchor: str,
    policy: Policy | None = None,
    fit_seat: bool = True,
    fit_draws: bool = True,
) -> RatingFit:
    """
    Fit BTDS to a count table by damped Newton, anchored on one entity.

    `anchor` is removed from the free parameter vector and held at exactly
    `prior.mean`, which is what fixes the otherwise-free additive constant in
    `theta`. Every reported rating is therefore a statement relative to that
    one program.
    """
    prior = prior or Prior()
    policy = policy or Policy()
    if anchor not in counts.entities:
        raise ValueError(
            f"anchor {anchor!r} is not in the count table; it must play, or be "
            f"passed to count_table(extra_entities=...)"
        )
    spec = ModelSpec(
        n_entities=len(counts.entities),
        anchor=counts.entities.index(anchor),
        anchor_rating=prior.mean,
        prior_mean=prior.mean,
        prior_sigma=prior.sigma,
        seat_sigma=prior.seat_sigma,
        draw_sigma=prior.draw_sigma,
        fit_seat=fit_seat,
        fit_draws=fit_draws,
    )
    free, covariance, report = solve(counts.to_arrays(), spec)
    return RatingFit(
        counts=counts,
        spec=spec,
        free=free,
        covariance=covariance,
        report=report,
        prior=prior,
        policy=policy,
    )


@dataclass
class LoadedFit:
    """
    A fit read back from `fit.json`.

    Deliberately not a `RatingFit`: the reloaded object carries estimates and
    covariance but no likelihood, so it can answer questions and cannot be
    mistaken for something that could refit. Refitting always goes back to
    `data/games/`.
    """

    anchor: str
    entities: list[str]
    ratings: dict[str, float]
    standard_errors: dict[str, float]
    records: dict[str, tuple[int, int, int]]
    covariance: np.ndarray
    seat_advantage: Estimate
    draw_log_nu: Estimate
    counts_digest: str
    prior: Prior
    policy: Policy
    excluded: dict[str, int]

    def rating(self, entity: str) -> float:
        return self.ratings[entity]

    def se(self, entity: str) -> float:
        return self.standard_errors[entity]

    def games(self, entity: str) -> int:
        return sum(self.records[entity])

    def delta(self, a: str, b: str) -> Delta:
        i, j = self.entities.index(a), self.entities.index(b)
        variance = self.covariance[i, i] + self.covariance[j, j] - 2.0 * self.covariance[i, j]
        return Delta(
            a=a, b=b, value=self.ratings[b] - self.ratings[a], se=math.sqrt(max(variance, 0.0))
        )


__all__ = [
    "Delta",
    "ELO_SCALE",
    "Estimate",
    "LoadedFit",
    "RatingFit",
    "fit_ratings",
]
