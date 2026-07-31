"""
The BTDS likelihood: Bradley-Terry + Davidson draws + a shared seat term.

Pure math. No IO, no `arena` imports, nothing but numpy — so it can be tested
against synthetic count tables alone, which is what keeps the recovery tests
fast enough to live in a 3 s suite.

Parameters, all in Elo points on the standard scale `s = 400 / ln 10`:

- `theta[e]` — strength of rated entity `e`
- `beta`     — shared seat-A advantage, one scalar for the whole pool
- `kappa`    — shared draw propensity, `log nu`, one scalar

For a game with entity `i` in seat A and `j` in seat B, with
`d = (theta_i - theta_j + beta) / s`:

    u_a = exp(d/2)   u_b = exp(-d/2)   u_draw = exp(kappa)
    Z   = u_a + u_b + u_draw
    P(a) = u_a/Z     P(b) = u_b/Z      P(draw) = u_draw/Z

This is Davidson's `nu * sqrt(p_a p_b)`, which collapses to `exp(kappa)`
because `sqrt(u_a u_b) = 1` under this parameterization. Equivalently: a
three-category multinomial logit with linear predictors `(d/2, -d/2, kappa)`.

The likelihood reads the games **only** through integer counts per ordered
pair, and the penalized objective is strictly convex (a log-partition function
plus a positive-definite quadratic). Those two facts together are what make
the fit order-independent: the sufficient statistics are exact integers, and
the optimum they determine is unique.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Elo points per unit of log-odds. 400 points is a factor of 10 in odds.
ELO_SCALE = 400.0 / math.log(10.0)

# Newton stops when the largest |gradient| component falls below this. The
# gradient is in units of (penalized nats) per Elo point, so this is far
# tighter than any reported figure needs.
GRADIENT_TOLERANCE = 1e-9
MAX_ITERATIONS = 100


@dataclass(frozen=True)
class ModelSpec:
    """Everything about the fit that is not data."""

    n_entities: int
    anchor: int  # index of the entity held at `anchor_rating`
    anchor_rating: float = 1500.0
    prior_mean: float = 1500.0
    prior_sigma: float = 200.0
    seat_sigma: float = 200.0
    draw_sigma: float = 2.0
    fit_seat: bool = True
    fit_draws: bool = True

    @property
    def n_free(self) -> int:
        """Free parameters: every entity but the anchor, plus the nuisances."""
        return self.n_entities - 1 + int(self.fit_seat) + int(self.fit_draws)

    @property
    def seat_index(self) -> int | None:
        """Index of `beta` in the **full** (pre-anchor-removal) vector."""
        return self.n_entities if self.fit_seat else None

    @property
    def draw_index(self) -> int | None:
        """Index of `kappa` in the **full** vector."""
        if not self.fit_draws:
            return None
        return self.n_entities + int(self.fit_seat)

    @property
    def n_full(self) -> int:
        return self.n_entities + int(self.fit_seat) + int(self.fit_draws)


@dataclass(frozen=True)
class CountArrays:
    """A count table flattened into parallel arrays, one entry per cell."""

    seat_a: np.ndarray  # int, entity index in seat A
    seat_b: np.ndarray  # int, entity index in seat B
    wins_a: np.ndarray  # float
    wins_b: np.ndarray
    draws: np.ndarray

    @property
    def games(self) -> np.ndarray:
        return self.wins_a + self.wins_b + self.draws


@dataclass
class Evaluation:
    """Penalized negative log-likelihood and its derivatives at one point."""

    value: float
    gradient: np.ndarray  # over free parameters
    hessian: np.ndarray | None  # over free parameters, positive definite


def initial_point(spec: ModelSpec) -> np.ndarray:
    """Flat start: every strength at the prior mean, both nuisances at zero."""
    full = np.zeros(spec.n_full)
    full[: spec.n_entities] = spec.prior_mean
    return drop_anchor(full, spec)


def drop_anchor(full: np.ndarray, spec: ModelSpec) -> np.ndarray:
    return np.delete(full, spec.anchor, axis=0)


def restore_anchor(free: np.ndarray, spec: ModelSpec) -> np.ndarray:
    """Re-insert the anchor's fixed rating, giving the full parameter vector."""
    full = np.insert(free, spec.anchor, spec.anchor_rating, axis=0)
    return full


def unpack(free: np.ndarray, spec: ModelSpec) -> tuple[np.ndarray, float, float]:
    """Split a free-parameter vector into `(theta, beta, kappa)`."""
    full = restore_anchor(free, spec)
    theta = full[: spec.n_entities]
    beta = float(full[spec.seat_index]) if spec.fit_seat else 0.0
    kappa = float(full[spec.draw_index]) if spec.fit_draws else -math.inf
    return theta, beta, kappa


def probabilities(
    theta: np.ndarray,
    beta: float,
    kappa: float,
    counts: CountArrays,
    *,
    fit_draws: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-cell `(half_d, log Z, p_a, p_b, p_draw)`, computed stably."""
    half = (theta[counts.seat_a] - theta[counts.seat_b] + beta) / (2.0 * ELO_SCALE)
    if fit_draws:
        shift = np.maximum(np.abs(half), kappa)
        draw_term = np.exp(kappa - shift)
    else:
        shift = np.abs(half)
        draw_term = np.zeros_like(half)
    exp_a = np.exp(half - shift)
    exp_b = np.exp(-half - shift)
    total = exp_a + exp_b + draw_term
    log_z = shift + np.log(total)
    return half, log_z, exp_a / total, exp_b / total, draw_term / total


def evaluate(
    free: np.ndarray,
    counts: CountArrays,
    spec: ModelSpec,
    *,
    need_hessian: bool = True,
) -> Evaluation:
    """
    Penalized negative log-likelihood, its gradient, and its Hessian.

    Derivatives are analytic. With `d` the linear predictor and `n` a cell's
    game count, the two that matter are

        d(logL)/dd   = (wins_a - wins_b - n(p_a - p_b)) / 2
        d2(logL)/dd2 = -(n/4)[(p_a + p_b) - (p_a - p_b)^2]

    The second is, up to the `1/s^2` from the chain rule, the per-game Fisher
    information `fit.games_to_resolve` inverts.
    """
    theta, beta, kappa = unpack(free, spec)
    half, log_z, p_a, p_b, p_draw = probabilities(
        theta, beta, kappa, counts, fit_draws=spec.fit_draws
    )
    n = counts.games
    margin = counts.wins_a - counts.wins_b

    log_likelihood = float(
        np.sum(margin * half + counts.draws * (kappa if spec.fit_draws else 0.0) - n * log_z)
    )

    # --- prior (MAP, not MLE) ---
    free_theta_mask = np.ones(spec.n_entities, dtype=bool)
    free_theta_mask[spec.anchor] = False
    centred = (theta - spec.prior_mean) * free_theta_mask
    penalty = float(np.sum(centred**2)) / (2.0 * spec.prior_sigma**2)
    if spec.fit_seat:
        penalty += beta**2 / (2.0 * spec.seat_sigma**2)
    if spec.fit_draws:
        penalty += kappa**2 / (2.0 * spec.draw_sigma**2)

    value = -log_likelihood + penalty

    # --- gradient of the negative log-likelihood, in the full parameter space ---
    d_ll_d_d = (margin - n * (p_a - p_b)) / 2.0
    per_theta = d_ll_d_d / ELO_SCALE

    grad_full = np.zeros(spec.n_full)
    np.add.at(grad_full, counts.seat_a, -per_theta)
    np.add.at(grad_full, counts.seat_b, per_theta)
    grad_full[: spec.n_entities] += centred / spec.prior_sigma**2
    if spec.fit_seat:
        grad_full[spec.seat_index] = -float(np.sum(per_theta)) + beta / spec.seat_sigma**2
    if spec.fit_draws:
        grad_full[spec.draw_index] = (
            -float(np.sum(counts.draws - n * p_draw)) + kappa / spec.draw_sigma**2
        )

    hess_full = None
    if need_hessian:
        curvature = (n / 4.0) * ((p_a + p_b) - (p_a - p_b) ** 2)
        scaled = curvature / ELO_SCALE**2
        hess_full = np.zeros((spec.n_full, spec.n_full))
        # v v^T with v = e_i - e_j (+ e_beta). A self-play cell has v = 0 over
        # the strengths, and the four scatter-adds below cancel exactly there.
        np.add.at(hess_full, (counts.seat_a, counts.seat_a), scaled)
        np.add.at(hess_full, (counts.seat_b, counts.seat_b), scaled)
        np.add.at(hess_full, (counts.seat_a, counts.seat_b), -scaled)
        np.add.at(hess_full, (counts.seat_b, counts.seat_a), -scaled)

        if spec.fit_seat:
            b = spec.seat_index
            np.add.at(hess_full, (counts.seat_a, np.full_like(counts.seat_a, b)), scaled)
            np.add.at(hess_full, (counts.seat_b, np.full_like(counts.seat_b, b)), -scaled)
            np.add.at(hess_full, (np.full_like(counts.seat_a, b), counts.seat_a), scaled)
            np.add.at(hess_full, (np.full_like(counts.seat_b, b), counts.seat_b), -scaled)
            hess_full[b, b] = float(np.sum(scaled)) + 1.0 / spec.seat_sigma**2

        if spec.fit_draws:
            k = spec.draw_index
            cross = (n / 2.0) * p_draw * (p_a - p_b) / ELO_SCALE
            np.add.at(hess_full, (counts.seat_a, np.full_like(counts.seat_a, k)), -cross)
            np.add.at(hess_full, (counts.seat_b, np.full_like(counts.seat_b, k)), cross)
            np.add.at(hess_full, (np.full_like(counts.seat_a, k), counts.seat_a), -cross)
            np.add.at(hess_full, (np.full_like(counts.seat_b, k), counts.seat_b), cross)
            if spec.fit_seat:
                b = spec.seat_index
                hess_full[b, k] = hess_full[k, b] = -float(np.sum(cross))
            hess_full[k, k] = (
                float(np.sum(n * p_draw * (1.0 - p_draw))) + 1.0 / spec.draw_sigma**2
            )

        diag = np.arange(spec.n_entities)
        hess_full[diag, diag] += 1.0 / spec.prior_sigma**2

    gradient = drop_anchor(grad_full, spec)
    hessian = None
    if hess_full is not None:
        hessian = np.delete(np.delete(hess_full, spec.anchor, axis=0), spec.anchor, axis=1)
    return Evaluation(value=value, gradient=gradient, hessian=hessian)


@dataclass(frozen=True)
class SolverReport:
    iterations: int
    max_abs_grad: float
    converged: bool


def solve(
    counts: CountArrays,
    spec: ModelSpec,
    *,
    tolerance: float = GRADIENT_TOLERANCE,
    max_iterations: int = MAX_ITERATIONS,
) -> tuple[np.ndarray, np.ndarray, SolverReport]:
    """
    Damped Newton on the penalized negative log-likelihood.

    Returns `(free_parameters, covariance, report)`. The objective is strictly
    convex, so the optimum is unique and the path taken to it cannot change the
    answer beyond the last few bits — the backtracking line search is there for
    the first step or two from a flat start, not to pick between optima.

    The covariance is the Laplace approximation `H^-1` at the optimum, over the
    free parameters. Because `H` is the Hessian of a strictly convex function
    it is positive definite, so Cholesky is both valid and a check.
    """
    if not spec.fit_draws and float(np.sum(counts.draws)) > 0.0:
        raise ValueError("fit_draws=False but the count table contains draws")

    x = initial_point(spec)
    evaluation = evaluate(x, counts, spec)
    iterations = 0

    def max_abs_grad(step_evaluation: Evaluation) -> float:
        if step_evaluation.gradient.size == 0:
            return 0.0
        return float(np.max(np.abs(step_evaluation.gradient)))

    while iterations < max_iterations and max_abs_grad(evaluation) >= tolerance:
        iterations += 1
        factor = np.linalg.cholesky(evaluation.hessian)
        step = -np.linalg.solve(factor.T, np.linalg.solve(factor, evaluation.gradient))
        slope = float(evaluation.gradient @ step)
        scale = 1.0
        for _ in range(40):
            trial = evaluate(x + scale * step, counts, spec, need_hessian=False)
            if trial.value <= evaluation.value + 1e-4 * scale * slope:
                break
            scale /= 2.0
        x = x + scale * step
        evaluation = evaluate(x, counts, spec)

    residual = max_abs_grad(evaluation)
    covariance = (
        np.linalg.inv(evaluation.hessian)
        if evaluation.hessian.size
        else np.zeros((0, 0))
    )
    report = SolverReport(
        iterations=iterations,
        max_abs_grad=residual,
        converged=residual < tolerance,
    )
    return x, covariance, report


def per_game_information(p_a: float, p_b: float) -> float:
    """
    Fisher information about a head-to-head contrast, per game, in Elo^-2.

    `I(d) = (1/4)[(p_a + p_b) - (p_a - p_b)^2] / s^2`. Evenly matched with no
    draws gives `SE = 347.4/sqrt(n)`; at a 35% draw rate, `430.9/sqrt(n)` —
    draws inflate the standard error 24% at fixed `n`, so ~54% more games buy
    the same precision.
    """
    return 0.25 * ((p_a + p_b) - (p_a - p_b) ** 2) / ELO_SCALE**2
