"""
The BTDS likelihood and solver, tested against synthetic count tables.

Recovery tests sample **count tables** from a multinomial rather than
simulating individual games, so "2000 games per pair" costs microseconds.
"""
from __future__ import annotations

import math
import os
import subprocess
import sys

import numpy as np
import pytest

from arena.paths import REPO_ROOT
from arena.records.ratings.counts import build
from arena.records.ratings.fit import fit_ratings
from arena.records.ratings.model import ELO_SCALE, ModelSpec, evaluate
from arena.records.ratings.policy import Policy, Prior

ENTITIES = [f"e{i}" for i in range(8)]
SPREAD = [1500, 1650, 1350, 1800, 1450, 1560, 1200, 1720]  # +/- 300 around 1500


def outcome_probabilities(theta_a: float, theta_b: float, beta: float, kappa: float):
    d = (theta_a - theta_b + beta) / ELO_SCALE
    weights = np.array([math.exp(d / 2), math.exp(-d / 2), math.exp(kappa)])
    return weights / weights.sum()


def sample_table(
    truth: dict[str, float],
    *,
    games: int,
    seed: int,
    beta: float = 0.0,
    kappa: float = -30.0,
    orientations: str = "both",
):
    """Draw a count table straight from the model, one multinomial per cell."""
    rng = np.random.default_rng(seed)
    names = list(truth)
    tallies: dict[tuple[str, str], tuple[int, int, int]] = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            pairs = [(a, b), (b, a)] if orientations == "both" else [(a, b)]
            for x, y in pairs:
                probabilities = outcome_probabilities(truth[x], truth[y], beta, kappa)
                drawn = rng.multinomial(games, probabilities)
                tallies[(x, y)] = tuple(int(c) for c in drawn)
    return build(tallies)


# --- derivatives: the whole solver rests on these being right ---------------


def test_gradient_and_hessian_match_finite_differences():
    table = sample_table(dict(zip(ENTITIES, SPREAD)), games=40, seed=3, beta=45.0, kappa=0.3)
    arrays = table.to_arrays()
    spec = ModelSpec(n_entities=len(table.entities), anchor=2, prior_sigma=180.0)
    x = np.array([1520.0, 1440.0, 1610.0, 1490.0, 1555.0, 1380.0, 1700.0, 35.0, 0.3])

    analytic = evaluate(x, arrays, spec)
    step = 1e-5
    numeric_gradient = np.zeros_like(x)
    numeric_hessian = np.zeros((x.size, x.size))
    for i in range(x.size):
        bump = np.zeros_like(x)
        bump[i] = step
        up = evaluate(x + bump, arrays, spec, need_hessian=False)
        down = evaluate(x - bump, arrays, spec, need_hessian=False)
        numeric_gradient[i] = (up.value - down.value) / (2 * step)
        numeric_hessian[i] = (up.gradient - down.gradient) / (2 * step)

    assert np.max(np.abs(analytic.gradient - numeric_gradient)) < 1e-6
    assert np.max(np.abs(analytic.hessian - numeric_hessian)) < 1e-6


def test_penalized_hessian_is_positive_definite():
    """Strict convexity is what makes the optimum unique, hence order-free."""
    table = sample_table(dict(zip(ENTITIES, SPREAD)), games=10, seed=4, kappa=0.2)
    spec = ModelSpec(n_entities=len(table.entities), anchor=0)
    evaluation = evaluate(
        np.concatenate([np.full(len(table.entities) - 1, 1500.0), [0.0, 0.0]]),
        table.to_arrays(),
        spec,
    )
    assert np.linalg.eigvalsh(evaluation.hessian).min() > 0


# --- T3: anchor stability ---------------------------------------------------


def test_anchor_sits_at_exactly_the_prior_mean():
    table = sample_table(dict(zip(ENTITIES, SPREAD)), games=30, seed=5, kappa=0.2)
    for anchor in ("e0", "e3", "e7"):
        fit = fit_ratings(table, anchor=anchor)
        assert fit.rating(anchor) == 1500.0
        assert fit.se(anchor) == 0.0


def test_adding_a_zero_game_entity_moves_nothing():
    truth = dict(zip(ENTITIES, SPREAD))
    table = sample_table(truth, games=30, seed=6, kappa=0.2)
    padded = build(
        {(c.seat_a, c.seat_b): (c.wins_a, c.wins_b, c.draws) for c in table.cells},
        entities=[*table.entities, "newcomer@abc"],
    )
    before = fit_ratings(table, anchor="e0")
    after = fit_ratings(padded, anchor="e0")

    for entity in table.entities:
        assert abs(before.rating(entity) - after.rating(entity)) < 1e-9
    assert after.rating("newcomer@abc") == pytest.approx(1500.0, abs=1e-9)
    assert after.se("newcomer@abc") == pytest.approx(Prior().sigma, rel=1e-6)


def test_a_disconnected_clique_does_not_shift_the_anchored_component():
    truth = dict(zip(ENTITIES, SPREAD))
    anchored = sample_table(truth, games=30, seed=7, kappa=0.2)
    island = sample_table({"x@1": 1600.0, "y@1": 1400.0}, games=30, seed=8, kappa=0.2)
    merged = build(
        {
            **{(c.seat_a, c.seat_b): (c.wins_a, c.wins_b, c.draws) for c in anchored.cells},
            **{(c.seat_a, c.seat_b): (c.wins_a, c.wins_b, c.draws) for c in island.cells},
        }
    )
    before = fit_ratings(anchored, anchor="e0")
    after = fit_ratings(merged, anchor="e0")
    # kappa is shared, so the island does move it; the strengths must not move.
    for entity in anchored.entities:
        assert abs(before.rating(entity) - after.rating(entity)) < 0.5


# --- T4: recovery of known strengths ---------------------------------------


def test_recovers_known_strengths():
    truth = dict(zip(ENTITIES, SPREAD))
    table = sample_table(truth, games=2000, seed=11, beta=0.0, kappa=0.2)
    fit = fit_ratings(table, anchor="e0")

    errors = []
    for entity in ENTITIES:
        expected = truth[entity] - truth["e0"] + 1500.0
        error = fit.rating(entity) - expected
        errors.append(error)
        if entity != "e0":
            assert abs(error) < 3 * fit.se(entity), entity
    assert math.sqrt(sum(e**2 for e in errors) / len(errors)) < 15.0


def test_solver_converges_hard():
    table = sample_table(dict(zip(ENTITIES, SPREAD)), games=200, seed=12, beta=50.0, kappa=0.3)
    fit = fit_ratings(table, anchor="e0")
    assert fit.solver.converged
    assert fit.solver.max_abs_grad < 1e-9
    assert fit.solver.iterations < 20


# --- T5: seat-advantage recovery -------------------------------------------


def test_recovers_the_seat_advantage_on_an_imbalanced_design():
    """
    The fixture is deliberately seat-imbalanced — every pair plays one
    orientation only, which is exactly what production no longer generates.
    It stays because it is the harder case for the seat term.
    """
    truth = dict(zip(ENTITIES, SPREAD))
    table = sample_table(
        truth, games=1500, seed=13, beta=60.0, kappa=0.2, orientations="forward"
    )
    fit = fit_ratings(table, anchor="e0")

    assert abs(fit.seat_advantage.value - 60.0) < 3 * fit.seat_advantage.se
    for entity in ENTITIES:
        expected = truth[entity] - truth["e0"] + 1500.0
        assert abs(fit.rating(entity) - expected) < 3 * fit.se(entity) + 1e-9


def test_ignoring_the_seat_term_biases_the_strengths():
    """The control that proves the parameter earns its place."""
    truth = dict(zip(ENTITIES, SPREAD))
    table = sample_table(
        truth, games=1500, seed=13, beta=60.0, kappa=0.2, orientations="forward"
    )
    with_seat = fit_ratings(table, anchor="e0")
    without_seat = fit_ratings(table, anchor="e0", fit_seat=False)

    def rmse(fit):
        errors = [
            fit.rating(e) - (truth[e] - truth["e0"] + 1500.0) for e in ENTITIES
        ]
        return math.sqrt(sum(x**2 for x in errors) / len(errors))

    assert rmse(without_seat) > 3 * rmse(with_seat)


# --- T6: the draw model -----------------------------------------------------


def test_draw_parameter_reproduces_the_simulated_draw_rate():
    truth = dict(zip(ENTITIES, SPREAD))
    for kappa in (-0.5, 0.0, 0.75):
        table = sample_table(truth, games=3000, seed=14, kappa=kappa)
        fit = fit_ratings(table, anchor="e0")
        assert abs(fit.draw_log_nu.value - kappa) < 3 * fit.draw_log_nu.se

        observed = sum(c.draws for c in table.cells) / table.games
        _, _, modelled = fit.expected("e1", "e2")
        assert abs(observed - modelled) < 0.05


def test_an_all_draw_pair_stays_at_the_prior_mean_with_a_large_se():
    """Both seat orders, so the seat term is identified and only draws differ."""
    all_draws = fit_ratings(
        build({("a@1", "b@1"): (0, 0, 60), ("b@1", "a@1"): (0, 0, 60)}), anchor="a@1"
    )
    decisive = fit_ratings(
        build({("a@1", "b@1"): (30, 30, 0), ("b@1", "a@1"): (30, 30, 0)}), anchor="a@1"
    )

    assert all_draws.rating("b@1") == pytest.approx(1500.0, abs=1e-6)
    assert decisive.rating("b@1") == pytest.approx(1500.0, abs=1e-6)
    # Draws say almost nothing about *which* side is stronger, so the prior is
    # nearly all that bounds it. The same 120 decisive games pin it 5x tighter,
    # landing on the documented 347.4/sqrt(n).
    assert all_draws.se("b@1") > 0.8 * Prior().sigma
    assert decisive.se("b@1") == pytest.approx(347.44 / math.sqrt(120), rel=0.05)


# --- T7: sparse and undefeated ---------------------------------------------


def test_an_undefeated_entity_is_finite_and_provisional():
    table = build({("anchor@1", "unbeaten@1"): (0, 6, 0)})
    fit = fit_ratings(table, anchor="anchor@1")

    rating = fit.rating("unbeaten@1")
    assert math.isfinite(rating)
    assert 1500.0 < rating < 1500.0 + 4 * Prior().sigma
    assert fit.provisional("unbeaten@1") is True
    assert fit.record("unbeaten@1") == (6, 0, 0)


def test_a_zero_game_entity_sits_at_the_prior_mean_with_prior_se():
    table = build({("anchor@1", "b@1"): (5, 5, 5)}, entities=["idle@1"])
    fit = fit_ratings(table, anchor="anchor@1")
    assert fit.rating("idle@1") == pytest.approx(1500.0, abs=1e-9)
    assert fit.se("idle@1") == pytest.approx(Prior().sigma, rel=1e-6)
    assert fit.games("idle@1") == 0


def test_hiding_a_provisional_entity_changes_no_rating():
    """Provisional is a display rule. Dropping one from the fit would not be."""
    truth = dict(zip(ENTITIES, SPREAD))
    table = sample_table(truth, games=60, seed=15, kappa=0.2)
    strict = fit_ratings(table, anchor="e0", policy=Policy(min_games_display=10_000))
    loose = fit_ratings(table, anchor="e0", policy=Policy(min_games_display=0))

    assert all(strict.provisional(e) for e in ENTITIES)
    assert not any(loose.provisional(e) for e in ENTITIES)
    for entity in ENTITIES:
        assert strict.rating(entity) == loose.rating(entity)
    assert strict.ranked() == []
    assert len(loose.ranked()) == len(ENTITIES)


# --- T8: probability and sample size ---------------------------------------


def test_p_stronger_is_the_normal_cdf_of_the_contrast():
    table = sample_table(dict(zip(ENTITIES, SPREAD)), games=200, seed=16, kappa=0.2)
    fit = fit_ratings(table, anchor="e0")
    delta = fit.delta("e2", "e3")
    expected = 0.5 * (1.0 + math.erf(delta.value / delta.se / math.sqrt(2.0)))
    assert fit.p_stronger("e3", "e2") == pytest.approx(expected, rel=1e-12)
    low, high = delta.ci
    assert low == pytest.approx(delta.value - 1.959963985 * delta.se, rel=1e-6)
    assert high == pytest.approx(delta.value + 1.959963985 * delta.se, rel=1e-6)


def test_games_to_resolve_predicts_the_standard_error_it_promises():
    truth = {"a@1": 1500.0, "b@1": 1520.0}
    table = sample_table(truth, games=100, seed=17, kappa=0.2)
    fit = fit_ratings(table, anchor="a@1")

    target = 12.75
    extra = fit.games_to_resolve("a@1", "b@1", target_se=target)
    assert extra > 0

    grown = sample_table(truth, games=100 + extra // 2, seed=17, kappa=0.2)
    achieved = fit_ratings(grown, anchor="a@1").delta("a@1", "b@1").se
    assert 0.8 * target < achieved < 1.2 * target


def test_games_to_resolve_is_zero_once_the_target_is_met():
    table = sample_table({"a@1": 1500.0, "b@1": 1520.0}, games=5000, seed=18, kappa=0.2)
    fit = fit_ratings(table, anchor="a@1")
    assert fit.games_to_resolve("a@1", "b@1", target_se=200.0) == 0


def test_per_game_information_matches_the_documented_sanity_values():
    from arena.records.ratings.model import per_game_information

    even_no_draws = per_game_information(0.5, 0.5)
    assert 1.0 / math.sqrt(even_no_draws) == pytest.approx(347.44, abs=0.05)
    at_35_percent_draws = per_game_information(0.325, 0.325)
    assert 1.0 / math.sqrt(at_35_percent_draws) == pytest.approx(430.9, abs=0.1)


# --- T14: determinism environment ------------------------------------------


def test_fit_is_unchanged_with_a_single_blas_thread():
    """
    Guards C2's floating-point contract: no BLAS pin is needed at this size.

    If this ever goes flaky, the fix is to pin the thread count in `fit.py` and
    assert the pin here instead.
    """
    script = (
        "import numpy as np;"
        "import sys; sys.path.insert(0, %r);" % str(REPO_ROOT)
        + "from tests.test_ratings_model import sample_table, ENTITIES, SPREAD;"
        "from arena.records.ratings.fit import fit_ratings;"
        "t = sample_table(dict(zip(ENTITIES, SPREAD)), games=300, seed=19, beta=40.0, kappa=0.3);"
        "f = fit_ratings(t, anchor='e0');"
        "print(repr([f.rating(e) for e in ENTITIES] + [f.seat_advantage.value, f.draw_log_nu.value]))"
    )

    def run(threads: str | None) -> str:
        env = dict(os.environ)
        for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            env.pop(key, None)
            if threads is not None:
                env[key] = threads
        return subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
            env=env,
            cwd=str(REPO_ROOT),
        ).stdout.strip()

    pinned = np.array(eval(run("1")))
    unpinned = np.array(eval(run(None)))
    assert np.max(np.abs(pinned - unpinned)) < 1e-9


# --- T15: independent solver oracle ----------------------------------------


def test_agrees_with_elotes_independently_written_bradley_terry():
    """
    Our damped Newton vs elote's minorization-maximization, same estimator.

    Matched conditions: no draws, seat term forced to 0, our prior effectively
    off, elote's regularizer effectively off. Both fits are then plain
    Bradley-Terry MLE, which no self-consistency test could check.
    """
    elote = pytest.importorskip("elote")

    truth = {f"e{i}@1": r for i, r in enumerate([1500, 1620, 1400, 1700, 1380, 1560])}
    table = sample_table(truth, games=40, seed=20, beta=0.0, kappa=-40.0)
    assert sum(c.draws for c in table.cells) == 0

    ours = fit_ratings(
        table,
        prior=Prior(sigma=1e7),
        anchor="e0@1",
        fit_seat=False,
        fit_draws=False,
    )

    elote.BradleyTerryCompetitor.configure_class(reg=1e-6, tol=1e-12)
    try:
        competitors = {name: elote.BradleyTerryCompetitor() for name in table.entities}
        for cell in table.cells:
            a, b = competitors[cell.seat_a], competitors[cell.seat_b]
            for _ in range(cell.wins_a):
                a.beat(b)
            for _ in range(cell.wins_b):
                a.lost_to(b)
        theirs = {name: c.rating for name, c in competitors.items()}
    finally:
        elote.BradleyTerryCompetitor.configure_class(reg=0.1, tol=1e-8)

    def centred(values: dict[str, float]) -> dict[str, float]:
        mean = sum(values.values()) / len(values)
        return {k: v - mean for k, v in values.items()}

    mine = centred({name: ours.rating(name) for name in table.entities})
    other = centred(theirs)
    assert max(abs(mine[k] - other[k]) for k in mine) < 0.01
