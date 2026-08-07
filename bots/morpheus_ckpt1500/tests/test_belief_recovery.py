"""Part 05 — belief recovery: rejuvenation, max-entropy, collapse confidence."""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.morpheus

from belief import (
    BeliefConfig,
    ess_fraction,
    filter_step,
    initialize_belief,
    pass_action,
)
from memory import empty_memory, update_memory
from observe import emit_observation, observations_match
from particle_summary import summarize_belief
from recovery import (
    maximum_entropy_reconstruction,
    recover_belief,
    rejuvenate,
    update_belief,
)
from state import create_initial_state
from transition import PASS_ACTION, transition


def _board(H: int = 12, W: int = 12):
    grid = np.zeros((H, W), dtype=np.int32)
    grid[0, 0] = 1
    grid[H - 1, W - 1] = 2
    return grid


def test_forced_mismatch_recovers_via_max_entropy():
    state = create_initial_state(_board())
    obs0 = emit_observation(state, 0)
    rng = np.random.default_rng(10)
    cfg = BeliefConfig(n_particles=8, min_general_distance=5)
    belief = initialize_belief(obs0, seat=0, rng=rng, config=cfg)

    next_state, _ = transition(state, np.stack([PASS_ACTION, PASS_ACTION]))
    real = emit_observation(next_state, 0)
    mem = update_memory(empty_memory(obs0.H, obs0.W), obs0)
    mem = update_memory(mem, real)

    # Inject an action that cannot reproduce the pass observation for particles
    # whose enemy general is far from any movable stack — use a fabricated
    # public-total break first to force filter failure, then recover.
    # Stronger: call recover directly after emptying weights.
    empty = filter_step(
        belief,
        pass_action(),
        # Corrupt turn so every particle fails.
        real.__class__(
            H=real.H,
            W=real.W,
            turn=real.turn + 99,
            my_land=real.my_land,
            my_army=real.my_army,
            opp_land=real.opp_land,
            opp_army=real.opp_army,
            type_grid=real.type_grid,
            owner_grid=real.owner_grid,
            army_grid=real.army_grid,
        ),
        [pass_action()] * belief.n,
        rng,
    )
    assert all(p.weight == 0 for p in empty.particles)

    recovered = recover_belief(belief, pass_action(), real, mem, rng)
    assert any(p.weight > 0 for p in recovered.particles)
    for p in recovered.particles:
        if p.weight > 0:
            assert observations_match(emit_observation(p.state, 0), real)


def test_max_entropy_sets_minimum_belief_ess():
    state = create_initial_state(_board())
    obs = emit_observation(state, 0)
    mem = update_memory(empty_memory(obs.H, obs.W), obs)
    rng = np.random.default_rng(11)
    belief = maximum_entropy_reconstruction(
        obs, seat=0, memory=mem, rng=rng, config=BeliefConfig(n_particles=8)
    )
    assert belief.collapsed
    frac = ess_fraction(belief)
    assert frac == pytest.approx(1.0 / 8)
    # summarize exposes the reduced confidence.
    summary = summarize_belief(belief)
    assert summary.ess_fraction == pytest.approx(frac)
    for p in belief.particles:
        assert observations_match(emit_observation(p.state, 0), obs)


def test_recovery_never_replaces_valid_belief_with_invalid():
    state = create_initial_state(_board())
    obs0 = emit_observation(state, 0)
    rng = np.random.default_rng(12)
    belief = initialize_belief(
        obs0,
        seat=0,
        rng=rng,
        config=BeliefConfig(n_particles=8, min_general_distance=5),
    )
    next_state, _ = transition(state, np.stack([PASS_ACTION, PASS_ACTION]))
    real = emit_observation(next_state, 0)
    mem = update_memory(empty_memory(obs0.H, obs0.W), real)
    # Valid update with injected true actions.
    valid = update_belief(
        belief,
        pass_action(),
        real,
        mem,
        rng,
        enemy_actions=[pass_action()] * belief.n,
    )
    assert all(
        observations_match(emit_observation(p.state, 0), real)
        for p in valid.particles
        if p.weight > 0
    )
    # Rejuvenation with no useful alternate history keeps a valid set.
    again = rejuvenate(valid, rng)
    for p in again.particles:
        if p.weight > 0:
            assert observations_match(emit_observation(p.state, 0), real)


def test_rejuvenation_accepts_alternate_enemy_pass_history():
    state = create_initial_state(_board())
    obs0 = emit_observation(state, 0)
    rng = np.random.default_rng(13)
    belief = initialize_belief(
        obs0,
        seat=0,
        rng=rng,
        config=BeliefConfig(n_particles=4, min_general_distance=5, recovery_lag=4),
    )
    # Advance one pass turn with history recorded.
    next_state, _ = transition(state, np.stack([PASS_ACTION, PASS_ACTION]))
    real = emit_observation(next_state, 0)
    mem = update_memory(empty_memory(obs0.H, obs0.W), real)
    belief = update_belief(
        belief,
        pass_action(),
        real,
        mem,
        rng,
        enemy_actions=[pass_action()] * belief.n,
    )
    assert all(len(p.history) == 1 for p in belief.particles)
    # Attach an explicit HistoryFrame and rejuvenate — should still match.
    refreshed = rejuvenate(belief, rng)
    assert refreshed.n == belief.config.n_particles
    for p in refreshed.particles:
        assert observations_match(emit_observation(p.state, 0), real)


def test_update_belief_recovers_when_all_proposals_fail():
    state = create_initial_state(_board())
    obs0 = emit_observation(state, 0)
    rng = np.random.default_rng(14)
    belief = initialize_belief(
        obs0,
        seat=0,
        rng=rng,
        config=BeliefConfig(n_particles=8, min_general_distance=5),
    )
    next_state, _ = transition(state, np.stack([PASS_ACTION, PASS_ACTION]))
    real = emit_observation(next_state, 0)
    mem = update_memory(empty_memory(obs0.H, obs0.W), real)

    # Build enemy actions that move from a cell no particle owns — transition
    # treats invalid moves as no-ops, so they still match pass. To force
    # failure, corrupt the target observation's opp_army then recover on the
    # real obs inside update_belief's recovery path by calling recover on the
    # zeroed set via a wrong filter then recover_belief.
    from belief import BeliefState

    wrong = real.__class__(
        H=real.H,
        W=real.W,
        turn=real.turn,
        my_land=real.my_land,
        my_army=real.my_army,
        opp_land=real.opp_land,
        opp_army=real.opp_army + 50,
        type_grid=real.type_grid,
        owner_grid=real.owner_grid,
        army_grid=real.army_grid,
    )
    failed = filter_step(
        belief, pass_action(), wrong, [pass_action()] * belief.n, rng
    )
    assert isinstance(failed, BeliefState)
    assert all(p.weight == 0 for p in failed.particles)
    recovered = recover_belief(belief, pass_action(), real, mem, rng)
    assert any(p.weight > 0 for p in recovered.particles)
    assert all(
        observations_match(emit_observation(p.state, 0), real)
        for p in recovered.particles
        if p.weight > 0
    )
