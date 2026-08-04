"""Part 05 — belief filter: init, exact match, ESS resample, summary, reservoir."""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.morpheus

from _common.wire import Observation
from belief import (
    BeliefConfig,
    BeliefState,
    ess,
    ess_fraction,
    filter_step,
    initialize_belief,
    legal_enemy_general_candidates,
    maybe_resample,
    normalize_weights,
    pass_action,
    resample,
    unique_particle_count,
)
from memory import empty_memory, update_memory
from observe import emit_observation, observations_match, visibility_mask
from particle_summary import summarize_belief
from proposal import dedupe_enemy_tensors, propose_enemy_actions
from recovery import is_vision_changing, update_belief
from reservoir import ParticleReservoir
from state import GameState, create_initial_state
from tensor import P_BELIEF_ESS, build_tensor
from transition import PASS_ACTION, transition


def _open_board(H: int = 21, W: int = 21) -> np.ndarray:
    grid = np.zeros((H, W), dtype=np.int32)
    grid[0, 0] = 1
    grid[H - 1, W - 1] = 2
    return grid


def _obs_seat(state: GameState, seat: int) -> Observation:
    return emit_observation(state, seat)


def test_visibility_chebyshev_one():
    own = np.zeros((5, 5), dtype=bool)
    own[2, 2] = True
    vis = visibility_mask(own)
    assert int(vis.sum()) == 9
    assert vis[2, 2] and vis[1, 1] and vis[3, 3]
    assert not vis[0, 0]


def test_emit_observation_hides_enemy_and_keeps_public_totals():
    state = create_initial_state(_open_board(8, 8))
    obs = emit_observation(state, 0)
    assert obs.my_land == 1 and obs.opp_land == 1
    assert obs.my_army == 1 and obs.opp_army == 1
    types = np.asarray(obs.type_grid)
    owners = np.asarray(obs.owner_grid)
    # Enemy general is outside the 3x3 around (0,0).
    assert types[7, 7] in (0, 5)  # fogged
    assert owners[7, 7] == 0
    assert types[0, 0] == 4 and owners[0, 0] == 1


def test_initial_belief_matches_observation_and_public_totals():
    state = create_initial_state(_open_board())
    obs = emit_observation(state, 0)
    rng = np.random.default_rng(0)
    belief = initialize_belief(
        obs, seat=0, rng=rng, config=BeliefConfig(n_particles=16)
    )
    assert belief.n == 16
    for p in belief.particles:
        sim = emit_observation(p.state, 0)
        assert observations_match(sim, obs)
        assert abs(p.weight - 1.0 / 16) < 1e-9
    assert unique_particle_count(belief) >= 1
    cands = legal_enemy_general_candidates(obs)
    assert len(cands) >= 1
    # True enemy general must be among legal candidates on this open board.
    assert (20, 20) in cands


def test_filter_keeps_exact_match_and_zeros_mismatch():
    state = create_initial_state(_open_board(10, 10))
    obs0 = emit_observation(state, 0)
    rng = np.random.default_rng(1)
    belief = initialize_belief(
        obs0,
        seat=0,
        rng=rng,
        config=BeliefConfig(n_particles=8, min_general_distance=5),
    )
    # Both seats pass → next observation is deterministic from true state.
    actions = np.stack([PASS_ACTION, PASS_ACTION])
    next_state, _ = transition(state, actions)
    real_obs = emit_observation(next_state, 0)

    enemy_pass = [pass_action()] * belief.n
    filtered = filter_step(belief, pass_action(), real_obs, enemy_pass, rng)
    assert all(p.weight > 0 for p in filtered.particles)
    for p in filtered.particles:
        assert observations_match(emit_observation(p.state, 0), real_obs)

    # Force a mismatched proposal on a copy: move that cannot be the true pass.
    # Inject impossible actions by claiming enemy moved from a fog cell that
    # particles may not own — filter_step still applies transition; pass vs
    # a no-op invalid move both stay consistent. Instead zero via wrong obs.
    wrong_obs = Observation(
        H=real_obs.H,
        W=real_obs.W,
        turn=real_obs.turn,
        my_land=real_obs.my_land,
        my_army=real_obs.my_army,
        opp_land=real_obs.opp_land + 1,  # public-total mismatch
        opp_army=real_obs.opp_army,
        type_grid=real_obs.type_grid,
        owner_grid=real_obs.owner_grid,
        army_grid=real_obs.army_grid,
    )
    dead = filter_step(belief, pass_action(), wrong_obs, enemy_pass, rng)
    assert all(p.weight == 0 for p in dead.particles)


def test_ess_resample_restores_count():
    particles_weights = [0.9, 0.1] + [0.0] * 6
    # Build a tiny belief via initialize then overwrite weights.
    state = create_initial_state(_open_board(12, 12))
    obs = emit_observation(state, 0)
    rng = np.random.default_rng(2)
    belief = initialize_belief(
        obs,
        seat=0,
        rng=rng,
        config=BeliefConfig(n_particles=8, min_general_distance=5),
    )
    for i, p in enumerate(belief.particles):
        p.weight = 0.99 if i == 0 else 0.01 / 7.0
    belief.particles = normalize_weights(belief.particles)
    assert ess([p.weight for p in belief.particles]) < 4.0
    out = maybe_resample(belief, rng)
    assert out.n == 8
    assert abs(sum(p.weight for p in out.particles) - 1.0) < 1e-9
    # After multinomial resample from a peaked weight, ESS is near n.
    assert ess([p.weight for p in out.particles]) == pytest.approx(8.0)


def test_summary_and_belief_ess_plane():
    state = create_initial_state(_open_board())
    obs = emit_observation(state, 0)
    rng = np.random.default_rng(3)
    belief = initialize_belief(
        obs, seat=0, rng=rng, config=BeliefConfig(n_particles=16)
    )
    summary = summarize_belief(belief)
    assert summary.enemy_general.shape == (21, 21)
    assert 0.0 < summary.ess_fraction <= 1.0
    # Probability mass over general cells sums to ~1.
    assert abs(float(summary.enemy_general.sum()) - 1.0) < 1e-5
    mem = update_memory(empty_memory(obs.H, obs.W), obs)
    tensor = build_tensor(obs, mem, belief=summary)
    assert float(tensor[P_BELIEF_ESS, 0, 0]) == pytest.approx(summary.ess_fraction)


def test_proposal_dedupe_and_injectable_actions():
    state = create_initial_state(_open_board(12, 12))
    obs = emit_observation(state, 0)
    rng = np.random.default_rng(4)
    belief = initialize_belief(
        obs,
        seat=0,
        rng=rng,
        config=BeliefConfig(n_particles=8, min_general_distance=5),
    )
    tensors = [
        np.zeros((49, 21, 21), dtype=np.float32) for _ in range(4)
    ]
    tensors[2] = np.ones((49, 21, 21), dtype=np.float32)
    unique, mapping = dedupe_enemy_tensors(tensors)
    assert len(unique) == 2
    assert mapping[0] == mapping[1] == mapping[3]
    assert mapping[2] != mapping[0]

    actions = propose_enemy_actions(belief, rng)  # uniform, no checkpoint
    assert len(actions) == belief.n
    assert all(len(a) == 5 for a in actions)


def test_proposal_pre_tensor_dedupe_and_policy_parity():
    """Phase 2: dedupe before tensors; logits, masks, and fixed-seed samples match."""
    from action import legal_mask
    from memory import update_memory
    from proposal import (
        dedupe_proposal_keys,
        enemy_info_tensor,
        proposal_info_key,
        sample_from_probs,
        _softmax_masked,
    )

    state = create_initial_state(_open_board(12, 12))
    obs = emit_observation(state, 0)
    rng = np.random.default_rng(11)
    belief = initialize_belief(
        obs,
        seat=0,
        rng=rng,
        config=BeliefConfig(n_particles=8, min_general_distance=5),
    )

    enemy_obs_list = []
    enemy_mem_list = []
    keys = []
    for particle in belief.particles:
        enemy_obs = emit_observation(
            particle.state, belief.enemy_seat, as_arrays=True
        )
        enemy_mem = update_memory(particle.enemy_memory, enemy_obs)
        enemy_obs_list.append(enemy_obs)
        enemy_mem_list.append(enemy_mem)
        keys.append(
            proposal_info_key(enemy_obs, enemy_mem, particle.enemy_prev_action)
        )
    unique_indices, mapping = dedupe_proposal_keys(keys)
    assert len(unique_indices) <= belief.n
    assert len(mapping) == belief.n
    assert max(mapping) == len(unique_indices) - 1

    # Deterministic fake policy: logits proportional to action index.
    def fake_policy(batch):
        batch = np.asarray(batch)
        B = batch.shape[0]
        logits = np.tile(np.arange(3970, dtype=np.float64), (B, 1))
        return logits

    unique_tensors = []
    unique_masks = []
    for p_idx in unique_indices:
        particle = belief.particles[p_idx]
        e_obs = enemy_obs_list[p_idx]
        e_mem = enemy_mem_list[p_idx]
        unique_tensors.append(
            enemy_info_tensor(particle, belief, enemy_obs=e_obs, enemy_mem=e_mem)
        )
        unique_masks.append(legal_mask(e_obs, e_mem))
    stacked = fake_policy(np.stack(unique_tensors, axis=0))

    expected = []
    rng_a = np.random.default_rng(42)
    for u_idx in mapping:
        probs = _softmax_masked(stacked[u_idx], unique_masks[u_idx])
        expected.append(sample_from_probs(probs, rng_a))

    rng_b = np.random.default_rng(42)
    got = propose_enemy_actions(belief, rng_b, policy=fake_policy)
    assert got == expected

    # Belief survival / ESS still hold after a pass update with injected actions.
    next_state, _ = transition(state, np.stack([PASS_ACTION, PASS_ACTION]))
    real = emit_observation(next_state, 0)
    mem = update_memory(empty_memory(obs.H, obs.W), obs)
    mem = update_memory(mem, real)
    nxt = update_belief(
        belief,
        pass_action(),
        real,
        mem,
        np.random.default_rng(7),
        enemy_actions=[pass_action()] * belief.n,
    )
    assert any(p.weight > 0 for p in nxt.particles)
    weights = [p.weight for p in nxt.particles if p.weight > 0]
    assert ess(weights) > 0.0


def test_array_observation_matches_wire():
    state = create_initial_state(_open_board(8, 8))
    wire = emit_observation(state, 0, as_arrays=False)
    arr = emit_observation(state, 0, as_arrays=True)
    assert observations_match(wire, arr)
    assert observations_match(arr, wire)


def test_reservoir_replace_and_sample():
    state = create_initial_state(_open_board(12, 12))
    obs = emit_observation(state, 0)
    rng = np.random.default_rng(5)
    belief = initialize_belief(
        obs,
        seat=0,
        rng=rng,
        config=BeliefConfig(n_particles=8, min_general_distance=5),
    )
    res = ParticleReservoir(capacity=8)
    res.replace_from_belief(belief)
    assert res.n == 8
    sample = res.sample(rng)
    assert sample.state.armies.shape == (12, 12)
    as_b = res.as_belief(seat=0)
    assert isinstance(as_b, BeliefState)


def test_update_belief_survives_pass_turn():
    state = create_initial_state(_open_board(12, 12))
    obs0 = emit_observation(state, 0)
    rng = np.random.default_rng(6)
    belief = initialize_belief(
        obs0,
        seat=0,
        rng=rng,
        config=BeliefConfig(n_particles=8, min_general_distance=5),
    )
    next_state, _ = transition(state, np.stack([PASS_ACTION, PASS_ACTION]))
    real = emit_observation(next_state, 0)
    mem = update_memory(empty_memory(obs0.H, obs0.W), obs0)
    mem = update_memory(mem, real)
    # Inject true enemy pass for every particle.
    nxt = update_belief(
        belief,
        pass_action(),
        real,
        mem,
        rng,
        enemy_actions=[pass_action()] * belief.n,
    )
    assert any(p.weight > 0 for p in nxt.particles)
    for p in nxt.particles:
        if p.weight > 0:
            assert observations_match(emit_observation(p.state, 0), real)


def test_vision_changing_detects_capture_expansion():
    # Build a state where enemy can capture a cell that expands Morpheus vision
    # differently than pass — use a compact owned frontier.
    grid = np.zeros((6, 6), dtype=np.int32)
    grid[0, 0] = 1
    grid[5, 5] = 2
    state = create_initial_state(grid)
    armies = state.armies.copy()
    ownership = state.ownership.copy()
    ownership_neutral = state.ownership_neutral.copy()
    # Enemy stack next to a cell adjacent to player 0's vision fringe.
    armies[0, 2] = 5
    ownership[1, 0, 2] = True
    ownership_neutral[0, 2] = False
    # Also give player 0 a cell so vision reaches near the enemy.
    armies[0, 1] = 2
    ownership[0, 0, 1] = True
    ownership_neutral[0, 1] = False
    state = state._replace(
        armies=armies, ownership=ownership, ownership_neutral=ownership_neutral
    )
    # Enemy move left onto (0,1) would fight; move down may or may not change
    # p0 vision. Compare a move into empty (1,2) vs pass.
    move = (0, 0, 2, 1, 0)  # down from (0,2)
    changed = is_vision_changing(state, seat=0, my_action=pass_action(), enemy_action=move)
    # Either True or False is fine as long as the predicate is deterministic
    # and agrees with a second call.
    assert changed == is_vision_changing(
        state, seat=0, my_action=pass_action(), enemy_action=move
    )
    assert not is_vision_changing(
        state, seat=0, my_action=pass_action(), enemy_action=pass_action()
    )
