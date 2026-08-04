"""Rejuvenation and collapse recovery for the Morpheus belief filter.

Recovery never replaces a valid belief with an invalid one. A maximum-entropy
reconstruction exposes reduced confidence through ``belief_ess``.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from _common.wire import Observation
from action import legal_mask
from belief import (
    Action5,
    BeliefConfig,
    BeliefState,
    Particle,
    as_action5,
    normalize_weights,
    pass_action,
    resample,
)
from memory import (
    TYPE_FOG,
    TYPE_STRUCTURE_FOG,
    VisibleMemory,
    empty_memory,
    update_memory,
)
from observe import emit_observation, observations_match, visibility_mask
from proposal import PolicyFn, policy_action_probs, top_legal_actions
from state import GameState
from transition import PASS_ACTION, transition

Array = np.ndarray


def is_vision_changing(
    state: GameState,
    seat: int,
    my_action: Action5,
    enemy_action: Action5,
) -> bool:
    """True when enemy ``enemy_action`` changes Morpheus visibility vs pass."""
    enemy_seat = 1 - seat
    actions_b = np.zeros((2, 5), dtype=np.int32)
    actions_b[seat] = np.asarray(my_action, dtype=np.int32)
    actions_b[enemy_seat] = np.asarray(enemy_action, dtype=np.int32)
    actions_pass = actions_b.copy()
    actions_pass[enemy_seat] = PASS_ACTION

    next_b, _ = transition(state, actions_b)
    next_p, _ = transition(state, actions_pass)
    vis_b = visibility_mask(next_b.ownership[seat])
    vis_p = visibility_mask(next_p.ownership[seat])
    return not np.array_equal(vis_b, vis_p)


def vision_changing_actions(
    state: GameState,
    seat: int,
    my_action: Action5,
    legal_actions: Sequence[Action5],
) -> list[Action5]:
    return [
        a
        for a in legal_actions
        if a != pass_action() and is_vision_changing(state, seat, my_action, a)
    ]


def _legal_enemy_actions(particle: Particle, belief: BeliefState) -> list[Action5]:
    from action import decode_action

    enemy_obs = emit_observation(particle.state, belief.enemy_seat)
    enemy_mem = update_memory(particle.enemy_memory, enemy_obs)
    mask = legal_mask(enemy_obs, enemy_mem)
    return [as_action5(decode_action(int(i))) for i in np.flatnonzero(mask)]


def _apply_joint(
    state: GameState,
    seat: int,
    my_action: Action5,
    enemy_action: Action5,
) -> GameState:
    actions = np.zeros((2, 5), dtype=np.int32)
    actions[seat] = np.asarray(my_action, dtype=np.int32)
    actions[1 - seat] = np.asarray(enemy_action, dtype=np.int32)
    next_state, _ = transition(state, actions)
    return next_state


def rejuvenate(
    belief: BeliefState,
    rng: np.random.Generator,
    *,
    policy: Optional[PolicyFn] = None,
) -> BeliefState:
    """Policy-guided bounded beam replay of the last recovery_lag turns.

    Accepts only histories that reproduce every stored observation. Admission
    control may stop early; the current valid belief is preserved on failure.
    """
    if belief.n == 0:
        return belief

    # Need at least one particle with history.
    seeds = [p for p in belief.particles if p.history]
    if not seeds:
        return belief

    config = belief.config
    completed: list[tuple[float, Particle]] = []
    transitions_used = 0

    for seed in seeds:
        if len(completed) >= config.max_completed_histories:
            break
        if transitions_used >= config.max_replayed_transitions:
            break

        frames = seed.history
        # Start from the ancestor before the first stored frame.
        start_state = frames[0].state
        start_mem = empty_memory(start_state.armies.shape[0], start_state.armies.shape[1])
        # Beam entries: (weight, state, enemy_memory, enemy_prev, path_actions)
        beam: list[tuple[float, GameState, VisibleMemory, Optional[Action5], tuple]] = [
            (1.0, start_state, start_mem, None, ())
        ]

        for t, frame in enumerate(frames):
            if transitions_used >= config.max_replayed_transitions:
                break
            next_beam: list[
                tuple[float, GameState, VisibleMemory, Optional[Action5], tuple]
            ] = []
            for weight, state, emem, eprev, path in beam:
                # Build a temporary particle for proposal probs.
                temp = Particle(
                    state=state,
                    weight=1.0,
                    enemy_memory=emem,
                    enemy_prev_action=eprev,
                )
                probs = policy_action_probs(belief, temp, policy=policy)
                enemy_obs = emit_observation(state, belief.enemy_seat)
                enemy_mem = update_memory(emem, enemy_obs)
                mask = legal_mask(enemy_obs, enemy_mem)

                if t == 0:
                    candidates = top_legal_actions(probs, mask, 4)
                    vision = vision_changing_actions(
                        state, belief.seat, frame.my_action, _legal_enemy_actions(temp, belief)
                    )
                    for v in vision:
                        if v not in candidates:
                            candidates.append(v)
                else:
                    # Sample up to beam_width distinct actions from the policy.
                    candidates = []
                    seen: set[Action5] = set()
                    for _ in range(config.beam_width * 2):
                        idx = int(rng.choice(len(probs), p=probs))
                        from action import decode_action

                        a = as_action5(decode_action(idx))
                        if a not in seen:
                            seen.add(a)
                            candidates.append(a)
                        if len(candidates) >= config.beam_width:
                            break
                    if not candidates:
                        candidates = [pass_action()]

                for enemy_action in candidates:
                    if transitions_used >= config.max_replayed_transitions:
                        break
                    transitions_used += 1
                    next_state = _apply_joint(
                        state, belief.seat, frame.my_action, enemy_action
                    )
                    sim = emit_observation(next_state, belief.seat)
                    if not observations_match(sim, frame.observation_after):
                        continue
                    # Acceptance weight: product of policy probs.
                    from action import encode_action

                    p_b = float(probs[encode_action(enemy_action)])
                    new_weight = weight * max(p_b, 1e-12)
                    e_obs = emit_observation(next_state, belief.enemy_seat)
                    new_emem = update_memory(enemy_mem, e_obs)
                    next_beam.append(
                        (
                            new_weight,
                            next_state,
                            new_emem,
                            enemy_action,
                            path + (enemy_action,),
                        )
                    )

            if not next_beam:
                beam = []
                break
            # Prune to beam width by weight.
            next_beam.sort(key=lambda x: -x[0])
            beam = next_beam[: config.beam_width]

        for weight, state, emem, eprev, path in beam:
            if len(completed) >= config.max_completed_histories:
                break
            # Carry forward the seed's trailing history window end.
            completed.append(
                (
                    weight,
                    Particle(
                        state=state,
                        weight=weight,
                        enemy_memory=emem,
                        enemy_prev_action=eprev,
                        history=frames,  # observations already matched
                    ),
                )
            )

    if not completed:
        return belief  # keep current valid belief

    weights = np.asarray([w for w, _ in completed], dtype=np.float64)
    weights = weights / weights.sum()
    particles = []
    for w, p in zip(weights, (p for _, p in completed)):
        particles.append(
            Particle(
                state=p.state,
                weight=float(w),
                enemy_memory=p.enemy_memory,
                enemy_prev_action=p.enemy_prev_action,
                history=p.history,
            )
        )
    particles = resample(particles, belief.config.n_particles, rng)
    return BeliefState(
        seat=belief.seat,
        particles=particles,
        config=belief.config,
        collapsed=False,
    )


def _hidden_cells(obs, memory: VisibleMemory) -> list[tuple[int, int]]:
    types = np.asarray(obs.type_grid, dtype=np.int32)
    owners = np.asarray(obs.owner_grid, dtype=np.int32)
    H, W = int(obs.H), int(obs.W)
    cells = []
    for r in range(H):
        for c in range(W):
            if types[r, c] not in (TYPE_FOG, TYPE_STRUCTURE_FOG):
                continue
            if owners[r, c] == 1:
                continue
            if memory.known_mountain[r, c]:
                continue
            if types[r, c] == TYPE_STRUCTURE_FOG and not memory.known_castle[r, c]:
                # Unseen structure fog may be mountain; skip for land allocation.
                if not memory.known_passable_base[r, c]:
                    continue
            cells.append((r, c))
    return cells


def maximum_entropy_reconstruction(
    obs,
    seat: int,
    memory: VisibleMemory,
    rng: np.random.Generator,
    *,
    config: Optional[BeliefConfig] = None,
) -> BeliefState:
    """Uniform-general / even-army reconstruction matching visible + totals."""
    config = config or BeliefConfig()
    H, W = int(obs.H), int(obs.W)
    types = np.asarray(obs.type_grid, dtype=np.int32)
    owners = np.asarray(obs.owner_grid, dtype=np.int32)
    armies_obs = np.asarray(obs.army_grid, dtype=np.int32)
    visible = (types != TYPE_FOG) & (types != TYPE_STRUCTURE_FOG)

    mountains = memory.known_mountain.copy()
    castles = memory.known_castle.copy()
    mountains |= (types == 2)
    castles |= (types == 3)
    # First-sight structure fog that was never passable → mountain.
    struct = types == TYPE_STRUCTURE_FOG
    mountains |= struct & ~memory.known_passable_base & ~memory.known_castle
    castles |= struct & (memory.known_castle | memory.known_passable_base)
    passable = ~mountains

    own_land_vis = int(((owners == 1) & visible).sum())
    opp_land_vis = int(((owners == 2) & visible).sum())
    own_army_vis = int(armies_obs[(owners == 1) & visible].sum())
    opp_army_vis = int(armies_obs[(owners == 2) & visible].sum())

    need_opp_land = int(obs.opp_land) - opp_land_vis
    need_opp_army = int(obs.opp_army) - opp_army_vis
    need_own_land = int(obs.my_land) - own_land_vis
    need_own_army = int(obs.my_army) - own_army_vis
    if need_opp_land < 0 or need_opp_army < 0 or need_own_land < 0 or need_own_army < 0:
        raise ValueError("observation totals inconsistent with visible cells")

    hidden = _hidden_cells(obs, memory)
    if need_opp_land > len(hidden):
        raise ValueError("not enough hidden cells for enemy land total")

    particles: list[Particle] = []
    n = config.n_particles
    # General candidates: all hidden cells when we need at least one enemy cell.
    gen_candidates = list(hidden) if need_opp_land >= 1 else []
    if need_opp_land >= 1 and not gen_candidates:
        raise ValueError("no hidden cell for enemy general")

    for _ in range(n):
        armies = np.zeros((H, W), dtype=np.int32)
        ownership = np.zeros((2, H, W), dtype=bool)
        generals = np.zeros((H, W), dtype=bool)

        # Visible copy.
        for r, c in zip(*np.where(visible)):
            armies[r, c] = int(armies_obs[r, c])
            if owners[r, c] == 1:
                ownership[seat, r, c] = True
            elif owners[r, c] == 2:
                ownership[1 - seat, r, c] = True
            if types[r, c] == 4:
                generals[r, c] = True

        enemy_cells: list[tuple[int, int]] = []
        if need_opp_land >= 1:
            gen = gen_candidates[int(rng.integers(0, len(gen_candidates)))]
            remaining = [c for c in hidden if c != gen]
            extra_n = need_opp_land - 1
            if extra_n > 0:
                pick = rng.choice(len(remaining), size=extra_n, replace=False)
                extras = [remaining[int(i)] for i in np.atleast_1d(pick)]
            else:
                extras = []
            enemy_cells = [gen] + extras
            # Even army split.
            base = need_opp_army // need_opp_land
            rem = need_opp_army % need_opp_land
            for i, (r, c) in enumerate(enemy_cells):
                ownership[1 - seat, r, c] = True
                armies[r, c] = base + (1 if i < rem else 0)
            generals[gen] = True

        # Own hidden land (rare under fog for Morpheus, but keep totals exact).
        own_hidden_needed = need_own_land
        if own_hidden_needed > 0:
            free = [c for c in hidden if c not in enemy_cells]
            if own_hidden_needed > len(free):
                continue
            pick = rng.choice(len(free), size=own_hidden_needed, replace=False)
            own_cells = [free[int(i)] for i in np.atleast_1d(pick)]
            base = need_own_army // own_hidden_needed if own_hidden_needed else 0
            rem = need_own_army % own_hidden_needed if own_hidden_needed else 0
            for i, (r, c) in enumerate(own_cells):
                ownership[seat, r, c] = True
                armies[r, c] = base + (1 if i < rem else 0)

        ownership_neutral = passable & ~ownership[0] & ~ownership[1]
        general_positions = np.full((2, 2), -1, dtype=np.int32)
        for p in (0, 1):
            pos = np.argwhere(generals & ownership[p])
            if len(pos):
                general_positions[p] = pos[0]

        # Ensure own general from memory / visible.
        own_pos = np.argwhere(memory.own_general)
        if len(own_pos):
            general_positions[seat] = own_pos[0]
            generals[own_pos[0, 0], own_pos[0, 1]] = True
            ownership[seat, own_pos[0, 0], own_pos[0, 1]] = True

        state = GameState(
            armies=armies,
            ownership=ownership,
            ownership_neutral=ownership_neutral,
            generals=generals,
            castles=castles,
            mountains=mountains,
            passable=passable,
            general_positions=general_positions,
            time=int(obs.turn),
            winner=-1,
            pool_idx=0,
        )
        sim = emit_observation(state, seat)
        if not observations_match(sim, obs):
            continue
        particles.append(
            Particle(
                state=state,
                weight=1.0 / float(n),
                enemy_memory=empty_memory(H, W),
                history=(),
            )
        )

    if not particles:
        raise ValueError("maximum-entropy reconstruction produced no valid particles")

    particles = resample(normalize_weights(particles), n, rng)
    return BeliefState(
        seat=seat,
        particles=particles,
        config=config,
        collapsed=True,
    )


def recover_belief(
    belief: BeliefState,
    my_action: Action5,
    real_obs: Observation,
    memory: VisibleMemory,
    rng: np.random.Generator,
    *,
    policy: Optional[PolicyFn] = None,
) -> BeliefState:
    """Collapse recovery: rejuvenate, rewind+enumerate, then max-entropy.

    Never returns an observation-inconsistent set when a legal reconstruction
    exists. On total failure, keeps the previous belief unchanged.
    """
    # 1) Rejuvenation from stored ancestors.
    if any(p.history for p in belief.particles):
        rej = rejuvenate(belief, rng, policy=policy)
        positive = [p for p in rej.particles if p.weight > 0]
        if positive:
            # Verify current obs match.
            ok = all(
                observations_match(emit_observation(p.state, belief.seat), real_obs)
                for p in positive
            )
            if ok:
                return rej

    # 2) Rewind one turn when history exists: enumerate pass, top policy, vision.
    rewind_ok: list[Particle] = []
    for particle in belief.particles:
        if not particle.history:
            continue
        frame = particle.history[-1]
        temp = Particle(
            state=frame.state,
            weight=1.0,
            enemy_memory=particle.enemy_memory,
            enemy_prev_action=None,
        )
        probs = policy_action_probs(belief, temp, policy=policy)
        enemy_obs = emit_observation(frame.state, belief.enemy_seat)
        enemy_mem = update_memory(empty_memory(real_obs.H, real_obs.W), enemy_obs)
        mask = legal_mask(enemy_obs, enemy_mem)
        candidates = top_legal_actions(probs, mask, 8)
        for v in vision_changing_actions(
            frame.state, belief.seat, frame.my_action, _legal_enemy_actions(temp, belief)
        ):
            if v not in candidates:
                candidates.append(v)
        for enemy_action in candidates:
            # Replay the stored joint action that produced this observation.
            next_state = _apply_joint(
                frame.state, belief.seat, frame.my_action, enemy_action
            )
            sim = emit_observation(next_state, belief.seat)
            if observations_match(sim, real_obs):
                e_obs = emit_observation(next_state, belief.enemy_seat)
                rewind_ok.append(
                    Particle(
                        state=next_state,
                        weight=1.0,
                        enemy_memory=update_memory(enemy_mem, e_obs),
                        enemy_prev_action=enemy_action,
                        history=particle.history,
                    )
                )
                break

    if rewind_ok:
        particles = resample(normalize_weights(rewind_ok), belief.config.n_particles, rng)
        return BeliefState(
            seat=belief.seat,
            particles=particles,
            config=belief.config,
            collapsed=False,
        )

    # 3) Maximum-entropy reconstruction.
    try:
        return maximum_entropy_reconstruction(
            real_obs, belief.seat, memory, rng, config=belief.config
        )
    except ValueError:
        return belief  # keep last valid belief


def update_belief(
    belief: BeliefState,
    my_action: Action5,
    real_obs: Observation,
    memory: VisibleMemory,
    rng: np.random.Generator,
    *,
    policy: Optional[PolicyFn] = None,
    enemy_actions: Optional[Sequence[Action5]] = None,
) -> BeliefState:
    """One real-turn update: propose (unless injected), filter, recover if empty.

    Inject ``enemy_actions`` for deterministic fixtures. Recovery runs only when
    every particle fails the exact observation check.
    """
    from belief import filter_step
    from proposal import propose_enemy_actions

    if enemy_actions is None:
        enemy_actions = propose_enemy_actions(belief, rng, policy=policy)
    nxt = filter_step(belief, my_action, real_obs, enemy_actions, rng)
    if any(p.weight > 0.0 for p in nxt.particles):
        return nxt
    return recover_belief(
        belief, my_action, real_obs, memory, rng, policy=policy
    )
