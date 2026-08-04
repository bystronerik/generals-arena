"""Belief-filter measurements for Morpheus Part 05.

Replays bootstrap trajectories and reports recovery rate / p99 cost after
forced proposal mismatch, opponent-belief log loss and survival, and an ESS
threshold sweep.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
_BOT = _REPO / "bots" / "morpheus"
for entry in (_REPO, _REPO / "bots", _BOT):
    s = str(entry)
    if s not in sys.path:
        sys.path.insert(0, s)

from arena.records.trajectories import read_trajectory, replay_states
from belief import (
    BeliefConfig,
    filter_step,
    initialize_belief,
    pass_action,
    unique_particle_count,
)
from memory import empty_memory, update_memory
from observe import emit_observation, observations_match
from proposal import policy_action_probs, propose_enemy_actions
from recovery import recover_belief, update_belief
from state import from_engine
from action import encode_action


def _traj_paths(trajectories_dir: Path, max_games: int) -> list[Path]:
    paths = sorted(Path(trajectories_dir).glob("*.traj.jsonl.gz"))
    if max_games > 0:
        paths = paths[:max_games]
    return paths


def _seat_actions(frame, seat: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if seat == 0:
        return frame.action_a, frame.action_b
    return frame.action_b, frame.action_a


def _as_action5(action) -> tuple[int, int, int, int, int]:
    return (int(action[0]), int(action[1]), int(action[2]), int(action[3]), int(action[4]))


def measure_belief_recovery(
    trajectories_dir: Path,
    *,
    max_games: int = 4,
    n_particles: int = 16,
    force_proposal_mismatch: bool = True,
    max_turns: int = 40,
) -> dict[str, Any]:
    """Forced-mismatch recovery rate and p99 wall cost on recorded games."""
    paths = _traj_paths(trajectories_dir, max_games)
    recoveries = 0
    attempts = 0
    costs_ms: list[float] = []
    game_count = 0

    for path in paths:
        traj = read_trajectory(path)
        game_count += 1
        rng = np.random.default_rng(hash(path.name) & 0xFFFFFFFF)
        cfg = BeliefConfig(n_particles=n_particles, min_general_distance=17)
        belief = None
        memory = None
        seat = 0
        turns = list(replay_states(traj))
        for i, (_turn, engine_state, _info) in enumerate(turns):
            if i >= max_turns:
                break
            state = from_engine(engine_state)
            obs = emit_observation(state, seat)
            if belief is None:
                try:
                    belief = initialize_belief(obs, seat=seat, rng=rng, config=cfg)
                except ValueError:
                    break
                memory = update_memory(empty_memory(obs.H, obs.W), obs)
                continue
            memory = update_memory(memory, obs)
            if i - 1 >= len(traj.frames):
                break
            fr = traj.frames[i - 1]
            my_a, _enemy_a = _seat_actions(fr, seat)
            my_a = _as_action5(my_a)
            t0 = time.perf_counter()
            if force_proposal_mismatch:
                wrong = [pass_action()] * belief.n
                bad_obs = obs.__class__(
                    H=obs.H,
                    W=obs.W,
                    turn=obs.turn,
                    my_land=obs.my_land,
                    my_army=obs.my_army,
                    opp_land=obs.opp_land,
                    opp_army=obs.opp_army + 7,
                    type_grid=obs.type_grid,
                    owner_grid=obs.owner_grid,
                    army_grid=obs.army_grid,
                )
                filter_step(belief, my_a, bad_obs, wrong, rng)
                attempts += 1
                recovered = recover_belief(belief, my_a, obs, memory, rng)
                ok = any(
                    p.weight > 0
                    and observations_match(emit_observation(p.state, seat), obs)
                    for p in recovered.particles
                )
                if ok:
                    recoveries += 1
                    belief = recovered
            else:
                enemy_a = _as_action5(_seat_actions(fr, seat)[1])
                belief = update_belief(
                    belief,
                    my_a,
                    obs,
                    memory,
                    rng,
                    enemy_actions=[enemy_a] * belief.n,
                )
            costs_ms.append((time.perf_counter() - t0) * 1000.0)

    costs = np.asarray(costs_ms, dtype=np.float64)
    rate = float(recoveries / attempts) if attempts else 0.0
    return {
        "measurement": "belief-recovery",
        "game_count": game_count,
        "attempts": attempts,
        "recoveries": recoveries,
        "recovery_rate": rate,
        "force_proposal_mismatch": force_proposal_mismatch,
        "n_particles": n_particles,
        "p50_cost_ms": float(np.quantile(costs, 0.50)) if costs.size else 0.0,
        "p99_cost_ms": float(np.quantile(costs, 0.99)) if costs.size else 0.0,
        "selected_recovery_bounds": {
            "recovery_lag": 8,
            "beam_width": 8,
            "max_completed_histories": 16,
            "max_replayed_transitions": 128,
        },
    }


def measure_opponent_belief(
    trajectories_dir: Path,
    *,
    max_games: int = 4,
    n_particles: int = 16,
    max_turns: int = 40,
) -> dict[str, Any]:
    """Enemy-action log loss under level-zero proposal and particle survival."""
    paths = _traj_paths(trajectories_dir, max_games)
    log_losses: list[float] = []
    survivals: list[float] = []
    game_count = 0

    for path in paths:
        traj = read_trajectory(path)
        game_count += 1
        rng = np.random.default_rng(hash(path.name) & 0xFFFFFFFF)
        cfg = BeliefConfig(n_particles=n_particles, min_general_distance=17)
        belief = None
        memory = None
        seat = 0
        turns = list(replay_states(traj))
        for i, (_turn, engine_state, _info) in enumerate(turns):
            if i >= max_turns:
                break
            state = from_engine(engine_state)
            obs = emit_observation(state, seat)
            if belief is None:
                try:
                    belief = initialize_belief(obs, seat=seat, rng=rng, config=cfg)
                except ValueError:
                    break
                memory = update_memory(empty_memory(obs.H, obs.W), obs)
                continue
            memory = update_memory(memory, obs)
            if i - 1 >= len(traj.frames):
                break
            fr = traj.frames[i - 1]
            my_a, enemy_a = _seat_actions(fr, seat)
            my_a = _as_action5(my_a)
            enemy_a = _as_action5(enemy_a)
            # Mixture probability of the recorded enemy action under level-zero
            # proposals across the current particle set.
            mix = 0.0
            wsum = 0.0
            for particle in belief.particles:
                w = max(particle.weight, 0.0)
                if w <= 0.0:
                    continue
                probs = policy_action_probs(belief, particle, policy=None)
                mix += w * float(probs[encode_action(enemy_a)])
                wsum += w
            p = (mix / wsum) if wsum > 0.0 else 0.0
            log_losses.append(float(-np.log(max(p, 1e-12))))

            before = sum(1 for p_ in belief.particles if p_.weight > 0)
            proposed = propose_enemy_actions(belief, rng, policy=None)
            nxt = filter_step(belief, my_a, obs, proposed, rng)
            after = sum(1 for p_ in nxt.particles if p_.weight > 0)
            if after == 0:
                nxt = recover_belief(belief, my_a, obs, memory, rng)
                after = sum(1 for p_ in nxt.particles if p_.weight > 0)
            survivals.append(float(after) / float(max(before, 1)))
            belief = nxt

    ll = np.asarray(log_losses, dtype=np.float64)
    surv = np.asarray(survivals, dtype=np.float64)
    return {
        "measurement": "opponent-belief",
        "game_count": game_count,
        "sample_count": int(ll.size),
        "mean_enemy_action_log_loss": float(ll.mean()) if ll.size else 0.0,
        "p50_enemy_action_log_loss": float(np.quantile(ll, 0.5)) if ll.size else 0.0,
        "mean_particle_survival": float(surv.mean()) if surv.size else 0.0,
        "recursive_opponent_particles": False,
        "recommendation": "keep_level_zero",
    }


def measure_ess_threshold(
    trajectories_dir: Path,
    *,
    max_games: int = 2,
    n_particles: int = 16,
    max_turns: int = 30,
    thresholds: tuple[float, ...] = (0.25, 0.5, 0.75),
) -> dict[str, Any]:
    """Sweep ESS resample thresholds; report uniqueness, survival, cost."""
    paths = _traj_paths(trajectories_dir, max_games)
    results = []
    for frac in thresholds:
        unique_counts: list[int] = []
        survivals: list[float] = []
        recoveries = 0
        steps = 0
        costs_ms: list[float] = []
        for path in paths:
            traj = read_trajectory(path)
            rng = np.random.default_rng(
                (hash(path.name) ^ int(frac * 1000)) & 0xFFFFFFFF
            )
            cfg = BeliefConfig(
                n_particles=n_particles,
                ess_threshold_fraction=frac,
                min_general_distance=17,
            )
            belief = None
            memory = None
            seat = 0
            turns = list(replay_states(traj))
            for i, (_turn, engine_state, _info) in enumerate(turns):
                if i >= max_turns:
                    break
                state = from_engine(engine_state)
                obs = emit_observation(state, seat)
                if belief is None:
                    try:
                        belief = initialize_belief(
                            obs, seat=seat, rng=rng, config=cfg
                        )
                    except ValueError:
                        break
                    memory = update_memory(empty_memory(obs.H, obs.W), obs)
                    continue
                memory = update_memory(memory, obs)
                if i - 1 >= len(traj.frames):
                    break
                fr = traj.frames[i - 1]
                my_a, enemy_a = _seat_actions(fr, seat)
                my_a = _as_action5(my_a)
                enemy_a = _as_action5(enemy_a)
                t0 = time.perf_counter()
                before = belief.n
                nxt = update_belief(
                    belief,
                    my_a,
                    obs,
                    memory,
                    rng,
                    enemy_actions=[enemy_a] * belief.n,
                )
                if nxt.collapsed:
                    recoveries += 1
                costs_ms.append((time.perf_counter() - t0) * 1000.0)
                unique_counts.append(unique_particle_count(nxt))
                survivals.append(
                    float(sum(1 for p in nxt.particles if p.weight > 0))
                    / float(before)
                )
                belief = nxt
                steps += 1
        costs = np.asarray(costs_ms, dtype=np.float64)
        results.append(
            {
                "ess_threshold_fraction": frac,
                "steps": steps,
                "mean_unique_particles": float(np.mean(unique_counts))
                if unique_counts
                else 0.0,
                "mean_survival": float(np.mean(survivals)) if survivals else 0.0,
                "recovery_frequency": float(recoveries / steps) if steps else 0.0,
                "p99_update_cost_ms": float(np.quantile(costs, 0.99))
                if costs.size
                else 0.0,
            }
        )

    return {
        "measurement": "ess-threshold",
        "n_particles": n_particles,
        "sweep": results,
        "selected_ess_threshold_fraction": 0.5,
        "deployment_note": (
            "Record selected_ess_threshold_fraction in the deployment "
            "configuration; default remains half the particle count."
        ),
    }
