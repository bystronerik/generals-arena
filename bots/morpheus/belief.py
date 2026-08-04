"""Weighted particle belief filter for Morpheus.

Owns initialization, exact observation filtering, ESS resampling, and the
shared configuration defaults. Proposal injection and recovery live in sibling
modules so fixtures do not require a trained checkpoint.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

from _common.wire import Observation
from memory import (
    TYPE_FOG,
    TYPE_MOUNTAIN,
    TYPE_STRUCTURE_FOG,
    VisibleMemory,
    empty_memory,
    update_memory,
)
from observe import emit_observation, observations_match, visibility_mask
from state import GameState
from transition import transition

Array = np.ndarray

# Defaults — initial guesses; measurements may replace them in deployment config.
N_PARTICLES = 64
ESS_THRESHOLD_FRACTION = 0.5
RECOVERY_LAG = 8
BEAM_WIDTH = 8
MAX_COMPLETED_HISTORIES = 16
MAX_REPLAYED_TRANSITIONS = 128
MIN_GENERAL_DISTANCE = 17


Action5 = tuple[int, int, int, int, int]
ProposalFn = Callable[
    ["BeliefState", Sequence["Particle"], np.random.Generator],
    Sequence[Action5],
]


@dataclass(frozen=True)
class BeliefConfig:
    n_particles: int = N_PARTICLES
    ess_threshold_fraction: float = ESS_THRESHOLD_FRACTION
    recovery_lag: int = RECOVERY_LAG
    beam_width: int = BEAM_WIDTH
    max_completed_histories: int = MAX_COMPLETED_HISTORIES
    max_replayed_transitions: int = MAX_REPLAYED_TRANSITIONS
    min_general_distance: int = MIN_GENERAL_DISTANCE


@dataclass(frozen=True)
class HistoryFrame:
    """State and joint actions immediately before one real transition."""

    state: GameState
    my_action: Action5
    enemy_action: Action5
    observation_after: Observation


@dataclass
class Particle:
    state: GameState
    weight: float
    enemy_memory: VisibleMemory
    enemy_prev_action: Optional[Action5] = None
    history: tuple[HistoryFrame, ...] = ()


@dataclass
class BeliefState:
    """Weighted particle set from Morpheus's seat."""

    seat: int  # 0 or 1 — Morpheus player index in GameState
    particles: list[Particle]
    config: BeliefConfig = field(default_factory=BeliefConfig)
    collapsed: bool = False  # True after max-entropy reconstruction

    @property
    def enemy_seat(self) -> int:
        return 1 - self.seat

    @property
    def n(self) -> int:
        return len(self.particles)


def ess(weights: Sequence[float]) -> float:
    """Effective sample size ``1 / sum(w_i^2)`` for normalized weights."""
    w = np.asarray(weights, dtype=np.float64)
    if w.size == 0:
        return 0.0
    total = float(w.sum())
    if total <= 0.0:
        return 0.0
    w = w / total
    return float(1.0 / np.sum(w * w))


def ess_fraction(belief: BeliefState) -> float:
    n_cfg = max(belief.config.n_particles, 1)
    if belief.collapsed:
        return 1.0 / float(n_cfg)
    if belief.n == 0:
        return 0.0
    return ess([p.weight for p in belief.particles]) / float(n_cfg)


def normalize_weights(particles: list[Particle]) -> list[Particle]:
    total = sum(max(p.weight, 0.0) for p in particles)
    if total <= 0.0:
        return particles
    return [
        Particle(
            state=p.state,
            weight=max(p.weight, 0.0) / total,
            enemy_memory=p.enemy_memory,
            enemy_prev_action=p.enemy_prev_action,
            history=p.history,
        )
        for p in particles
    ]


def resample(
    particles: list[Particle],
    n: int,
    rng: np.random.Generator,
) -> list[Particle]:
    """Multinomial resample to ``n`` equal-weight particles."""
    if not particles or n <= 0:
        return []
    weights = np.asarray([max(p.weight, 0.0) for p in particles], dtype=np.float64)
    total = float(weights.sum())
    if total <= 0.0:
        idx = rng.integers(0, len(particles), size=n)
    else:
        probs = weights / total
        idx = rng.choice(len(particles), size=n, replace=True, p=probs)
    out: list[Particle] = []
    w = 1.0 / float(n)
    for i in idx:
        src = particles[int(i)]
        out.append(
            Particle(
                state=src.state,
                weight=w,
                enemy_memory=src.enemy_memory,
                enemy_prev_action=src.enemy_prev_action,
                history=src.history,
            )
        )
    return out


def maybe_resample(
    belief: BeliefState,
    rng: np.random.Generator,
) -> BeliefState:
    """Resample when ESS falls below half the configured particle count."""
    if belief.n == 0:
        return belief
    threshold = belief.config.ess_threshold_fraction * belief.config.n_particles
    if ess([p.weight for p in belief.particles]) >= threshold:
        return belief
    return BeliefState(
        seat=belief.seat,
        particles=resample(belief.particles, belief.config.n_particles, rng),
        config=belief.config,
        collapsed=belief.collapsed,
    )


def _bfs_distances(passable: Array, origin: tuple[int, int]) -> Array:
    """BFS step counts over 4-connected passable cells; ``-1`` unreachable."""
    H, W = passable.shape
    dist = np.full((H, W), -1, dtype=np.int32)
    or_, oc = origin
    if not (0 <= or_ < H and 0 <= oc < W) or not passable[or_, oc]:
        return dist
    dist[or_, oc] = 0
    queue = [(or_, oc)]
    head = 0
    while head < len(queue):
        r, c = queue[head]
        head += 1
        d = int(dist[r, c])
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and passable[nr, nc] and dist[nr, nc] < 0:
                dist[nr, nc] = d + 1
                queue.append((nr, nc))
    return dist


def _terrain_from_first_obs(obs: Observation) -> tuple[Array, Array]:
    """Return ``(mountains, passable)`` from the first-frame observation."""
    types = np.asarray(obs.type_grid, dtype=np.int32)
    mountains = (types == TYPE_MOUNTAIN) | (types == TYPE_STRUCTURE_FOG)
    # Type 0 fog and every non-mountain cell are passable base at start.
    passable = ~mountains
    return mountains, passable


def _own_general_from_obs(obs: Observation) -> tuple[int, int]:
    types = np.asarray(obs.type_grid, dtype=np.int32)
    owners = np.asarray(obs.owner_grid, dtype=np.int32)
    pos = np.argwhere((types == 4) & (owners == 1))
    if len(pos) == 0:
        raise ValueError("observation has no visible own general")
    return int(pos[0, 0]), int(pos[0, 1])


def legal_enemy_general_candidates(
    obs: Observation,
    *,
    min_distance: int = MIN_GENERAL_DISTANCE,
) -> list[tuple[int, int]]:
    """Uniform prior support for the initial enemy general."""
    mountains, passable = _terrain_from_first_obs(obs)
    own_r, own_c = _own_general_from_obs(obs)
    types = np.asarray(obs.type_grid, dtype=np.int32)
    owners = np.asarray(obs.owner_grid, dtype=np.int32)
    visible = (types != TYPE_FOG) & (types != TYPE_STRUCTURE_FOG)
    # Own vision at seat ownership = owned cells in the obs.
    own_cells = owners == 1
    # Prefer visibility from owned cells when available; fall back to type mask.
    if np.any(own_cells):
        vis = visibility_mask(own_cells)
    else:
        vis = visible

    dist = _bfs_distances(passable, (own_r, own_c))
    candidates: list[tuple[int, int]] = []
    H, W = passable.shape
    for r in range(H):
        for c in range(W):
            if not passable[r, c]:
                continue
            if vis[r, c]:
                continue
            # Must be fog type 0 (can hide general) or never contradict.
            if types[r, c] not in (TYPE_FOG,):
                # Structure fog is mountain at start — already excluded by passable.
                continue
            if int(dist[r, c]) < min_distance:
                continue
            candidates.append((r, c))
    return candidates


def _state_from_initial(
    obs: Observation,
    seat: int,
    enemy_general: tuple[int, int],
) -> GameState:
    """Build a complete initial GameState for one particle."""
    H, W = int(obs.H), int(obs.W)
    mountains, passable = _terrain_from_first_obs(obs)
    own_r, own_c = _own_general_from_obs(obs)
    er, ec = enemy_general
    enemy_seat = 1 - seat

    armies = np.zeros((H, W), dtype=np.int32)
    ownership = np.zeros((2, H, W), dtype=bool)
    generals = np.zeros((H, W), dtype=bool)
    castles = np.zeros((H, W), dtype=bool)

    armies[own_r, own_c] = 1
    armies[er, ec] = 1
    ownership[seat, own_r, own_c] = True
    ownership[enemy_seat, er, ec] = True
    generals[own_r, own_c] = True
    generals[er, ec] = True
    ownership_neutral = passable & ~ownership[0] & ~ownership[1]

    general_positions = np.full((2, 2), -1, dtype=np.int32)
    general_positions[seat] = (own_r, own_c)
    general_positions[enemy_seat] = (er, ec)

    # Visible own cells beyond the general (should be none at true start, but
    # copy any owned cells from the observation to stay rule-consistent).
    owners = np.asarray(obs.owner_grid, dtype=np.int32)
    armies_obs = np.asarray(obs.army_grid, dtype=np.int32)
    types = np.asarray(obs.type_grid, dtype=np.int32)
    visible = (types != TYPE_FOG) & (types != TYPE_STRUCTURE_FOG)
    for r, c in zip(*np.where(visible & (owners == 1))):
        ownership[seat, r, c] = True
        ownership[enemy_seat, r, c] = False
        ownership_neutral[r, c] = False
        armies[r, c] = int(armies_obs[r, c])
        if types[r, c] == 4:
            generals[r, c] = True

    return GameState(
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


def _public_totals_match(state: GameState, obs: Observation, seat: int) -> bool:
    info_land = (
        int(state.ownership[seat].sum()),
        int(state.ownership[1 - seat].sum()),
    )
    info_army = (
        int((state.armies * state.ownership[seat]).sum()),
        int((state.armies * state.ownership[1 - seat]).sum()),
    )
    return (
        info_land[0] == int(obs.my_land)
        and info_land[1] == int(obs.opp_land)
        and info_army[0] == int(obs.my_army)
        and info_army[1] == int(obs.opp_army)
    )


def initialize_belief(
    obs: Observation,
    seat: int,
    rng: np.random.Generator,
    *,
    config: Optional[BeliefConfig] = None,
) -> BeliefState:
    """Sample the initial particle set from the conditioned uniform prior."""
    config = config or BeliefConfig()
    candidates = legal_enemy_general_candidates(
        obs, min_distance=config.min_general_distance
    )
    if not candidates:
        # Fall back to any fog passable cell at distance >= 1 so tests on tiny
        # boards can still construct a legal particle.
        candidates = legal_enemy_general_candidates(obs, min_distance=1)
    if not candidates:
        raise ValueError("no legal enemy-general candidates for initial belief")

    n = config.n_particles
    choices = [candidates[int(i)] for i in rng.integers(0, len(candidates), size=n)]
    # Enemy memory of Morpheus starts empty; updated when we build enemy obs.
    particles: list[Particle] = []
    w = 1.0 / float(n)
    for gen in choices:
        state = _state_from_initial(obs, seat, gen)
        if not _public_totals_match(state, obs, seat):
            continue
        # Verify simulated observation matches the real first frame.
        sim = emit_observation(state, seat)
        if not observations_match(sim, obs):
            continue
        enemy_mem = empty_memory(obs.H, obs.W)
        particles.append(
            Particle(state=state, weight=w, enemy_memory=enemy_mem, history=())
        )

    if not particles:
        raise ValueError("initial belief produced no observation-consistent particles")

    # Pad by resampling if some candidates failed the exact match.
    if len(particles) < n:
        particles = resample(normalize_weights(particles), n, rng)
    else:
        particles = normalize_weights(particles[:n])
        if len(particles) < n:
            particles = resample(particles, n, rng)

    return BeliefState(seat=seat, particles=particles, config=config, collapsed=False)


def _append_history(
    history: tuple[HistoryFrame, ...],
    frame: HistoryFrame,
    lag: int,
) -> tuple[HistoryFrame, ...]:
    merged = history + (frame,)
    if len(merged) > lag:
        merged = merged[-lag:]
    return merged


def filter_step(
    belief: BeliefState,
    my_action: Action5,
    real_obs: Observation,
    enemy_actions: Sequence[Action5],
    rng: np.random.Generator,
) -> BeliefState:
    """Apply ``(my_action, enemy_action)`` per particle and hard-filter.

    ``enemy_actions`` must align with ``belief.particles``. Proposal injection
    stays outside this function.
    """
    if len(enemy_actions) != belief.n:
        raise ValueError("enemy_actions length must match particle count")

    survivors: list[Particle] = []
    for particle, enemy_action in zip(belief.particles, enemy_actions):
        actions = np.zeros((2, 5), dtype=np.int32)
        actions[belief.seat] = np.asarray(my_action, dtype=np.int32)
        actions[belief.enemy_seat] = np.asarray(enemy_action, dtype=np.int32)
        next_state, _info = transition(particle.state, actions)
        sim = emit_observation(next_state, belief.seat)
        if not observations_match(sim, real_obs):
            survivors.append(
                Particle(
                    state=particle.state,
                    weight=0.0,
                    enemy_memory=particle.enemy_memory,
                    enemy_prev_action=particle.enemy_prev_action,
                    history=particle.history,
                )
            )
            continue

        enemy_obs = emit_observation(next_state, belief.enemy_seat)
        enemy_mem = update_memory(particle.enemy_memory, enemy_obs)
        frame = HistoryFrame(
            state=particle.state,
            my_action=tuple(int(x) for x in my_action),  # type: ignore[arg-type]
            enemy_action=tuple(int(x) for x in enemy_action),  # type: ignore[arg-type]
            observation_after=real_obs,
        )
        survivors.append(
            Particle(
                state=next_state,
                weight=particle.weight,  # importance ratio 1 under shared policy
                enemy_memory=enemy_mem,
                enemy_prev_action=tuple(int(x) for x in enemy_action),  # type: ignore[arg-type]
                history=_append_history(
                    particle.history, frame, belief.config.recovery_lag
                ),
            )
        )

    positive = [p for p in survivors if p.weight > 0.0]
    if not positive:
        # Caller (recovery) must handle collapse. Return zero-weight set.
        return BeliefState(
            seat=belief.seat,
            particles=survivors,
            config=belief.config,
            collapsed=False,
        )

    normalized = normalize_weights(positive)
    # Keep configured count via resample pad when needed.
    if len(normalized) < belief.config.n_particles:
        normalized = resample(normalized, belief.config.n_particles, rng)
    out = BeliefState(
        seat=belief.seat,
        particles=normalized,
        config=belief.config,
        collapsed=False,
    )
    return maybe_resample(out, rng)


def unique_particle_count(belief: BeliefState) -> int:
    """Count distinct hidden general cells among positive-weight particles."""
    keys = set()
    for p in belief.particles:
        if p.weight <= 0.0:
            continue
        g = p.state.general_positions[belief.enemy_seat]
        keys.add((int(g[0]), int(g[1])))
    return len(keys)


def pass_action() -> Action5:
    return (1, 0, 0, 0, 0)


def as_action5(action: Sequence[int]) -> Action5:
    return (int(action[0]), int(action[1]), int(action[2]), int(action[3]), int(action[4]))
