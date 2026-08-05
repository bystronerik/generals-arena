"""Replay a curriculum prefix and reconstruct seat views, memory, and belief."""

from __future__ import annotations

import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
_BOT = _REPO / "bots" / "morpheus"
for entry in (_REPO, _REPO / "bots", _BOT):
    s = str(entry)
    if s not in sys.path:
        sys.path.insert(0, s)

from arena.records.trajectories import (  # noqa: E402
    Trajectory,
    read_trajectory,
    replay_states,
    require_same_era,
)
from belief import BeliefConfig, BeliefState, initialize_belief  # noqa: E402
from memory import VisibleMemory, empty_memory, update_memory  # noqa: E402
from observe import emit_observation  # noqa: E402
from recovery import update_belief  # noqa: E402
from state import from_engine  # noqa: E402
from training.morpheus.curriculum.schema import CurriculumItem  # noqa: E402

Action5 = tuple[int, int, int, int, int]


@dataclass
class SeatReconstruction:
    """One seat's reconstructed view after replaying a prefix."""

    seat: int
    observation: Any
    memory: VisibleMemory
    belief: BeliefState
    action_history: list[Action5]
    belief_seed: int


@dataclass
class PrefixReconstruction:
    """Both seats after replaying ``item.prefix_len`` joint actions."""

    item: CurriculumItem
    turn: int
    engine_state: Any
    seats: dict[int, SeatReconstruction]


def _as_action5(action: Sequence[int]) -> Action5:
    return (
        int(action[0]),
        int(action[1]),
        int(action[2]),
        int(action[3]),
        int(action[4]),
    )


def engine_fog_observation(state, seat: int):
    """Competition fog view from engine state (Part 10 verify path)."""
    from generals.core.game import get_observation

    return get_observation(state, int(seat))


def observation_fingerprint(obs) -> tuple:
    """Comparable fingerprint for fog verify."""
    # Engine Observation (masks + public totals).
    if hasattr(obs, "owned_land_count") and hasattr(obs, "armies"):
        return (
            int(np.asarray(obs.timestep).reshape(-1)[0]),
            int(np.asarray(obs.owned_land_count).reshape(-1)[0]),
            int(np.asarray(obs.owned_army_count).reshape(-1)[0]),
            int(np.asarray(obs.opponent_land_count).reshape(-1)[0]),
            int(np.asarray(obs.opponent_army_count).reshape(-1)[0]),
            tuple(np.asarray(obs.armies, dtype=np.int32).reshape(-1).tolist()),
            tuple(np.asarray(obs.generals, dtype=bool).reshape(-1).tolist()),
            tuple(np.asarray(obs.owned_cells, dtype=bool).reshape(-1).tolist()),
            tuple(np.asarray(obs.opponent_cells, dtype=bool).reshape(-1).tolist()),
            tuple(np.asarray(obs.fog_cells, dtype=bool).reshape(-1).tolist()),
        )
    # Morpheus wire / array observation.
    return (
        int(obs.turn),
        int(obs.H),
        int(obs.W),
        int(obs.my_land),
        int(obs.my_army),
        int(obs.opp_land),
        int(obs.opp_army),
        tuple(np.asarray(obs.type_grid, dtype=np.int32).reshape(-1).tolist()),
        tuple(np.asarray(obs.owner_grid, dtype=np.int32).reshape(-1).tolist()),
        tuple(np.asarray(obs.army_grid, dtype=np.int32).reshape(-1).tolist()),
    )


def load_trajectory_for_item(
    item: CurriculumItem,
    *,
    trajectories_root: Path | None = None,
    traj: Trajectory | None = None,
) -> Trajectory | None:
    """Load the source trajectory for classes 1–4. Class 5 has none."""
    if traj is not None:
        return traj
    if item.class_id == 5 or not item.trajectory_relpath:
        return None
    root = trajectories_root or Path(".")
    path = Path(item.trajectory_relpath)
    if not path.is_file():
        path = root / item.trajectory_relpath
    return read_trajectory(path)


def replay_prefix_states(
    *,
    traj: Trajectory | None,
    map_seed: int,
    prefix_len: int,
    mode: str = "competition",
    engine: str | None = None,
) -> tuple[int, Any]:
    """Replay ``prefix_len`` joint actions and return ``(turn, state)``."""
    if traj is None:
        if prefix_len != 0:
            raise ValueError("traj-less replay requires prefix_len == 0")
        from arena.matches.loop import make_board
        from generals import GeneralsEnv

        env = GeneralsEnv(mode=mode)
        state = make_board(env, int(map_seed))
        return 0, state

    require_same_era(traj, engine=engine)
    last_turn = 0
    last_state = None
    for turn, state, _info in replay_states(traj):
        last_turn, last_state = int(turn), state
        if turn >= prefix_len:
            return int(turn), state
    if last_state is None:
        raise RuntimeError("empty trajectory replay")
    if prefix_len > last_turn:
        raise ValueError(
            f"prefix_len {prefix_len} exceeds trajectory length {last_turn}"
        )
    return last_turn, last_state


def _snapshot(
    item: CurriculumItem,
    *,
    turn: int,
    engine_state: Any,
    seat_obs: dict[int, Any],
    seat_mem: dict[int, VisibleMemory],
    seat_belief: dict[int, BeliefState],
    seat_hist: dict[int, list[Action5]],
    seeds: dict[int, int],
) -> PrefixReconstruction:
    return PrefixReconstruction(
        item=item,
        turn=turn,
        engine_state=engine_state,
        seats={
            seat: SeatReconstruction(
                seat=seat,
                observation=seat_obs[seat],
                memory=seat_mem[seat],
                belief=seat_belief[seat],
                action_history=list(seat_hist[seat]),
                belief_seed=seeds[seat],
            )
            for seat in (0, 1)
        },
    )


def iter_prefix_reconstructions(
    items: Sequence[CurriculumItem],
    *,
    trajectories_root: Path | None = None,
    traj: Trajectory | None = None,
    n_particles: int = 8,
    update_belief_flag: bool = True,
    engine: str | None = None,
) -> Iterator[tuple[CurriculumItem, PrefixReconstruction]]:
    """
    Replay one game once and yield a reconstruction at each item's prefix.

    All items must share the same ``game_id`` / trajectory. Belief RNG is shared
    per seat across prefixes. Yields in ascending ``prefix_len`` order; items
    with equal ``prefix_len`` all receive the same turn snapshot.
    """
    if not items:
        return

    ordered = sorted(items, key=lambda it: (int(it.prefix_len), it.item_id))
    game_ids = {str(it.game_id or "") for it in ordered}
    if len(game_ids) > 1:
        raise ValueError(
            f"iter_prefix_reconstructions requires one game_id, got {sorted(game_ids)}"
        )

    first = ordered[0]
    source = load_trajectory_for_item(
        first, trajectories_root=trajectories_root, traj=traj
    )
    if source is not None:
        require_same_era(source, engine=engine or first.engine_version)

    # Shared seat seeds (prefix_len no longer enters the hash).
    seeds = {0: first.belief_seed(0), 1: first.belief_seed(1)}
    for it in ordered[1:]:
        for seat in (0, 1):
            if it.belief_seed(seat) != seeds[seat]:
                raise ValueError(
                    f"belief seed mismatch within game for {it.item_id} seat {seat}"
                )

    cfg = BeliefConfig(n_particles=int(n_particles))
    seat_mem: dict[int, VisibleMemory] = {}
    seat_belief: dict[int, BeliefState] = {}
    seat_hist: dict[int, list[Action5]] = {0: [], 1: []}
    seat_obs: dict[int, Any] = {}
    rngs = {s: np.random.default_rng(seeds[s]) for s in (0, 1)}
    frames_by_turn = {f.turn: f for f in source.frames} if source is not None else {}

    by_prefix: dict[int, list[CurriculumItem]] = {}
    for it in ordered:
        by_prefix.setdefault(int(it.prefix_len), []).append(it)
    targets = sorted(by_prefix.keys())
    max_prefix = targets[-1]
    target_i = 0

    if source is None:
        if max_prefix != 0:
            raise ValueError("traj-less reconstruct requires all prefix_len == 0")
        turn, engine_state = replay_prefix_states(
            traj=None,
            map_seed=first.map_seed,
            prefix_len=0,
            engine=engine,
        )
        state = from_engine(engine_state)
        for seat in (0, 1):
            obs = emit_observation(state, seat, as_arrays=True)
            seat_obs[seat] = obs
            seat_mem[seat] = update_memory(empty_memory(obs.H, obs.W), obs)
            if update_belief_flag:
                seat_belief[seat] = initialize_belief(
                    obs, seat=seat, rng=rngs[seat], config=cfg
                )
            else:
                seat_belief[seat] = BeliefState(
                    seat=seat, particles=[], config=cfg, collapsed=True
                )
        for it in by_prefix[0]:
            yield it, _snapshot(
                it,
                turn=turn,
                engine_state=engine_state,
                seat_obs=seat_obs,
                seat_mem=seat_mem,
                seat_belief=seat_belief,
                seat_hist=seat_hist,
                seeds=seeds,
            )
        return

    last_turn = 0
    last_engine_state = None
    for turn, engine_state, _info in replay_states(source):
        last_turn, last_engine_state = int(turn), engine_state
        state = from_engine(engine_state)
        for seat in (0, 1):
            obs = emit_observation(state, seat, as_arrays=True)
            seat_obs[seat] = obs
            if turn == 0:
                seat_mem[seat] = update_memory(empty_memory(obs.H, obs.W), obs)
                if update_belief_flag:
                    seat_belief[seat] = initialize_belief(
                        obs, seat=seat, rng=rngs[seat], config=cfg
                    )
                else:
                    seat_belief[seat] = BeliefState(
                        seat=seat, particles=[], config=cfg, collapsed=True
                    )
            else:
                seat_mem[seat] = update_memory(seat_mem[seat], obs)
                frame = frames_by_turn[turn]
                my_action = _as_action5(
                    frame.action_a if seat == 0 else frame.action_b
                )
                enemy_action = _as_action5(
                    frame.action_b if seat == 0 else frame.action_a
                )
                seat_hist[seat].append(my_action)
                if update_belief_flag and seat_belief[seat].n > 0:
                    n = seat_belief[seat].n
                    seat_belief[seat] = update_belief(
                        seat_belief[seat],
                        my_action,
                        obs,
                        seat_mem[seat],
                        rngs[seat],
                        enemy_actions=[enemy_action] * n,
                    )

        while target_i < len(targets) and turn >= targets[target_i]:
            for it in by_prefix[targets[target_i]]:
                yield it, _snapshot(
                    it,
                    turn=last_turn,
                    engine_state=last_engine_state,
                    seat_obs=seat_obs,
                    seat_mem=seat_mem,
                    seat_belief=seat_belief,
                    seat_hist=seat_hist,
                    seeds=seeds,
                )
            target_i += 1

        if turn >= max_prefix:
            break

    if last_engine_state is None:
        raise RuntimeError(f"failed to replay {first.item_id}")
    if target_i < len(targets):
        missing = targets[target_i]
        raise ValueError(
            f"prefix_len {missing} exceeds trajectory length {last_turn}"
        )


def reconstruct_prefixes(
    items: Sequence[CurriculumItem],
    *,
    trajectories_root: Path | None = None,
    traj: Trajectory | None = None,
    n_particles: int = 8,
    update_belief_flag: bool = True,
    engine: str | None = None,
) -> dict[str, PrefixReconstruction]:
    """Reconstruct many prefixes of one game; map ``item_id`` → reconstruction."""
    return {
        it.item_id: recon
        for it, recon in iter_prefix_reconstructions(
            items,
            trajectories_root=trajectories_root,
            traj=traj,
            n_particles=n_particles,
            update_belief_flag=update_belief_flag,
            engine=engine,
        )
    }


def reconstruct_prefix(
    item: CurriculumItem,
    *,
    trajectories_root: Path | None = None,
    traj: Trajectory | None = None,
    n_particles: int = 8,
    update_belief_flag: bool = True,
    engine: str | None = None,
) -> PrefixReconstruction:
    """
    Replay from turn 0 and rebuild both seats' obs, memory, belief, history.

    Belief updates inject the recorded enemy action so particle RNG only
    affects initialization and recovery resampling.
    """
    results = reconstruct_prefixes(
        [item],
        trajectories_root=trajectories_root,
        traj=traj,
        n_particles=n_particles,
        update_belief_flag=update_belief_flag,
        engine=engine,
    )
    return results[item.item_id]


def fog_observations_at_prefix(
    item: CurriculumItem,
    *,
    trajectories_root: Path | None = None,
    traj: Trajectory | None = None,
    engine: str | None = None,
) -> dict[int, Any]:
    """Engine fog observations for both seats at the prefix end."""
    source = load_trajectory_for_item(
        item, trajectories_root=trajectories_root, traj=traj
    )
    _turn, state = replay_prefix_states(
        traj=source,
        map_seed=item.map_seed,
        prefix_len=item.prefix_len,
        engine=engine or item.engine_version,
    )
    return {0: engine_fog_observation(state, 0), 1: engine_fog_observation(state, 1)}
