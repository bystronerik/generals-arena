"""Build one seat-local training sample from a curriculum item.

Reconstruction is a local prep step. Write samples to the replay buffer, then
point Modal ``train`` at that buffer directory.

Materialize continues one belief history per game (shared seat seed across
prefixes). Rematerialize buffers after the seed contract change; do not reuse
``*.sample.npz`` built under the old per-prefix seed.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from arena.records.trajectories import Trajectory, read_trajectory, replay_states
from training.morpheus.curriculum.reconstruct import (
    PrefixReconstruction,
    reconstruct_prefix,
)
from training.morpheus.curriculum.schema import CurriculumItem
from training.morpheus.objective.targets import SeatTargets, build_seat_targets
from training.morpheus.self_play.schema import SparsePolicy

REPO = Path(__file__).resolve().parents[3]
MORPHEUS_BOT = REPO / "bots" / "morpheus"

Array = np.ndarray


def _ensure_bot_path() -> None:
    for entry in (REPO, REPO / "bots", MORPHEUS_BOT):
        s = str(entry)
        if s not in sys.path:
            sys.path.insert(0, s)


@dataclass
class TrainSample:
    """One seat-local training example for the Part 14 trainer buffer."""

    item_id: str
    sample_seat: int
    tensor: Array  # (49, 21, 21)
    legal_mask: Array  # (N_ACTIONS,) bool
    targets: SeatTargets
    source_label: str
    outcome: str | None


# Back-compat alias while callers migrate off the old pilot name.
PilotSample = TrainSample


def _terminal_metrics(traj: Trajectory) -> dict[str, Any]:
    """Land/army/castle totals and winner at the recorded terminal."""
    last = traj.frames[-1]
    winner = str(traj.end.get("winner", "draw"))
    terminal_turn = int(traj.end.get("turns") or last.turn)
    castles = (0, 0)
    for _turn, state, info in replay_states(traj):
        if bool(getattr(info, "is_done", False)) or _turn >= terminal_turn:
            ownership = np.asarray(state.ownership, dtype=bool)
            castle_plane = np.asarray(state.castles, dtype=bool)
            castles = (
                int((castle_plane & ownership[0]).sum()),
                int((castle_plane & ownership[1]).sum()),
            )
            land = (int(info.land[0]), int(info.land[1]))
            army = (int(info.army[0]), int(info.army[1]))
            return {
                "winner": winner,
                "terminal_turn": int(_turn),
                "final_land": land,
                "final_army": army,
                "final_castles": castles,
            }
    return {
        "winner": winner,
        "terminal_turn": terminal_turn,
        "final_land": last.land,
        "final_army": last.army,
        "final_castles": castles,
    }


def _next_action_policy(traj: Trajectory, *, seat: int, prefix_len: int) -> SparsePolicy:
    """Behavior-clone the recorded action taken from the prefix state."""
    _ensure_bot_path()
    from action import PASS_INDEX, encode_action

    by_turn = {f.turn: f for f in traj.frames}
    frame = by_turn.get(int(prefix_len) + 1)
    if frame is None:
        return SparsePolicy(indices=(PASS_INDEX,), probs=(1.0,))
    action = frame.action_a if int(seat) == 0 else frame.action_b
    idx = int(encode_action(tuple(int(x) for x in action)))
    return SparsePolicy(indices=(idx,), probs=(1.0,))


def build_train_sample_from_recon(
    item: CurriculumItem,
    recon: PrefixReconstruction,
    *,
    traj: Trajectory,
    terminal_cache: dict[str, dict[str, Any]] | None = None,
) -> TrainSample:
    """Assemble a seat sample from an already-reconstructed prefix."""
    if item.sample_seat is None:
        raise ValueError(f"{item.item_id}: sample_seat is required")
    seat = int(item.sample_seat)
    if seat not in (0, 1):
        raise ValueError(f"{item.item_id}: invalid sample_seat {seat}")

    _ensure_bot_path()
    from action import legal_mask
    from particle_summary import summarize_belief
    from tensor import build_tensor

    seat_rec = recon.seats[seat]
    summary = summarize_belief(seat_rec.belief) if seat_rec.belief.n > 0 else None
    prev = seat_rec.action_history[-1] if seat_rec.action_history else None
    tensor = build_tensor(
        seat_rec.observation,
        seat_rec.memory,
        belief=summary,
        previous_action=prev,
    )
    mask = np.asarray(legal_mask(seat_rec.observation, seat_rec.memory), dtype=bool)

    cache = terminal_cache if terminal_cache is not None else {}
    key = str(item.game_id or item.item_id)
    if key not in cache:
        cache[key] = _terminal_metrics(traj)
    terminal = cache[key]

    engine_state = recon.engine_state
    targets = build_seat_targets(
        winner=str(terminal["winner"]),
        seat=seat,
        policy=_next_action_policy(traj, seat=seat, prefix_len=item.prefix_len),
        ownership=np.asarray(engine_state.ownership, dtype=bool),
        armies=np.asarray(engine_state.armies),
        generals=np.asarray(engine_state.generals, dtype=bool),
        castles=np.asarray(engine_state.castles, dtype=bool),
        final_land=terminal["final_land"],
        final_army=terminal["final_army"],
        final_castles=terminal["final_castles"],
        current_turn=int(recon.turn),
        terminal_turn=int(terminal["terminal_turn"]),
    )
    return TrainSample(
        item_id=item.item_id,
        sample_seat=seat,
        tensor=np.asarray(tensor, dtype=np.float32),
        legal_mask=mask,
        targets=targets,
        source_label=item.source_label,
        outcome=item.outcome,
    )


def build_full_start_train_sample(
    item: CurriculumItem,
    *,
    seat: int,
    n_particles: int = 4,
    truncation_turns: int = 1200,
) -> TrainSample:
    """Assemble a class-5 sample from a fresh competition board (no trajectory).

    Policy is pass (no recorded action). WDL is draw until self-play supplies
    outcomes. Hidden-state heads use engine truth at turn 0.
    """
    if int(item.class_id) != 5:
        raise ValueError(f"{item.item_id}: build_full_start_train_sample needs class 5")
    if int(item.prefix_len) != 0:
        raise ValueError(f"{item.item_id}: class 5 requires prefix_len == 0")
    seat = int(seat)
    if seat not in (0, 1):
        raise ValueError(f"{item.item_id}: invalid seat {seat}")

    recon = reconstruct_prefix(
        item,
        trajectories_root=REPO,
        traj=None,
        n_particles=n_particles,
        update_belief_flag=True,
        engine=item.engine_version,
    )
    _ensure_bot_path()
    from action import PASS_INDEX, legal_mask
    from particle_summary import summarize_belief
    from tensor import build_tensor

    seat_rec = recon.seats[seat]
    summary = summarize_belief(seat_rec.belief) if seat_rec.belief.n > 0 else None
    tensor = build_tensor(
        seat_rec.observation,
        seat_rec.memory,
        belief=summary,
        previous_action=None,
    )
    mask = np.asarray(legal_mask(seat_rec.observation, seat_rec.memory), dtype=bool)
    engine_state = recon.engine_state
    ownership = np.asarray(engine_state.ownership, dtype=bool)
    armies = np.asarray(engine_state.armies)
    castles = np.asarray(engine_state.castles, dtype=bool)
    land = (
        int(ownership[0].sum()),
        int(ownership[1].sum()),
    )
    army = (
        int(np.asarray(armies)[ownership[0]].sum()) if ownership[0].any() else 0,
        int(np.asarray(armies)[ownership[1]].sum()) if ownership[1].any() else 0,
    )
    castle_counts = (
        int((castles & ownership[0]).sum()),
        int((castles & ownership[1]).sum()),
    )
    targets = build_seat_targets(
        winner="draw",
        seat=seat,
        policy=SparsePolicy(indices=(PASS_INDEX,), probs=(1.0,)),
        ownership=ownership,
        armies=armies,
        generals=np.asarray(engine_state.generals, dtype=bool),
        castles=castles,
        final_land=land,
        final_army=army,
        final_castles=castle_counts,
        current_turn=int(recon.turn),
        terminal_turn=int(truncation_turns),
    )
    return TrainSample(
        item_id=item.item_id,
        sample_seat=seat,
        tensor=np.asarray(tensor, dtype=np.float32),
        legal_mask=mask,
        targets=targets,
        source_label=item.source_label,
        outcome="draw",
    )


def build_train_sample(
    item: CurriculumItem,
    *,
    trajectories_root: Path | None = None,
    traj: Trajectory | None = None,
    n_particles: int = 4,
    terminal_cache: dict[str, dict[str, Any]] | None = None,
) -> TrainSample:
    """Reconstruct the prefix and assemble seat targets for ``sample_seat``."""
    if int(item.class_id) == 5:
        seat = 0 if item.sample_seat is None else int(item.sample_seat)
        return build_full_start_train_sample(
            item, seat=seat, n_particles=n_particles
        )

    source = traj
    if source is None and item.trajectory_relpath:
        path = Path(item.trajectory_relpath)
        if not path.is_file() and trajectories_root is not None:
            path = Path(trajectories_root) / item.trajectory_relpath
        if not path.is_file() and trajectories_root is None:
            path = REPO / item.trajectory_relpath
        source = read_trajectory(path)

    if source is None:
        raise ValueError(f"{item.item_id}: missing trajectory for train sample")

    recon = reconstruct_prefix(
        item,
        trajectories_root=trajectories_root or REPO,
        traj=source,
        n_particles=n_particles,
        update_belief_flag=True,
        engine=item.engine_version,
    )
    return build_train_sample_from_recon(
        item,
        recon,
        traj=source,
        terminal_cache=terminal_cache,
    )


# Old name used by earlier pilot docs/call sites.
build_pilot_sample = build_train_sample
