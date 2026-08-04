"""Executable classifiers for the five curriculum classes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np

from arena.records.trajectories import Trajectory, replay_states
from training.morpheus.corpus.coverage import (
    _enemy_general_visible,
    _ownership_contact,
)
from training.morpheus.curriculum.definitions import (
    CLASS_AFTER_SIGHT,
    CLASS_CONTACT,
    CLASS_PRE_CONTACT,
    CLASS_TACTICAL,
    TACTICAL_HORIZON,
)


@dataclass(frozen=True)
class TurnFeatures:
    """Per-turn predicates used by exclusive class assignment."""

    turn: int
    both_generals_alive: bool
    contact: bool
    sight: bool
    capture_or_defense: bool
    pre_contact_distance: int  # BFS between generals; -1 if unreachable


@dataclass(frozen=True)
class ClassifiedPrefix:
    """One candidate prefix with its exclusive class id."""

    turn: int  # prefix_len == turn (actions applied to reach this state)
    class_id: int
    pre_contact_distance: int | None
    first_contact_turn: int | None
    first_sight_turn: int | None


def both_generals_alive(state) -> bool:
    generals = np.asarray(state.generals, dtype=bool)
    ownership = np.asarray(state.ownership, dtype=bool)
    if ownership.ndim != 3 or ownership.shape[0] != 2:
        raise ValueError(f"expected ownership (2,H,W), got {ownership.shape}")
    return bool((generals & ownership[0]).any() and (generals & ownership[1]).any())


def capture_or_defense_threat(state) -> bool:
    """True when a seat has army ≥ 2 orthogonally next to the enemy general."""
    generals = np.asarray(state.generals, dtype=bool)
    ownership = np.asarray(state.ownership, dtype=bool)
    armies = np.asarray(state.armies)
    H, W = generals.shape
    for seat in (0, 1):
        enemy = 1 - seat
        enemy_gen = generals & ownership[enemy]
        if not enemy_gen.any():
            continue
        for er, ec in np.argwhere(enemy_gen):
            er, ec = int(er), int(ec)
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                r, c = er + dr, ec + dc
                if 0 <= r < H and 0 <= c < W and ownership[seat, r, c] and int(armies[r, c]) >= 2:
                    return True
    return False


def general_bfs_distance(state) -> int:
    """BFS steps between the two generals over passable cells; -1 if none."""
    positions = np.asarray(state.general_positions)
    passable = np.asarray(state.passable, dtype=bool)
    r0, c0 = int(positions[0, 0]), int(positions[0, 1])
    r1, c1 = int(positions[1, 0]), int(positions[1, 1])
    if not passable[r0, c0] or not passable[r1, c1]:
        return -1
    H, W = passable.shape
    dist = np.full((H, W), -1, dtype=np.int32)
    dist[r0, c0] = 0
    queue = [(r0, c0)]
    head = 0
    while head < len(queue):
        r, c = queue[head]
        head += 1
        if r == r1 and c == c1:
            return int(dist[r, c])
        d = int(dist[r, c])
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and passable[nr, nc] and dist[nr, nc] < 0:
                dist[nr, nc] = d + 1
                queue.append((nr, nc))
    return -1


def _sight_either_seat(state) -> bool:
    from generals.core.game import get_observation

    for player in (0, 1):
        obs = get_observation(state, player)
        if _enemy_general_visible(obs):
            return True
    return False


def iter_turn_features(traj: Trajectory) -> Iterator[TurnFeatures]:
    """Yield turn features while replaying a trajectory."""
    for turn, state, _info in replay_states(traj):
        yield TurnFeatures(
            turn=int(turn),
            both_generals_alive=both_generals_alive(state),
            contact=_ownership_contact(np.asarray(state.ownership)),
            sight=_sight_either_seat(state),
            capture_or_defense=capture_or_defense_threat(state),
            pre_contact_distance=general_bfs_distance(state),
        )


def assign_class(
    *,
    turn: int,
    both_alive: bool,
    contact_occurred: bool,
    sight_occurred: bool,
    capture_or_defense: bool,
    decisive: bool,
    terminal_turn: int,
) -> int | None:
    """Exclusive class id for a panel-trajectory prefix, or None to skip."""
    if not both_alive:
        return None
    tactical = capture_or_defense or (
        decisive
        and terminal_turn > turn
        and (terminal_turn - turn) <= TACTICAL_HORIZON
    )
    if tactical:
        return CLASS_TACTICAL
    if sight_occurred:
        return CLASS_AFTER_SIGHT
    if contact_occurred:
        return CLASS_CONTACT
    return CLASS_PRE_CONTACT


def classify_trajectory_prefixes(
    traj: Trajectory,
    *,
    decisive_only_for_early: bool = True,
) -> list[ClassifiedPrefix]:
    """
    Classify every reachable prefix on a panel trajectory.

    Classes 1–3 require a decisive game when ``decisive_only_for_early`` is
    True (training.md bootstrap rule). Class 4 uses all legal prefixes.
    """
    end = traj.end
    terminal_turn = int(end.get("turns", len(traj.frames)))
    winner = str(end.get("winner", "draw"))
    decisive = winner in ("a", "b")

    first_contact: int | None = None
    first_sight: int | None = None
    out: list[ClassifiedPrefix] = []

    for feat in iter_turn_features(traj):
        if feat.contact and first_contact is None:
            first_contact = feat.turn
        if feat.sight and first_sight is None:
            first_sight = feat.turn

        contact_occurred = first_contact is not None and feat.turn >= first_contact
        sight_occurred = first_sight is not None and feat.turn >= first_sight

        class_id = assign_class(
            turn=feat.turn,
            both_alive=feat.both_generals_alive,
            contact_occurred=contact_occurred,
            sight_occurred=sight_occurred,
            capture_or_defense=feat.capture_or_defense,
            decisive=decisive,
            terminal_turn=terminal_turn,
        )
        if class_id is None:
            continue
        if decisive_only_for_early and class_id in (
            CLASS_TACTICAL,
            CLASS_AFTER_SIGHT,
            CLASS_CONTACT,
        ) and not decisive:
            continue

        out.append(
            ClassifiedPrefix(
                turn=feat.turn,
                class_id=class_id,
                pre_contact_distance=(
                    feat.pre_contact_distance if class_id == CLASS_PRE_CONTACT else None
                ),
                first_contact_turn=first_contact,
                first_sight_turn=first_sight,
            )
        )
    return out


def sample_prefixes_for_build(
    classified: list[ClassifiedPrefix],
    *,
    max_per_class: dict[int, int] | None = None,
) -> list[ClassifiedPrefix]:
    """Bound the pilot: keep the latest / spaced prefixes per class."""
    limits = {
        CLASS_TACTICAL: 2,
        CLASS_AFTER_SIGHT: 2,
        CLASS_CONTACT: 2,
        CLASS_PRE_CONTACT: 4,
        **(max_per_class or {}),
    }
    by_class: dict[int, list[ClassifiedPrefix]] = {}
    for row in classified:
        by_class.setdefault(row.class_id, []).append(row)

    selected: list[ClassifiedPrefix] = []
    for class_id, rows in by_class.items():
        limit = int(limits.get(class_id, 2))
        if class_id == CLASS_PRE_CONTACT:
            # Prefer later (smaller remaining distance) and earlier extremes.
            rows_sorted = sorted(
                rows,
                key=lambda r: (
                    -(r.pre_contact_distance or -1),
                    r.turn,
                ),
            )
            # Take evenly spaced after sorting by distance descending.
            if len(rows_sorted) <= limit:
                chosen = rows_sorted
            else:
                idxs = [
                    int(round(i * (len(rows_sorted) - 1) / (limit - 1)))
                    for i in range(limit)
                ]
                chosen = [rows_sorted[i] for i in dict.fromkeys(idxs)]
        else:
            # Latest prefixes are closest to the class event.
            chosen = rows[-limit:]
        selected.extend(chosen)
    selected.sort(key=lambda r: (r.class_id, r.turn))
    return selected


def classify_summary(classified: list[ClassifiedPrefix]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for row in classified:
        key = str(row.class_id)
        counts[key] = counts.get(key, 0) + 1
    return {"prefix_count": len(classified), "class_counts": counts}
