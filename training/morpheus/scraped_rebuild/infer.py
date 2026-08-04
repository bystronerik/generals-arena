"""Exact joint-action inference from scraped replay state ticks.

Replays store states, not actions. For each tick this module enumerates a
small candidate set of actions per seat, simulates every pair with a numpy
mirror of the competition transition, and accepts the pair that reproduces
the next frame bit-for-bit.

Lifted from ``scripts/fable-analyze/fable_kubic_common.py`` and made
player-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from arena.instrument.replay.loader import Replay

DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))
BASE_COST = 35
PROXIMITY_PENALTY = 14
PROXIMITY_DECAY = 2
DEATHTOUCH_TURN = 800


@dataclass(frozen=True)
class Action:
    kind: str  # pass | move | build
    src: tuple[int, int] | None = None
    dst: tuple[int, int] | None = None
    split: int = 0
    cell: tuple[int, int] | None = None
    cost: int = 0

    def key(self):
        return (self.kind, self.src, self.dst, self.split, self.cell)


PASS = Action("pass")


@dataclass
class InferenceResult:
    """Outcome of reconstructing one scraped replay."""

    match_id: str
    players: tuple[str, str]
    winner: int
    outcome: str
    rows: int
    cols: int
    seed: int | None
    total_ticks: int
    unresolved: list[int] = field(default_factory=list)
    ambiguous: list[int] = field(default_factory=list)
    ticks: list[dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.unresolved and bool(self.ticks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "match_id": self.match_id,
            "players": list(self.players),
            "winner": self.winner,
            "outcome": self.outcome,
            "rows": self.rows,
            "cols": self.cols,
            "seed": self.seed,
            "total_ticks": self.total_ticks,
            "unresolved": list(self.unresolved),
            "ambiguous": list(self.ambiguous),
            "ticks": list(self.ticks),
        }


class _Sim:
    __slots__ = ("armies", "owners", "castles", "generals", "passable", "winner")

    def __init__(self, armies, owners, castles, generals, passable):
        self.armies = armies
        self.owners = owners
        self.castles = castles
        self.generals = generals
        self.passable = passable
        self.winner = -1


def build_cost(cell: tuple[int, int], structures: set[tuple[int, int]]) -> int:
    r, c = cell
    cost = BASE_COST
    for sr, sc in structures:
        cost += max(0, PROXIMITY_PENALTY - PROXIMITY_DECAY * (abs(r - sr) + abs(c - sc)))
    return cost


def _own_structures(sim: _Sim, p: int) -> set[tuple[int, int]]:
    cells = {(int(r), int(c)) for r, c in np.argwhere(sim.castles & (sim.owners == p))}
    gr, gc = sim.generals[p]
    if sim.owners[gr, gc] == p:
        cells.add((gr, gc))
    return cells


def _moved_amount(army: int, split: int) -> int:
    moved = army // 2 if split == 1 else army - 1
    return max(0, min(moved, army - 1))


def _first_player(a0: Action, a1: Action, sim: _Sim) -> int:
    if a0.kind != "move" or a1.kind != "move":
        return 0 if a0.kind == "move" else 1
    chase0 = a0.dst == a1.src
    chase1 = a1.dst == a0.src
    if chase1 and not chase0:
        return 1
    if chase0 != chase1:
        return 0
    rein0 = sim.owners[a0.dst] == 0
    rein1 = sim.owners[a1.dst] == 1
    if rein1 and not rein0:
        return 1
    if rein0 != rein1:
        return 0
    if sim.armies[a1.src] < sim.armies[a0.src]:
        return 1
    return 0


def _general_mask(sim: _Sim, p: int) -> np.ndarray:
    mask = np.zeros_like(sim.castles)
    mask[sim.generals[p]] = True
    return mask


def _execute(sim: _Sim, p: int, act: Action, old_time: int) -> None:
    if act.kind == "build":
        sim.armies[act.cell] -= act.cost
        sim.castles[act.cell] = True
        return
    if act.kind != "move":
        return
    sr, sc = act.src
    dr, dc = act.dst
    if sim.owners[sr, sc] != p:
        return
    moved = _moved_amount(int(sim.armies[sr, sc]), act.split)
    if moved <= 0 or not sim.passable[dr, dc]:
        return
    enemy_general = act.dst == sim.generals[1 - p]
    if sim.owners[dr, dc] == p:
        sim.armies[dr, dc] += moved
        sim.armies[sr, sc] -= moved
        return
    target = int(sim.armies[dr, dc])
    sim.armies[sr, sc] -= moved
    sim.armies[dr, dc] = abs(target - moved)
    if moved > target:
        sim.owners[dr, dc] = p
        if enemy_general:
            sim.winner = p
    if enemy_general and old_time >= DEATHTOUCH_TURN:
        sim.winner = p


def _simulate(base: _Sim, a0: Action, a1: Action, new_time: int) -> _Sim:
    sim = _Sim(
        base.armies.copy(),
        base.owners.copy(),
        base.castles.copy(),
        base.generals,
        base.passable,
    )
    for p, act in ((0, a0), (1, a1)):
        if act.kind == "build":
            _execute(sim, p, act, new_time - 1)
    first = _first_player(a0, a1, sim)
    acts = (a0, a1)
    for p in (first, 1 - first):
        if acts[p].kind == "move":
            _execute(sim, p, acts[p], new_time - 1)
    if sim.winner >= 0:
        loser = 1 - sim.winner
        sim.owners[sim.owners == loser] = sim.winner
    else:
        if new_time % 50 == 0:
            sim.armies[sim.owners >= 0] += 1
        if new_time % 2 == 0:
            for p in (0, 1):
                mask = (sim.owners == p) & (sim.castles | _general_mask(sim, p))
                sim.armies[mask] += 1
    return sim


def _predict_pass(base: _Sim, new_time: int) -> _Sim:
    return _simulate(base, PASS, PASS, new_time)


def _candidates(
    base: _Sim, p: int, diff_cells: list[tuple[int, int]], armies_next: np.ndarray
) -> list[Action]:
    del armies_next  # reserved for tighter filters
    out = [PASS]
    diff = set(diff_cells)
    srcs: set[tuple[int, int]] = set()
    for r, c in diff:
        if base.owners[r, c] == p and base.armies[r, c] >= 2:
            srcs.add((r, c))
        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < base.owners.shape[0] and 0 <= nc < base.owners.shape[1]:
                if base.owners[nr, nc] == p and base.armies[nr, nc] >= 2:
                    srcs.add((nr, nc))
    for src in sorted(srcs):
        army = int(base.armies[src])
        for dr, dc in DIRECTIONS:
            dst = (src[0] + dr, src[1] + dc)
            if not (0 <= dst[0] < base.owners.shape[0] and 0 <= dst[1] < base.owners.shape[1]):
                continue
            if not base.passable[dst]:
                continue
            if dst not in diff and src not in diff:
                continue
            out.append(Action("move", src=src, dst=dst, split=0))
            if army >= 3 and army // 2 != army - 1:
                out.append(Action("move", src=src, dst=dst, split=1))
    structures = _own_structures(base, p)
    for cell in sorted(diff):
        if base.owners[cell] != p or base.castles[cell] or cell in (base.generals[p],):
            continue
        cost = build_cost(cell, structures)
        if base.armies[cell] >= cost:
            out.append(Action("build", cell=cell, cost=cost))
    return out


def _serialize(sim: _Sim, p: int, act: Action) -> dict[str, Any]:
    if act.kind == "pass":
        return {"kind": "pass"}
    if act.kind == "build":
        return {
            "kind": "build",
            "cell": [int(x) for x in act.cell],  # type: ignore[arg-type]
            "cost": int(act.cost),
        }
    src, dst = act.src, act.dst
    assert src is not None and dst is not None
    army = int(sim.armies[src])
    moved = int(_moved_amount(army, act.split))
    dst_owner = int(sim.owners[dst])
    if dst_owner == p:
        target = "own"
    elif dst_owner == -1:
        target = "neutral"
    elif dst == sim.generals[1 - p]:
        target = "enemy_general"
    else:
        target = "enemy"
    return {
        "kind": "move",
        "src": [int(x) for x in src],
        "dst": [int(x) for x in dst],
        "split": int(act.split),
        "moved": moved,
        "src_army": army,
        "dst_army": int(sim.armies[dst]),
        "target": target,
        "captured": target != "own" and moved > int(sim.armies[dst]),
    }


def infer_joint_actions(replay: Replay, *, max_pairs: int = 4000) -> InferenceResult:
    """Reconstruct both seats' actions for every tick of one scraped replay."""
    rows, cols = replay.rows, replay.cols
    passable = np.ones((rows, cols), dtype=bool)
    for r, c in replay.mountains:
        passable[r, c] = False
    frame0 = replay.ticks[0]
    sim = _Sim(
        np.array(frame0.armies, dtype=np.int64),
        np.array(frame0.owners, dtype=np.int64),
        np.zeros((rows, cols), dtype=bool),
        (tuple(replay.generals[0]), tuple(replay.generals[1])),
        passable,
    )

    ticks_out: list[dict[str, Any]] = []
    unresolved: list[int] = []
    ambiguous: list[int] = []

    for t in range(1, len(replay.ticks)):
        new_time = t
        actual_armies = np.array(replay.ticks[t].armies, dtype=np.int64)
        actual_owners = np.array(replay.ticks[t].owners, dtype=np.int64)

        pred = _predict_pass(sim, new_time)
        mism = (pred.armies != actual_armies) | (pred.owners != actual_owners)
        diff_cells = [(int(r), int(c)) for r, c in np.argwhere(mism)]

        if not diff_cells:
            ticks_out.append({"t": t, "p0": {"kind": "pass"}, "p1": {"kind": "pass"}})
            sim = pred
            continue

        final = bool(replay.winner >= 0 and t == len(replay.ticks) - 1)
        if final:
            army_diff = [
                (int(r), int(c)) for r, c in np.argwhere(pred.armies != actual_armies)
            ]
            cand_cells = army_diff + [tuple(g) for g in replay.generals]
        else:
            cand_cells = diff_cells
        cand0 = _candidates(sim, 0, cand_cells, actual_armies)
        cand1 = _candidates(sim, 1, cand_cells, actual_armies)

        matches: list[tuple[Action, Action, _Sim]] = []
        tried = 0
        for a0 in cand0:
            for a1 in cand1:
                tried += 1
                if tried > max_pairs:
                    break
                out = _simulate(sim, a0, a1, new_time)
                if np.array_equal(out.armies, actual_armies) and np.array_equal(
                    out.owners, actual_owners
                ):
                    matches.append((a0, a1, out))
            if tried > max_pairs:
                break

        if not matches:
            recovered = False
            if new_time % 2 == 0:
                extra = [
                    (r, c)
                    for r, c in diff_cells
                    if actual_owners[r, c] == pred.owners[r, c]
                    and actual_armies[r, c] == pred.armies[r, c] + 1
                    and not sim.castles[r, c]
                    and (r, c) not in (sim.generals[0], sim.generals[1])
                ]
                if extra:
                    for cell in extra:
                        sim.castles[cell] = True
                    pred2 = _predict_pass(sim, new_time)
                    if np.array_equal(pred2.armies, actual_armies) and np.array_equal(
                        pred2.owners, actual_owners
                    ):
                        ticks_out.append(
                            {
                                "t": t,
                                "p0": {"kind": "pass"},
                                "p1": {"kind": "pass"},
                                "castle_recovered": [list(c) for c in extra],
                            }
                        )
                        sim = pred2
                        recovered = True
            if not recovered:
                unresolved.append(t)
                ticks_out.append(
                    {
                        "t": t,
                        "p0": {"kind": "unresolved"},
                        "p1": {"kind": "unresolved"},
                    }
                )
                sim.armies = actual_armies.copy()
                sim.owners = actual_owners.copy()
            continue

        keys = {(m[0].key(), m[1].key()) for m in matches}
        a0, a1, out = sorted(
            matches,
            key=lambda m: (
                m[0].key() == PASS.key(),
                m[1].key() == PASS.key(),
                str(m[0].key()),
                str(m[1].key()),
            ),
        )[0]
        rec: dict[str, Any] = {
            "t": t,
            "p0": _serialize(sim, 0, a0),
            "p1": _serialize(sim, 1, a1),
        }
        if len(keys) > 1:
            rec["ambiguous"] = len(keys)
            ambiguous.append(t)
        ticks_out.append(rec)
        sim = out

    seed = replay.seed
    if seed is None and replay.meta is not None:
        seed = replay.meta.seed

    return InferenceResult(
        match_id=replay.match_id,
        players=tuple(replay.players),
        winner=int(replay.winner),
        outcome=str(replay.outcome),
        rows=rows,
        cols=cols,
        seed=int(seed) if seed is not None else None,
        total_ticks=int(replay.total_ticks),
        unresolved=unresolved,
        ambiguous=ambiguous,
        ticks=ticks_out,
    )
