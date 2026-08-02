#!/usr/bin/env python3
"""
Shared infrastructure for the Kubic behavior-spec analysis (fable-kubic-*).

Replays store state, not actions. This module reconstructs both players'
per-tick actions by exact forward simulation of the competition engine
(competition-module/generals/core/game.py + modifiers build_castles.py /
deathtouch.py, reimplemented here in numpy): for each tick it enumerates a
small candidate set of actions per player, simulates every pair, and accepts
the pair that reproduces the next frame exactly. A tick no pair reproduces is
recorded as unresolved and the simulator resyncs to the observed frame.

Engine semantics mirrored exactly:
- move sends floor(a/2) (split=1) or a-1 (split=0), needs >0 moved, src owned,
  dst passable; invalid move = silent pass.
- both moves same tick, one resolves fully first: chasing > reinforcing >
  smaller army; ties fall through, final tie -> player 0 first. Army for the
  second move is re-read after the first resolved (chases strip sources).
- combat: attacker takes the cell iff moved > defenders, cell army becomes
  |defenders - moved|; defender keeps on tie.
- builds resolve before either move; cost 35 + sum over own structures of
  max(0, 14 - 2*manhattan); remainder stays; cell becomes a castle.
- after both moves, time increments to T; if T%50==0 every owned cell +1; if
  T%2==0 every general/castle +1. On a general capture (or a deathtouch touch
  at time>=800) the loser's cells transfer to the winner instead of growth.

The action cache lives under competition-replays/Kubic/_derived_actions/
(gitignored with the replays). Fit/holdout split comes from
docs/research/measurements/fable-kubic-split.json (see fable_kubic_split.py).
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arena.instrument.replay.loader import Replay, open_replay

PLAYER = "Kubic"
SPLIT_MANIFEST = REPO_ROOT / "docs/research/measurements/fable-kubic-split.json"
ACTIONS_CACHE = REPO_ROOT / "competition-replays" / PLAYER / "_derived_actions"

DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))  # up, down, left, right
BASE_COST = 35
PROXIMITY_PENALTY = 14
PROXIMITY_DECAY = 2
DEATHTOUCH_TURN = 800


# ---------------------------------------------------------------- split I/O

def load_split() -> dict:
    return json.loads(SPLIT_MANIFEST.read_text())


def split_ids(which: str) -> list[str]:
    """Match ids for `which` in {'fit', 'holdout', 'all'}, id-sorted."""
    rows = load_split()["replays"]
    if which != "all":
        rows = [r for r in rows if r["set"] == which]
    return [r["match_id"] for r in rows]


def split_rows(which: str) -> list[dict]:
    rows = load_split()["replays"]
    if which != "all":
        rows = [r for r in rows if r["set"] == which]
    return rows


# ------------------------------------------------------------- build prices

def build_cost(cell: tuple[int, int], structures: set[tuple[int, int]]) -> int:
    r, c = cell
    cost = BASE_COST
    for sr, sc in structures:
        cost += max(0, PROXIMITY_PENALTY - PROXIMITY_DECAY * (abs(r - sr) + abs(c - sc)))
    return cost


# ------------------------------------------------------------ action record

@dataclass(frozen=True)
class Action:
    kind: str  # 'pass' | 'move' | 'build'
    src: tuple[int, int] | None = None
    dst: tuple[int, int] | None = None
    split: int = 0
    cell: tuple[int, int] | None = None  # build target
    cost: int = 0

    def key(self):
        return (self.kind, self.src, self.dst, self.split, self.cell)


PASS = Action("pass")


class _Sim:
    """Mutable board state for candidate simulation."""

    __slots__ = ("armies", "owners", "castles", "generals", "passable", "winner")

    def __init__(self, armies, owners, castles, generals, passable):
        self.armies = armies
        self.owners = owners
        self.castles = castles
        self.generals = generals  # tuple of two cells
        self.passable = passable
        self.winner = -1


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


def _execute(sim: _Sim, p: int, act: Action, old_time: int) -> None:
    if act.kind == "build":
        # validity was pre-checked by candidate generation
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
        sim.winner = p  # touch executes: wins regardless of combat outcome


def _simulate(base: _Sim, a0: Action, a1: Action, new_time: int) -> _Sim:
    sim = _Sim(
        base.armies.copy(), base.owners.copy(), base.castles.copy(),
        base.generals, base.passable,
    )
    for p, act in ((0, a0), (1, a1)):
        if act.kind == "build":
            _execute(sim, p, act, new_time - 1)
    first = _first_player(a0, a1, sim)
    order = (first, 1 - first)
    acts = (a0, a1)
    for p in order:
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


def _general_mask(sim: _Sim, p: int) -> np.ndarray:
    mask = np.zeros_like(sim.castles)
    mask[sim.generals[p]] = True
    return mask


def _predict_pass(base: _Sim, new_time: int) -> _Sim:
    return _simulate(base, PASS, PASS, new_time)


def _candidates(
    base: _Sim, p: int, diff_cells: list[tuple[int, int]], armies_next: np.ndarray
) -> list[Action]:
    out = [PASS]
    diff = set(diff_cells)
    srcs = set()
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
        for d, (dr, dc) in enumerate(DIRECTIONS):
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
    # builds: an owned, non-structure diff cell whose army dropped and can afford
    structures = _own_structures(base, p)
    for cell in sorted(diff):
        if base.owners[cell] != p or base.castles[cell] or cell in (base.generals[p],):
            continue
        cost = build_cost(cell, structures)
        if base.armies[cell] >= cost:
            out.append(Action("build", cell=cell, cost=cost))
    return out


def infer_game(replay: Replay, max_pairs: int = 4000) -> dict:
    """Reconstruct both players' actions for every tick of one replay."""
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

    ticks_out = []
    unresolved = []
    ambiguous = []
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

        final = bool(
            replay.winner >= 0
            and t == len(replay.ticks) - 1
        )
        if final:
            # ownership transfer flips every loser cell — candidates from the
            # ownership diff would explode. Army diffs plus the generals pin
            # down both moves.
            army_diff = [(int(r), int(c)) for r, c in np.argwhere(pred.armies != actual_armies)]
            cand_cells = army_diff + [tuple(g) for g in replay.generals]
        else:
            cand_cells = diff_cells
        cand0 = _candidates(sim, 0, cand_cells, actual_armies)
        cand1 = _candidates(sim, 1, cand_cells, actual_armies)

        matches = []
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
            # recovery: an unrecorded castle keeps producing +1 on even ticks.
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
                            {"t": t, "p0": {"kind": "pass"}, "p1": {"kind": "pass"},
                             "castle_recovered": [list(c) for c in extra]}
                        )
                        sim = pred2
                        recovered = True
            if not recovered:
                unresolved.append(t)
                ticks_out.append({"t": t, "p0": {"kind": "unresolved"}, "p1": {"kind": "unresolved"}})
                sim.armies = actual_armies.copy()
                sim.owners = actual_owners.copy()
            continue

        # canonical pick: prefer non-pass explanations sorted deterministically
        keys = {(m[0].key(), m[1].key()) for m in matches}
        a0, a1, out = sorted(
            matches, key=lambda m: (m[0].key() == PASS.key(), m[1].key() == PASS.key(), str(m[0].key()), str(m[1].key()))
        )[0]
        rec = {"t": t, "p0": _serialize(sim, 0, a0), "p1": _serialize(sim, 1, a1)}
        if len(keys) > 1:
            rec["ambiguous"] = len(keys)
            ambiguous.append(t)
        ticks_out.append(rec)
        sim = out

    us = replay.seat_of(PLAYER)
    return {
        "match_id": replay.match_id,
        "kubic_seat": us,
        "players": list(replay.players),
        "winner": replay.winner,
        "outcome": replay.outcome,
        "rows": rows,
        "cols": cols,
        "generals": [list(g) for g in replay.generals],
        "total_ticks": replay.total_ticks,
        "unresolved": unresolved,
        "ambiguous": ambiguous,
        "ticks": ticks_out,
    }


def _dedupe(actions: list[Action]) -> list[Action]:
    seen = set()
    out = []
    for a in actions:
        if a.key() not in seen:
            seen.add(a.key())
            out.append(a)
    return out


def _serialize(sim: _Sim, p: int, act: Action) -> dict:
    if act.kind == "pass":
        return {"kind": "pass"}
    if act.kind == "build":
        return {"kind": "build", "cell": [int(x) for x in act.cell], "cost": int(act.cost)}
    src, dst = act.src, act.dst
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


# ------------------------------------------------------------------- cache

def actions_path(match_id: str) -> Path:
    return ACTIONS_CACHE / f"{match_id}.json"


def load_actions(match_id: str) -> dict:
    """Cached inferred actions for one game; infers and caches on miss."""
    path = actions_path(match_id)
    if path.is_file():
        return json.loads(path.read_text())
    replay = open_replay(PLAYER, match_id)
    result = infer_game(replay)
    ACTIONS_CACHE.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{match_id}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(result))
    tmp.rename(path)  # atomic; concurrent writers converge on identical content
    return result


def kubic_actions(game: dict) -> list[dict]:
    """Per-tick action dicts for Kubic's seat, tagged with the tick index."""
    key = f"p{game['kubic_seat']}"
    out = []
    for tick in game["ticks"]:
        act = dict(tick[key])
        act["t"] = tick["t"]
        if "ambiguous" in tick:
            act["ambiguous"] = True
        out.append(act)
    return out


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    ids = split_ids(which)
    total_unresolved = 0
    total_ticks = 0
    for i, match_id in enumerate(ids):
        game = load_actions(match_id)
        total_unresolved += len(game["unresolved"])
        total_ticks += len(game["ticks"])
        if (i + 1) % 25 == 0 or i == len(ids) - 1:
            print(
                f"{i + 1}/{len(ids)} cached; unresolved {total_unresolved}/{total_ticks}"
                f" ({100 * total_unresolved / max(1, total_ticks):.2f}%)",
                flush=True,
            )
