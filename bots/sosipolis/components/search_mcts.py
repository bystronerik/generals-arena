"""SearchMCTS: explore to locate the enemy general under a wall-clock budget."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from components.army import (
    Action,
    gather_toward,
    is_wall,
    move_action,
    neighbors,
    pass_action,
)
from components.clock import Deadline
from params import Params


@dataclass
class SearchStats:
    iterations: int = 0
    root_moves: int = 0


@dataclass
class _Node:
    action: Action | None
    prior: float
    visits: int = 0
    value: float = 0.0
    children: list["_Node"] = field(default_factory=list)

    def uct(self, parent_visits: int, c: float) -> float:
        if self.visits == 0:
            return float("inf")
        return self.value / self.visits + c * math.sqrt(
            math.log(parent_visits + 1) / self.visits
        )


class SearchMCTS:
    def __init__(self, params: Params):
        self.params = params
        self.stats = SearchStats()

    def search(self, obs, state, deadline: Deadline) -> Action:
        self.stats = SearchStats()
        root_moves = self._root_moves(obs, state)
        self.stats.root_moves = len(root_moves)
        if not root_moves:
            return pass_action()

        # Heuristic best is always child 0.
        root_moves.sort(key=lambda item: -item[1])
        root = _Node(action=None, prior=1.0)
        for action, prior in root_moves[: self.params.MCTS_MAX_ROOT]:
            root.children.append(_Node(action=action, prior=max(prior, 0.01)))

        c = self.params.MCTS_C
        depth = self.params.MCTS_ROLLOUT_DEPTH
        while not deadline.expired():
            node = self._select(root, c)
            if node.action is None:
                break
            value = self._rollout(obs, state, node.action, depth, deadline)
            node.visits += 1
            node.value += value
            root.visits += 1
            self.stats.iterations += 1
            if deadline.expired():
                break

        best = max(root.children, key=lambda n: (n.visits, n.value, n.prior))
        return best.action if best.action is not None else pass_action()

    def _select(self, root: _Node, c: float) -> _Node:
        if not root.children:
            return root
        return max(root.children, key=lambda n: n.uct(root.visits, c) + n.prior)

    def _score_expand(self, obs, state, r: int, c: int, nr: int, nc: int) -> float:
        army = obs.army_grid[r][c]
        dest_army = obs.army_grid[nr][nc]
        owner = obs.owner_grid[nr][nc]
        # Cannot capture.
        if owner != 1 and army - 1 <= dest_army:
            return -1.0

        score = float(army)
        # Land capture is mandatory tempo (Kubic contact ~82 needs map cover).
        if owner == 0:
            score += 50.0
            if not state.memory.ever_seen[nr][nc]:
                # Section prior biases *which* fog frontier, not whether to take land.
                score += 25.0 * (0.4 + state.sections.score_cell(nr, nc))
            else:
                score += 12.0
        elif owner == 2:
            score += 45.0 + min(dest_army, 20)
            score += 20.0 * state.sections.score_cell(nr, nc)
        else:
            # Reinforcing own cell — weak.
            score += 2.0

        score += 15.0 * state.sections.score_cell(nr, nc)

        # Penalize thin corridor probes into low-prior fog.
        if owner == 0 and not state.memory.ever_seen[nr][nc]:
            owned_adj = sum(
                1
                for ar, ac in neighbors(obs.H, obs.W, nr, nc)
                if obs.owner_grid[ar][ac] == 1
            )
            prior = state.sections.score_cell(nr, nc)
            if owned_adj <= self.params.CORRIDOR_WIDTH_MAX and prior < 0.12:
                score *= 0.5
        return score

    def _root_moves(self, obs, state) -> list[tuple[Action, float]]:
        H, W = obs.H, obs.W
        dead = state.dead_pockets
        out: list[tuple[Action, float]] = []
        for r in range(H):
            for c in range(W):
                if obs.owner_grid[r][c] != 1 or obs.army_grid[r][c] <= 1:
                    continue
                for nr, nc in neighbors(H, W, r, c):
                    if is_wall(obs.type_grid, nr, nc):
                        continue
                    if (nr, nc) in dead and obs.owner_grid[nr][nc] != 2:
                        if not self._pocket_has_candidates(state, nr, nc):
                            continue
                    prior = self._score_expand(obs, state, r, c, nr, nc)
                    if prior < 0:
                        continue
                    out.append((move_action(r, c, nr, nc, 0), prior))
        if not out:
            rally = self._best_frontier(obs, state)
            if rally is not None:
                blocked = lambda rr, cc: is_wall(obs.type_grid, rr, cc) or (
                    (rr, cc) in dead and obs.owner_grid[rr][cc] != 2
                )
                act = gather_toward(obs, rally, blocked)
                if act is not None:
                    out.append((act, 1.0))
        return out

    def _pocket_has_candidates(self, state, r: int, c: int) -> bool:
        if len(state.memory.candidates) <= 3:
            return (r, c) in state.memory.candidates or any(
                cell in state.dead_pockets for cell in state.memory.candidates
            )
        return False

    def _best_frontier(self, obs, state) -> tuple[int, int] | None:
        best = None
        best_s = -1.0
        H, W = obs.H, obs.W
        for r in range(H):
            for c in range(W):
                if obs.owner_grid[r][c] != 1:
                    continue
                for nr, nc in neighbors(H, W, r, c):
                    if is_wall(obs.type_grid, nr, nc):
                        continue
                    if obs.owner_grid[nr][nc] == 1:
                        continue
                    if (nr, nc) in state.dead_pockets and obs.owner_grid[nr][nc] != 2:
                        continue
                    s = state.sections.score_cell(nr, nc)
                    if not state.memory.ever_seen[nr][nc]:
                        s += 0.5
                    if s > best_s:
                        best_s = s
                        best = (nr, nc)
        return best

    def _rollout(
        self, obs, state, action: Action, depth: int, deadline: Deadline
    ) -> float:
        # Lightweight value: score the immediate action features; no full sim.
        if deadline.expired():
            return 0.0
        _, r, c, d, _ = action
        if action[0] == 1:
            return 0.0
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        dr, dc = dirs[d]
        nr, nc = r + dr, c + dc
        if not (0 <= nr < obs.H and 0 <= nc < obs.W):
            return -1.0
        value = self._score_expand(obs, state, r, c, nr, nc) / 100.0
        # Candidate proximity bonus.
        if state.memory.candidates:
            mind = min(
                abs(nr - cr) + abs(nc - cc) for cr, cc in state.memory.candidates
            )
            value += 2.0 / (1.0 + mind)
        # Noise keeps UCT exploring when scores tie.
        value += random.random() * 0.05
        # Depth unused in this cheap model but kept for API stability.
        _ = depth
        return value
