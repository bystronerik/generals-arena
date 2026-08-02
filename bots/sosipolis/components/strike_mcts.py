"""StrikeMCTS: cheapest path to a known enemy general under a wall-clock budget."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from components.army import (
    Action,
    bfs_dist,
    gather_toward,
    is_wall,
    largest_owned_stack,
    move_action,
    neighbors,
    pass_action,
    step_toward,
)
from components.clock import Deadline
from params import Params


@dataclass
class StrikeStats:
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


class StrikeMCTS:
    def __init__(self, params: Params):
        self.params = params
        self.stats = StrikeStats()

    def search(self, obs, state, deadline: Deadline) -> Action:
        self.stats = StrikeStats()
        goal = state.memory.enemy_general
        if goal is None:
            return pass_action()

        # Immediate kill.
        kill = self._kill_shot(obs, goal)
        if kill is not None:
            return kill

        root_moves = self._root_moves(obs, state, goal)
        self.stats.root_moves = len(root_moves)
        if not root_moves:
            return pass_action()

        root_moves.sort(key=lambda item: -item[1])
        root = _Node(action=None, prior=1.0)
        for action, prior in root_moves[: self.params.MCTS_MAX_ROOT]:
            root.children.append(_Node(action=action, prior=max(prior, 0.01)))

        c = self.params.MCTS_C
        while not deadline.expired():
            node = max(root.children, key=lambda n: n.uct(root.visits, c) + n.prior)
            if node.action is None:
                break
            value = self._evaluate(obs, state, goal, node.action, deadline)
            node.visits += 1
            node.value += value
            root.visits += 1
            self.stats.iterations += 1
            if deadline.expired():
                break

        best = max(root.children, key=lambda n: (n.visits, n.value, n.prior))
        return best.action if best.action is not None else pass_action()

    def _kill_shot(self, obs, goal) -> Action | None:
        gr, gc = goal
        need = self._capture_need(obs, gr, gc)
        for r, c in neighbors(obs.H, obs.W, gr, gc):
            if obs.owner_grid[r][c] != 1:
                continue
            if obs.army_grid[r][c] - 1 >= need:
                return move_action(r, c, gr, gc, 0)
        return None

    def _capture_need(self, obs, gr: int, gc: int) -> int:
        if obs.turn >= self.params.DEATHTOUCH_TURN:
            return 2
        defender = 0
        if obs.owner_grid[gr][gc] == 2:
            defender = obs.army_grid[gr][gc]
        return defender + self.params.FINISH_MARGIN

    def _blocked(self, obs):
        return lambda r, c: is_wall(obs.type_grid, r, c)

    def _root_moves(self, obs, state, goal) -> list[tuple[Action, float]]:
        blocked = self._blocked(obs)
        gr, gc = goal
        need = self._capture_need(obs, gr, gc)
        dist = bfs_dist(obs.H, obs.W, [goal], blocked)
        out: list[tuple[Action, float]] = []

        # Advance closest strong stacks toward the general.
        stacks: list[tuple[int, int, int, int]] = []
        for r in range(obs.H):
            for c in range(obs.W):
                if obs.owner_grid[r][c] != 1 or obs.army_grid[r][c] <= 1:
                    continue
                if (r, c) not in dist:
                    continue
                stacks.append((obs.army_grid[r][c], -dist[(r, c)], r, c))
        stacks.sort(reverse=True)

        for army, neg_d, r, c in stacks[:8]:
            act = step_toward(obs, (r, c), goal, blocked)
            if act is None:
                continue
            d = -neg_d
            # Toward bias from Kubic toward-move share.
            prior = army * self.params.STRIKE_TOWARD_BIAS / (1.0 + d)
            if army - 1 >= need and d <= 1:
                prior += 100.0
            out.append((act, prior))

        # Gather toward the largest stack nearest the goal (staging).
        if stacks:
            _, _, rr, cc = stacks[0]
            gact = gather_toward(obs, (rr, cc), blocked)
            if gact is not None:
                out.append((gact, 15.0 * self.params.GATHER_WAVE_HINT))

        # Also gather toward a cell adjacent to the general if we own one.
        for r, c in neighbors(obs.H, obs.W, gr, gc):
            if obs.owner_grid[r][c] == 1:
                gact = gather_toward(obs, (r, c), blocked)
                if gact is not None:
                    out.append((gact, 20.0))
                break

        if not out:
            src = largest_owned_stack(obs)
            if src is not None:
                act = step_toward(obs, src, goal, blocked)
                if act is not None:
                    out.append((act, 1.0))
        return out

    def _evaluate(
        self, obs, state, goal, action: Action, deadline: Deadline
    ) -> float:
        if deadline.expired() or action[0] == 1:
            return 0.0
        _, r, c, d, _ = action
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        dr, dc = dirs[d]
        nr, nc = r + dr, c + dc
        gr, gc = goal
        before = abs(r - gr) + abs(c - gc)
        after = abs(nr - gr) + abs(nc - gc)
        value = 0.0
        if after < before:
            value += self.params.STRIKE_TOWARD_BIAS
        elif after > before:
            value -= 0.5
        # Prefer moving larger stacks.
        value += min(obs.army_grid[r][c], 50) / 50.0
        if (nr, nc) == goal:
            value += 5.0
        value += random.random() * 0.02
        return value
