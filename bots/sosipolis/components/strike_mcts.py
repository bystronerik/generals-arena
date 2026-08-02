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
from components.threat import defense_score_delta, home_threat
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

    def _land_captures(self, obs, state) -> list[tuple[Action, float]]:
        """Free / cheap land captures reserved in the strike root set."""
        out: list[tuple[Action, float]] = []
        H, W = obs.H, obs.W
        for r in range(H):
            for c in range(W):
                if obs.owner_grid[r][c] != 1 or obs.army_grid[r][c] <= 1:
                    continue
                for nr, nc in neighbors(H, W, r, c):
                    if is_wall(obs.type_grid, nr, nc):
                        continue
                    owner = obs.owner_grid[nr][nc]
                    if owner == 1:
                        continue
                    if obs.army_grid[r][c] - 1 <= obs.army_grid[nr][nc]:
                        continue
                    score = 12.0 + obs.army_grid[r][c] * 0.1
                    if owner == 0:
                        score += 8.0
                    else:
                        score += 15.0
                    score += defense_score_delta(
                        obs, state, self.params, (r, c), (nr, nc)
                    )
                    out.append((move_action(r, c, nr, nc, 0), score))
        out.sort(key=lambda item: -item[1])
        return out[: self.params.STRIKE_LAND_ROOT_SLOTS]

    def _root_moves(self, obs, state, goal) -> list[tuple[Action, float]]:
        blocked = self._blocked(obs)
        gr, gc = goal
        need = self._capture_need(obs, gr, gc)
        dist = bfs_dist(obs.H, obs.W, [goal], blocked)
        out: list[tuple[Action, float]] = []

        stacks: list[tuple[int, int, int, int]] = []
        for r in range(obs.H):
            for c in range(obs.W):
                if obs.owner_grid[r][c] != 1 or obs.army_grid[r][c] <= 1:
                    continue
                if (r, c) not in dist:
                    continue
                stacks.append((obs.army_grid[r][c], -dist[(r, c)], r, c))
        stacks.sort(reverse=True)

        path_long = False
        if stacks:
            army0, neg_d0, _, _ = stacks[0]
            d0 = -neg_d0
            if d0 * need > army0:
                path_long = True

        for army, neg_d, r, c in stacks[:8]:
            act = step_toward(obs, (r, c), goal, blocked)
            if act is None:
                continue
            d = -neg_d
            prior = army * self.params.STRIKE_TOWARD_BIAS / (1.0 + d)
            if army - 1 >= need and d <= 1:
                prior += 100.0
            # Defense delta on the step destination.
            _, sr, sc, di, _ = act
            dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            dr, dc = dirs[di]
            prior += defense_score_delta(
                obs, state, self.params, (sr, sc), (sr + dr, sc + dc)
            )
            # Prefer intercept if enemy is closer to home than we are to goal.
            threat = home_threat(obs, state, self.params)
            home = state.memory.own_general
            if (
                threat.dist is not None
                and home is not None
                and threat.dist < d
                and threat.nearest is not None
            ):
                intercept = step_toward(obs, (r, c), threat.nearest, blocked)
                if intercept is not None:
                    out.append((intercept, prior + self.params.DEFENSE_WEIGHT_STRIKE))
            out.append((act, prior))

        # Multi-wave gather when the path is longer than the stack can afford.
        gather_hint = self.params.GATHER_WAVE_HINT
        if path_long:
            gather_hint = max(gather_hint, self.params.STRIKE_GATHER_WAVES_HINT)
        if stacks:
            _, _, rr, cc = stacks[0]
            gact = gather_toward(obs, (rr, cc), blocked)
            if gact is not None:
                out.append((gact, 15.0 * gather_hint))

        for r, c in neighbors(obs.H, obs.W, gr, gc):
            if obs.owner_grid[r][c] == 1:
                gact = gather_toward(obs, (r, c), blocked)
                if gact is not None:
                    out.append((gact, 20.0 * (1.5 if path_long else 1.0)))
                break

        # Keep land tempo (Kubic still grows tiles during conversion).
        out.extend(self._land_captures(obs, state))

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
        if not (0 <= nr < obs.H and 0 <= nc < obs.W):
            return -1.0
        gr, gc = goal
        before = abs(r - gr) + abs(c - gc)
        after = abs(nr - gr) + abs(nc - gc)
        value = 0.0
        if after < before:
            value += self.params.STRIKE_TOWARD_BIAS
        elif after > before:
            value -= 0.5
        value += min(obs.army_grid[r][c], 50) / 50.0
        if (nr, nc) == goal:
            value += 5.0
        value += defense_score_delta(obs, state, self.params, (r, c), (nr, nc)) / 20.0
        value += random.random() * 0.02
        return value
