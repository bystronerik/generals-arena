"""StrikeMCTS: tip path-gather, finish gate, then tip march to known general."""
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
from components.threat import defense_score_delta, home_threat, imminent_loss_move
from params import Params


Cell = tuple[int, int]


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

        # Imminent home loss still overrides tip feed (same as brain hard path).
        imminent = imminent_loss_move(obs, state)
        if imminent is not None:
            return imminent

        tip = self._select_tip(obs, state, goal)
        ready = self._tip_ready(obs, tip, goal)
        root_moves = self._root_moves(obs, state, goal, tip, ready)
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
            value = self._evaluate(obs, state, goal, tip, ready, node.action, deadline)
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

    def _tip_need(self, obs, tip: Cell, goal: Cell) -> int:
        """Army the tip must hold to finish: gen + margin + buffer per remaining hop."""
        need = self._capture_need(obs, goal[0], goal[1])
        blocked = self._blocked(obs)
        dist = bfs_dist(obs.H, obs.W, [goal], blocked)
        d = dist.get(tip, obs.H + obs.W)
        return need + self.params.STRIKE_PATH_BUFFER * max(0, d)

    def _tip_ready(self, obs, tip: Cell | None, goal: Cell) -> bool:
        if tip is None:
            return False
        army = obs.army_grid[tip[0]][tip[1]]
        if army <= 1:
            return False
        return army - 1 >= self._tip_need(obs, tip, goal)

    def _select_tip(self, obs, state, goal: Cell) -> Cell | None:
        """Owned cell on a path to the general with best army/(1+dist)."""
        blocked = self._blocked(obs)
        dist = bfs_dist(obs.H, obs.W, [goal], blocked)
        cached = getattr(state, "strike_tip", None)
        cached_turn = getattr(state, "strike_tip_turn", -10_000)
        if (
            cached is not None
            and obs.owner_grid[cached[0]][cached[1]] == 1
            and obs.army_grid[cached[0]][cached[1]] > 1
            and cached in dist
            and obs.turn - cached_turn < self.params.STRIKE_TIP_HOLD
        ):
            return cached

        best: Cell | None = None
        best_key = None
        for r in range(obs.H):
            for c in range(obs.W):
                if obs.owner_grid[r][c] != 1 or obs.army_grid[r][c] <= 1:
                    continue
                if (r, c) not in dist:
                    continue
                d = dist[(r, c)]
                army = obs.army_grid[r][c]
                # Prefer close + strong; break ties toward higher army then closer.
                key = (army / (1.0 + d), -d, army, -r, -c)
                if best_key is None or key > best_key:
                    best_key, best = key, (r, c)

        if best is not None:
            state.strike_tip = best
            state.strike_tip_turn = obs.turn
        return best

    def _path_gather_roots(
        self, obs, tip: Cell, blocked
    ) -> list[tuple[Action, float]]:
        """Feed surplus stacks onto the tip."""
        out: list[tuple[Action, float]] = []
        gact = gather_toward(obs, tip, blocked)
        if gact is not None:
            out.append((gact, self.params.STRIKE_TIP_FEED_BONUS))
        # Also step secondary stacks toward the tip.
        stacks: list[tuple[int, int, int]] = []
        for r in range(obs.H):
            for c in range(obs.W):
                if (r, c) == tip:
                    continue
                if obs.owner_grid[r][c] != 1 or obs.army_grid[r][c] <= 1:
                    continue
                stacks.append((obs.army_grid[r][c], r, c))
        stacks.sort(reverse=True)
        for army, r, c in stacks[:6]:
            act = step_toward(obs, (r, c), tip, blocked)
            if act is None:
                continue
            out.append((act, self.params.STRIKE_TIP_FEED_BONUS * 0.85 + army * 0.3))
        return out

    def _land_captures(
        self, obs, state, tip: Cell | None, ready: bool
    ) -> list[tuple[Action, float]]:
        """Land roots only when the tip is finish-ready (Kubic still grows then)."""
        if not ready:
            return []
        slots = self.params.STRIKE_LAND_ROOT_SLOTS
        if slots <= 0:
            return []
        out: list[tuple[Action, float]] = []
        H, W = obs.H, obs.W
        for r in range(H):
            for c in range(W):
                if tip is not None and (r, c) == tip:
                    continue  # never peel the tip for land
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
                    # Prefer land next to the tip path (near tip).
                    score = 12.0 + obs.army_grid[r][c] * 0.1
                    if tip is not None:
                        score += 20.0 / (
                            1.0 + abs(nr - tip[0]) + abs(nc - tip[1])
                        )
                    if owner == 0:
                        score += 8.0
                    else:
                        score += 15.0
                    score += defense_score_delta(
                        obs, state, self.params, (r, c), (nr, nc)
                    )
                    out.append((move_action(r, c, nr, nc, 0), score))
        out.sort(key=lambda item: -item[1])
        return out[:slots]

    def _root_moves(
        self, obs, state, goal: Cell, tip: Cell | None, ready: bool
    ) -> list[tuple[Action, float]]:
        blocked = self._blocked(obs)
        out: list[tuple[Action, float]] = []

        if tip is None:
            src = largest_owned_stack(obs)
            if src is not None:
                act = step_toward(obs, src, goal, blocked)
                if act is not None:
                    out.append((act, 1.0))
            return out

        tip_army = obs.army_grid[tip[0]][tip[1]]
        need = self._tip_need(obs, tip, goal)

        # Always prioritize path-gather while the tip is under the finish bar.
        if not ready:
            out.extend(self._path_gather_roots(obs, tip, blocked))
            # Tip may still step onto free path cells that do not spend below need.
            act = step_toward(obs, tip, goal, blocked)
            if act is not None:
                # Only allow tip advance if leaving enough for remaining path after step.
                # Conservative: require tip army already at need (ready) — so skip while feeding.
                pass
            return out if out else self._path_gather_roots(obs, tip, blocked)

        # Tip ready: march the tip; keep feeding; dilute land only lightly.
        tip_step = step_toward(obs, tip, goal, blocked)
        if tip_step is not None:
            prior = (
                self.params.STRIKE_TIP_FEED_BONUS
                + tip_army * self.params.STRIKE_TOWARD_BIAS
            )
            out.append((tip_step, prior))

        out.extend(self._path_gather_roots(obs, tip, blocked))

        # Imminent-only intercept: lethal adjacent threat (also handled above).
        threat = home_threat(obs, state, self.params)
        home = state.memory.own_general
        if (
            threat.dist is not None
            and threat.dist <= 1
            and home is not None
            and threat.nearest is not None
        ):
            intercept = step_toward(obs, tip, threat.nearest, blocked)
            if intercept is not None:
                out.append((intercept, self.params.STRIKE_TIP_FEED_BONUS * 0.5))

        out.extend(self._land_captures(obs, state, tip, ready))

        if not out:
            act = step_toward(obs, tip, goal, blocked)
            if act is not None:
                out.append((act, 1.0))
        _ = need
        return out

    def _evaluate(
        self,
        obs,
        state,
        goal: Cell,
        tip: Cell | None,
        ready: bool,
        action: Action,
        deadline: Deadline,
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
        value = 0.0

        if tip is not None and not ready:
            # Reward feeding the tip / closing onto tip.
            before = abs(r - tip[0]) + abs(c - tip[1])
            after = abs(nr - tip[0]) + abs(nc - tip[1])
            if (nr, nc) == tip or after < before:
                value += 2.5
            elif after > before:
                value -= 1.0
            if (r, c) == tip:
                value -= 1.5  # tip should not wander while underfed
        else:
            before = abs(r - gr) + abs(c - gc)
            after = abs(nr - gr) + abs(nc - gc)
            if after < before:
                value += self.params.STRIKE_TOWARD_BIAS
            elif after > before:
                value -= 0.5
            if tip is not None and (r, c) == tip and after < before:
                value += 2.0
            if (nr, nc) == goal:
                value += 5.0

        value += min(obs.army_grid[r][c], 50) / 50.0
        value += defense_score_delta(obs, state, self.params, (r, c), (nr, nc)) / 40.0
        value += random.random() * 0.02
        return value
