"""SearchMCTS: explore under mod-50 clock + chain masks."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from components.army import (
    Action,
    gather_toward,
    is_wall,
    largest_owned_stack,
    move_action,
    neighbors,
    pass_action,
    step_toward,
)
from components.clock import Deadline
from components.conveyor import prefer_chain_roots, prune_by_clock, prune_opening
from components.threat import defense_score_delta, recall_armed
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

    def search(self, obs, state, deadline: Deadline, opening: bool = False) -> Action:
        self.stats = SearchStats()
        hunt = state.objective or state.memory.hunt_target(obs, self.params)
        clock = state.clock_phase
        root_moves = self._root_moves(obs, state, hunt, clock, opening)
        self.stats.root_moves = len(root_moves)
        if not root_moves:
            return pass_action()

        max_root = (
            self.params.MCTS_MAX_ROOT_GATHER
            if clock == "gather"
            else self.params.MCTS_MAX_ROOT_WAVE
        )
        root_moves.sort(key=lambda item: -item[1])
        root = _Node(action=None, prior=1.0)
        for action, prior in root_moves[:max_root]:
            root.children.append(_Node(action=action, prior=max(prior, 0.01)))

        c = self.params.MCTS_C
        depth = self.params.MCTS_ROLLOUT_DEPTH
        while not deadline.expired():
            node = self._select(root, c)
            if node.action is None:
                break
            value = self._rollout(obs, state, node.action, depth, hunt, deadline)
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

    def _score_expand(
        self, obs, state, r: int, c: int, nr: int, nc: int, hunt, clock: str
    ) -> float:
        army = obs.army_grid[r][c]
        dest_army = obs.army_grid[nr][nc]
        owner = obs.owner_grid[nr][nc]
        if owner != 1 and army - 1 <= dest_army:
            return -1.0

        score = float(army)
        muster = state.muster
        if clock == "gather":
            if owner != 1:
                return -1.0
            if muster is not None:
                before = abs(r - muster[0]) + abs(c - muster[1])
                after = abs(nr - muster[0]) + abs(nc - muster[1])
                if after < before:
                    score += 60.0
            # Drain bonus from structures.
            from params import T_CASTLE, T_GENERAL

            if obs.type_grid[r][c] in (T_GENERAL, T_CASTLE):
                score += 40.0
            score += 2.0
        else:
            if owner == 0:
                score += self.params.SEARCH_LAND_BONUS
                if not state.memory.ever_seen[nr][nc]:
                    score += self.params.SEARCH_FOG_BONUS * (
                        0.4 + state.sections.score_cell(nr, nc)
                    )
                else:
                    score += self.params.SEARCH_LAND_BONUS * 0.24
            elif owner == 2:
                if army - 1 > dest_army:
                    score += self.params.SEARCH_ENEMY_BONUS + min(dest_army, 20)
                else:
                    return -1.0
            else:
                score += 2.0

            score += 15.0 * state.sections.score_cell(nr, nc)
            if hunt is not None:
                before = abs(r - hunt[0]) + abs(c - hunt[1])
                after = abs(nr - hunt[0]) + abs(nc - hunt[1])
                if after < before:
                    score += self.params.SEARCH_HUNT_BONUS
                    if not state.memory.ever_seen[nr][nc]:
                        score += self.params.SEARCH_HUNT_BONUS * 0.4
                elif after > before:
                    score -= self.params.SEARCH_HUNT_BONUS * 0.25

            if owner == 0 and not state.memory.ever_seen[nr][nc]:
                owned_adj = sum(
                    1
                    for ar, ac in neighbors(obs.H, obs.W, nr, nc)
                    if obs.owner_grid[ar][ac] == 1
                )
                prior = state.sections.score_cell(nr, nc)
                if owned_adj <= self.params.CORRIDOR_WIDTH_MAX and prior < 0.12:
                    closing = (
                        hunt is not None
                        and abs(nr - hunt[0]) + abs(nc - hunt[1])
                        < abs(r - hunt[0]) + abs(c - hunt[1])
                    )
                    if not closing:
                        score *= 0.5

        if recall_armed(obs, state, self.params):
            score += defense_score_delta(obs, state, self.params, (r, c), (nr, nc))
        return score

    def _root_moves(
        self, obs, state, hunt, clock: str, opening: bool
    ) -> list[tuple[Action, float]]:
        H, W = obs.H, obs.W
        dead = state.dead_pockets
        blocked = lambda rr, cc: is_wall(obs.type_grid, rr, cc)
        out: list[tuple[Action, float]] = []
        muster = state.muster

        if clock == "gather":
            rally = muster or hunt or self._best_frontier(obs, state)
            if rally is not None:
                gact = gather_toward(obs, rally, blocked)
                if gact is not None:
                    out.append((gact, 80.0))
            # Structure drains / own-land steps toward muster.
            for r in range(H):
                for c in range(W):
                    if obs.owner_grid[r][c] != 1 or obs.army_grid[r][c] <= 1:
                        continue
                    for nr, nc in neighbors(H, W, r, c):
                        if is_wall(obs.type_grid, nr, nc):
                            continue
                        prior = self._score_expand(
                            obs, state, r, c, nr, nc, hunt, clock
                        )
                        if prior < 0:
                            continue
                        out.append((move_action(r, c, nr, nc, 0), prior))
        else:
            if hunt is not None:
                stack = largest_owned_stack(obs)
                if stack is not None:
                    act = step_toward(obs, stack, hunt, blocked)
                    if act is not None:
                        out.append(
                            (
                                act,
                                self.params.SEARCH_HUNT_BONUS
                                + obs.army_grid[stack[0]][stack[1]] * 0.3,
                            )
                        )
                gact = gather_toward(obs, hunt, blocked)
                if gact is not None:
                    out.append((gact, self.params.SEARCH_HUNT_BONUS * 0.6))

            # Prefer chain / tip neighbourhood over full H×W scan.
            seeds: list[tuple[int, int]] = []
            if state.chain_head is not None:
                seeds.append(state.chain_head)
            tip = largest_owned_stack(obs)
            if tip is not None:
                seeds.append(tip)
            for r, c in seeds:
                if obs.owner_grid[r][c] != 1 or obs.army_grid[r][c] <= 1:
                    continue
                for nr, nc in neighbors(H, W, r, c):
                    if is_wall(obs.type_grid, nr, nc):
                        continue
                    if (nr, nc) in dead and obs.owner_grid[nr][nc] != 2:
                        if not self._pocket_has_candidates(state, nr, nc):
                            continue
                    prior = self._score_expand(
                        obs, state, r, c, nr, nc, hunt, clock
                    )
                    if prior < 0:
                        continue
                    out.append((move_action(r, c, nr, nc, 0), prior))

            # Fallback: light frontier scan from owned border only.
            if len(out) < 4:
                for r in range(H):
                    for c in range(W):
                        if obs.owner_grid[r][c] != 1 or obs.army_grid[r][c] <= 1:
                            continue
                        for nr, nc in neighbors(H, W, r, c):
                            if is_wall(obs.type_grid, nr, nc):
                                continue
                            if obs.owner_grid[nr][nc] == 1:
                                continue
                            if (nr, nc) in dead and obs.owner_grid[nr][nc] != 2:
                                continue
                            prior = self._score_expand(
                                obs, state, r, c, nr, nc, hunt, clock
                            )
                            if prior < 0:
                                continue
                            out.append((move_action(r, c, nr, nc, 0), prior))

        out = prune_by_clock(obs, out, clock, hunt, muster)
        if opening:
            out = prune_opening(obs, state, out, self.params)
        out = prefer_chain_roots(obs, state.chain_head, out, self.params)

        if not out:
            rally = hunt or muster or self._best_frontier(obs, state)
            if rally is not None:
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
        self, obs, state, action: Action, depth: int, hunt, deadline: Deadline
    ) -> float:
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
        clock = state.clock_phase
        value = self._score_expand(obs, state, r, c, nr, nc, hunt, clock) / 100.0
        if state.chain_head is not None and (r, c) == state.chain_head:
            value += 1.0
        if hunt is not None:
            before = abs(r - hunt[0]) + abs(c - hunt[1])
            after = abs(nr - hunt[0]) + abs(nc - hunt[1])
            if after < before:
                value += 1.2
        value += random.random() * 0.05
        _ = depth
        return value
