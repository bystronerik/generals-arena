"""StrikeMCTS: tip march after floor; gather consolidates, wave advances tip."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from components.army import (
    Action,
    is_wall,
    largest_owned_stack,
    move_action,
    neighbors,
    pass_action,
    step_toward,
)
from components.clock import Deadline
from components.conveyor import prefer_chain_roots, prune_by_clock
from mcts_diag import record_root_pick
from components.threat import defense_score_delta, imminent_loss_move, recall_armed
from components.tip import (
    feed_tip_action,
    path_feed_roots,
    select_mass_tip,
    tip_below_sight_floor,
    tip_is_ready,
)
from params import Params


Cell = tuple[int, int]


@dataclass
class StrikeStats:
    iterations: int = 0
    root_moves: int = 0
    overrode: int = 0
    prior_rank: int = -1
    best_visits: int = 0
    prior0_visits: int = 0


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

        imminent = imminent_loss_move(obs, state)
        if imminent is not None:
            return imminent

        tip = self._select_tip(obs, state, goal)
        # Brain already exclusive-feeds below TIP_AT_SIGHT_FLOOR; also gate here.
        if tip is not None and tip_below_sight_floor(obs, tip, self.params):
            feed = feed_tip_action(obs, tip)
            if feed is not None:
                self.stats.root_moves = 1
                return feed

        ready = tip_is_ready(obs, tip, goal, self.params)
        clock = state.clock_phase
        root_moves = self._root_moves(obs, state, goal, tip, ready, clock)
        self.stats.root_moves = len(root_moves)
        if not root_moves:
            return pass_action()

        max_root = self.params.MCTS_MAX_ROOT_STRIKE
        root_moves.sort(key=lambda item: -item[1])
        root = _Node(action=None, prior=1.0)
        for action, prior in root_moves[:max_root]:
            root.children.append(_Node(action=action, prior=max(prior, 0.01)))

        c = self.params.MCTS_C
        while not deadline.expired():
            node = max(root.children, key=lambda n: n.uct(root.visits, c) + n.prior)
            if node.action is None:
                break
            value = self._evaluate(
                obs, state, goal, tip, ready, node.action, deadline, clock
            )
            node.visits += 1
            node.value += value
            root.visits += 1
            self.stats.iterations += 1
            if deadline.expired():
                break

        best = max(root.children, key=lambda n: (n.visits, n.value, n.prior))
        record_root_pick(self.stats, root.children, best)
        return best.action if best.action is not None else pass_action()

    def _kill_shot(self, obs, goal) -> Action | None:
        from params import T_GENERAL

        goals = [goal] if goal is not None else []
        for r in range(obs.H):
            for c in range(obs.W):
                if obs.owner_grid[r][c] == 2 and obs.type_grid[r][c] == T_GENERAL:
                    cell = (r, c)
                    if cell not in goals:
                        goals.append(cell)
        for g in goals:
            gr, gc = g
            need = self._capture_need(obs, gr, gc)
            for r, c in neighbors(obs.H, obs.W, gr, gc):
                if obs.owner_grid[r][c] != 1:
                    continue
                if obs.army_grid[r][c] - 1 >= need:
                    return move_action(r, c, gr, gc, 0)
        return None

    def _capture_need(self, obs, gr: int, gc: int) -> int:
        if obs.turn >= self.params.DEATHTOUCH_TURN:
            # RULES §07: one attacking unit wins; leave-1 needs army >= 2.
            return 1
        defender = 0
        if obs.owner_grid[gr][gc] == 2:
            defender = obs.army_grid[gr][gc]
        return defender + self.params.FINISH_MARGIN

    def _select_tip(self, obs, state, goal: Cell) -> Cell | None:
        tip = select_mass_tip(
            obs,
            goal,
            self.params,
            getattr(state, "strike_tip", None),
            getattr(state, "strike_tip_turn", -10_000),
        )
        if tip is not None:
            state.strike_tip = tip
            state.strike_tip_turn = obs.turn
            state.muster = tip
        return tip

    def _root_moves(
        self,
        obs,
        state,
        goal: Cell,
        tip: Cell | None,
        ready: bool,
        clock: str,
    ) -> list[tuple[Action, float]]:
        blocked = lambda r, c: is_wall(obs.type_grid, r, c)
        out: list[tuple[Action, float]] = []
        muster = tip or state.muster

        if tip is None:
            src = largest_owned_stack(obs)
            if src is not None:
                act = step_toward(obs, src, goal, blocked)
                if act is not None:
                    out.append((act, 1.0))
            return out

        if clock == "gather" or not ready:
            # Consolidate into tip even if tip is "ready" during gather residues.
            out.extend(path_feed_roots(obs, tip, self.params.STRIKE_TIP_FEED_BONUS))
        else:
            tip_army = obs.army_grid[tip[0]][tip[1]]
            tip_step = step_toward(obs, tip, goal, blocked)
            if tip_step is not None:
                prior = (
                    self.params.STRIKE_TIP_FEED_BONUS
                    + tip_army * self.params.STRIKE_TOWARD_BIAS
                )
                out.append((tip_step, prior))
            # On-path captures from tip only.
            for nr, nc in neighbors(obs.H, obs.W, *tip):
                if is_wall(obs.type_grid, nr, nc):
                    continue
                owner = obs.owner_grid[nr][nc]
                if owner == 1:
                    continue
                if tip_army - 1 <= obs.army_grid[nr][nc]:
                    continue
                before = abs(tip[0] - goal[0]) + abs(tip[1] - goal[1])
                after = abs(nr - goal[0]) + abs(nc - goal[1])
                if after > before and owner != 2:
                    continue
                out.append(
                    (
                        move_action(tip[0], tip[1], nr, nc, 0),
                        self.params.STRIKE_TIP_FEED_BONUS * 0.7,
                    )
                )

        out = prune_by_clock(obs, out, clock, goal, muster)
        # Prefer tip as chain head.
        head = tip if tip is not None else state.chain_head
        out = prefer_chain_roots(obs, head, out, self.params)

        if not out:
            tip_step = step_toward(obs, tip, goal, blocked)
            if tip_step is not None:
                out.append((tip_step, 1.0))
            else:
                feed = feed_tip_action(obs, tip)
                if feed is not None:
                    out.append((feed, 1.0))
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
        clock: str,
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

        if clock == "gather" or (tip is not None and not ready):
            if tip is not None:
                before = abs(r - tip[0]) + abs(c - tip[1])
                after = abs(nr - tip[0]) + abs(nc - tip[1])
                if (nr, nc) == tip or after < before:
                    value += 2.5
                elif after > before:
                    value -= 1.0
                if (r, c) == tip:
                    value -= 1.5
        else:
            before = abs(r - gr) + abs(c - gc)
            after = abs(nr - gr) + abs(nc - gc)
            if after < before:
                value += self.params.STRIKE_TOWARD_BIAS
            elif after > before:
                value -= 0.5
            if tip is not None and (r, c) == tip and after < before:
                value += 2.0
            elif tip is not None and (r, c) != tip:
                value -= 2.0
            if (nr, nc) == goal:
                value += 5.0

        value += min(obs.army_grid[r][c], 50) / 50.0
        if recall_armed(obs, state, self.params):
            value += defense_score_delta(obs, state, self.params, (r, c), (nr, nc)) / 40.0
        value += random.random() * 0.02
        return value
