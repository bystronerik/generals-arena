"""ContactMCTS: hunt after enemy land known; clocked gather/wave + chain."""
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
from components.conveyor import prefer_chain_roots, prune_by_clock
from mcts_diag import record_root_pick
from components.threat import defense_score_delta, recall_armed
from components.tip import feed_tip_action, path_feed_roots, tip_is_ready
from params import Params


@dataclass
class ContactStats:
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


class ContactMCTS:
    def __init__(self, params: Params):
        self.params = params
        self.stats = ContactStats()

    def search(self, obs, state, deadline: Deadline) -> Action:
        self.stats = ContactStats()
        hunt = state.objective or state.memory.hunt_target(obs, self.params)
        tip = largest_owned_stack(obs)
        stack_army = obs.army_grid[tip[0]][tip[1]] if tip else 0
        assault_ready = stack_army >= self.params.CONTACT_ASSAULT_STACK
        if (
            not assault_ready
            and tip is not None
            and hunt is not None
            and tip_is_ready(obs, tip, hunt, self.params)
        ):
            assault_ready = True

        self._sharpen_priors(state, hunt)
        clock = state.clock_phase
        root_moves = self._root_moves(obs, state, hunt, tip, assault_ready, clock)
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
        while not deadline.expired():
            node = max(
                root.children, key=lambda n: n.uct(root.visits, c) + n.prior
            )
            if node.action is None:
                break
            value = self._rollout(
                obs, state, node.action, hunt, assault_ready, deadline
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

    def _sharpen_priors(self, state, hunt) -> None:
        focus = hunt
        if focus is None:
            footprint = state.enemy_footprint()
            if not footprint:
                return
            cr = sum(r for r, _ in footprint) / len(footprint)
            cc = sum(c for _, c in footprint) / len(footprint)
            focus = (int(round(cr)), int(round(cc)))
        state.sections.reweight_contact(focus, self.params.CONTACT_SECTOR_FOCUS)
        if state.memory.candidates:
            state.sections.prune_to_candidates(state.memory.candidates)

    def _contact_sector_candidates(self, state) -> set:
        top = state.sections.top_section()
        return {
            cell
            for cell in state.memory.candidates
            if state.sections.section_of(*cell) == top
            or state.sections.score_cell(*cell) >= 0.08
        }

    def _root_moves(self, obs, state, hunt, tip, assault_ready: bool, clock: str):
        H, W = obs.H, obs.W
        dead = state.dead_pockets
        sector_cands = self._contact_sector_candidates(state)
        blocked = lambda rr, cc: is_wall(obs.type_grid, rr, cc)
        out: list[tuple[Action, float]] = []
        muster = state.muster or tip

        if clock == "gather":
            # Soft feed to muster/tip only; no hunt march.
            rally = muster or tip
            if hunt is not None and tip is None:
                tip = largest_owned_stack(obs)
            if hunt is not None and tip is not None:
                # Prefer tip that will march next wave: feed the tip near hunt.
                rally = tip
            if rally is not None:
                out.extend(
                    path_feed_roots(
                        obs, rally, self.params.STRIKE_TIP_FEED_BONUS * 0.45
                    )
                )
                gact = gather_toward(obs, rally, blocked)
                if gact is not None:
                    out.append((gact, 70.0))
        else:
            # Wave: tip→hunt first (contact→sight tempo). Soft feed stays off.
            if tip is None:
                tip = largest_owned_stack(obs)
            if hunt is not None and tip is not None:
                tip_army = obs.army_grid[tip[0]][tip[1]]
                act = step_toward(obs, tip, hunt, blocked)
                if act is not None:
                    prior = self.params.HUNT_STEP_BONUS * 2.0 + tip_army
                    if not state.memory.ever_seen[hunt[0]][hunt[1]]:
                        prior += 60.0
                    out.append((act, prior))
                # Second stack also closes if tip is thin.
                stacks: list[tuple[int, int, int]] = []
                for r in range(H):
                    for c in range(W):
                        if (r, c) == tip:
                            continue
                        if obs.owner_grid[r][c] != 1 or obs.army_grid[r][c] <= 1:
                            continue
                        stacks.append((obs.army_grid[r][c], r, c))
                stacks.sort(reverse=True)
                for army, r, c in stacks[:4]:
                    act = step_toward(obs, (r, c), hunt, blocked)
                    if act is None:
                        continue
                    prior = self.params.HUNT_STEP_BONUS + army * 0.5
                    out.append((act, prior))
                gact = gather_toward(obs, hunt, blocked)
                if gact is not None:
                    out.append((gact, 90.0 if assault_ready else 70.0))

            # On-route neutrals/enemies from tip + chain (post-contact expand).
            seeds = []
            if tip is not None:
                seeds.append(tip)
            if state.chain_head is not None:
                seeds.append(state.chain_head)
            for r, c in seeds:
                if obs.owner_grid[r][c] != 1 or obs.army_grid[r][c] <= 1:
                    continue
                for nr, nc in neighbors(H, W, r, c):
                    if is_wall(obs.type_grid, nr, nc):
                        continue
                    if (nr, nc) in dead and obs.owner_grid[nr][nc] != 2:
                        if not self._pocket_ok(state, nr, nc, sector_cands):
                            continue
                    prior = self._score(
                        obs, state, r, c, nr, nc, hunt, assault_ready
                    )
                    if prior < 0:
                        continue
                    out.append((move_action(r, c, nr, nc, 0), prior))

        out = prune_by_clock(obs, out, clock, hunt, muster)
        # Prefer tip as chain head during contact wave.
        head = tip if (clock == "wave" and tip is not None) else state.chain_head
        out = prefer_chain_roots(obs, head, out, self.params)

        if not out and hunt is not None:
            act = gather_toward(obs, hunt, blocked)
            if act is not None:
                out.append((act, 1.0))
        if not out and tip is not None:
            feed = feed_tip_action(obs, tip)
            if feed is not None:
                out.append((feed, 1.0))
        return out

    def _pocket_ok(self, state, r: int, c: int, sector_cands) -> bool:
        if (r, c) in sector_cands:
            return True
        if any(cell in state.dead_pockets for cell in sector_cands):
            return len(sector_cands) <= 5
        return False

    def _score(
        self, obs, state, r: int, c: int, nr: int, nc: int, hunt, assault_ready: bool
    ) -> float:
        army = obs.army_grid[r][c]
        dest_army = obs.army_grid[nr][nc]
        owner = obs.owner_grid[nr][nc]
        if owner != 1 and army - 1 <= dest_army:
            return -1.0

        prior = state.sections.score_cell(nr, nc)
        score = float(army) + 40.0 * prior
        hunt_scale = 1.0 if assault_ready else 0.55
        if hunt is not None:
            before = abs(r - hunt[0]) + abs(c - hunt[1])
            after = abs(nr - hunt[0]) + abs(nc - hunt[1])
            if after < before:
                score += self.params.HUNT_STEP_BONUS * 0.35 * hunt_scale
                if not state.memory.ever_seen[nr][nc]:
                    score += self.params.HUNT_STEP_BONUS * 0.25 * hunt_scale
            elif after > before:
                score -= 25.0 * hunt_scale

        if owner == 2:
            score += self.params.CONTACT_ENEMY_BONUS + min(dest_army, 20)
            score += 15.0 * prior
        elif owner == 0:
            score += self.params.CONTACT_LAND_BONUS
            if not state.memory.ever_seen[nr][nc]:
                score += 40.0 * (0.3 + prior) * hunt_scale
                if hunt is not None:
                    score += (
                        self.params.CONTACT_CANDIDATE_BONUS
                        * hunt_scale
                        / (1.0 + abs(nr - hunt[0]) + abs(nc - hunt[1]))
                    )
            else:
                score += 8.0
        else:
            score += 1.0

        if owner == 0 and not state.memory.ever_seen[nr][nc] and prior < 0.1:
            owned_adj = sum(
                1
                for ar, ac in neighbors(obs.H, obs.W, nr, nc)
                if obs.owner_grid[ar][ac] == 1
            )
            if owned_adj <= self.params.CORRIDOR_WIDTH_MAX:
                if hunt is None or abs(nr - hunt[0]) + abs(nc - hunt[1]) >= abs(
                    r - hunt[0]
                ) + abs(c - hunt[1]):
                    score *= 0.4
        if recall_armed(obs, state, self.params):
            score += defense_score_delta(obs, state, self.params, (r, c), (nr, nc))
        return score

    def _rollout(
        self, obs, state, action: Action, hunt, assault_ready: bool, deadline: Deadline
    ) -> float:
        if deadline.expired() or action[0] != 0:
            return 0.0
        _, r, c, d, _ = action
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        dr, dc = dirs[d]
        nr, nc = r + dr, c + dc
        if not (0 <= nr < obs.H and 0 <= nc < obs.W):
            return -1.0
        value = self._score(obs, state, r, c, nr, nc, hunt, assault_ready) / 150.0
        if state.chain_head is not None and (r, c) == state.chain_head:
            value += 1.0
        if hunt is not None and (assault_ready or state.clock_phase == "wave"):
            before = abs(r - hunt[0]) + abs(c - hunt[1])
            after = abs(nr - hunt[0]) + abs(nc - hunt[1])
            if after < before:
                value += 1.5
        value += random.random() * 0.04
        return value
