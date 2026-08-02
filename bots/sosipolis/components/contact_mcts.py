"""ContactMCTS: post-contact hunt ownership — belief, sticky probes, path score."""
from __future__ import annotations

import math
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


Cell = tuple[int, int]


@dataclass
class ContactStats:
    iterations: int = 0
    root_moves: int = 0
    overrode: int = 0
    prior_rank: int = -1
    best_visits: int = 0
    prior0_visits: int = 0
    macro_kind: str = "none"
    commit_age: int = 0
    switch_count: int = 0
    candidate_mass: float = 0.0
    waypoint: Cell | None = None


@dataclass(frozen=True)
class ProbeMacro:
    kind: str  # cluster | frontier | mid_edge | split
    waypoint: Cell
    evidence_anchor: Cell
    candidate_cells: frozenset[Cell]
    score: float


@dataclass
class ContactCommitment:
    macro: ProbeMacro
    committed_turn: int
    last_score: float
    switch_count: int = 0


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


@dataclass
class _ContactCache:
    cand_epoch: int = -1
    terrain_epoch: int = -1
    enemy_obs_epoch: int = -1
    tip: Cell | None = None
    waypoint: Cell | None = None
    tip_bfs: dict[Cell, int] = field(default_factory=dict)
    footprint_bfs: dict[Cell, int] = field(default_factory=dict)
    waypoint_bfs: dict[Cell, int] = field(default_factory=dict)
    belief: dict[Cell, float] = field(default_factory=dict)
    reveal_count: dict[Cell, int] = field(default_factory=dict)
    belief_sum: float = 0.0


class ContactMCTS:
    def __init__(self, params: Params):
        self.params = params
        self.stats = ContactStats()
        self.commitment: ContactCommitment | None = None
        self._cache = _ContactCache()

    def prepare_contact(self, obs, state, deadline: Deadline) -> ContactCommitment | None:
        """Own post-contact probe target: belief → macros → sticky commit."""
        if state.memory.enemy_general is not None:
            self.commitment = None
            state.contact_commitment = None
            return None
        if state.memory.first_contact is None and not state.enemy_land_known():
            return None

        tip = state.strike_tip or largest_owned_stack(obs)
        self._refresh_cache(obs, state, tip, deadline)
        if deadline.expired():
            if self.commitment is not None:
                state.contact_commitment = self.commitment
                state.objective = self.commitment.macro.waypoint
            return self.commitment

        macros = self._generate_probe_macros(obs, state, tip)
        chosen = self._commit_macro(obs, state, macros, tip)
        if chosen is None:
            return self.commitment

        state.contact_commitment = chosen
        state.objective = chosen.macro.waypoint
        self.commitment = chosen
        self.stats.macro_kind = chosen.macro.kind
        self.stats.commit_age = obs.turn - chosen.committed_turn
        self.stats.switch_count = chosen.switch_count
        self.stats.waypoint = chosen.macro.waypoint
        self.stats.candidate_mass = sum(
            self._cache.belief.get(c, 0.0) for c in chosen.macro.candidate_cells
        )
        return chosen

    def search(self, obs, state, deadline: Deadline) -> Action:
        self.stats = ContactStats(
            macro_kind=self.stats.macro_kind,
            commit_age=self.stats.commit_age,
            switch_count=self.stats.switch_count,
            candidate_mass=self.stats.candidate_mass,
            waypoint=self.stats.waypoint,
        )
        hunt = None
        if state.contact_commitment is not None:
            hunt = state.contact_commitment.macro.waypoint
        elif state.objective is not None:
            hunt = state.objective
        tip = state.strike_tip or largest_owned_stack(obs)
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
        if hunt is not None:
            self._ensure_waypoint_bfs(state, hunt)

        clock = state.clock_phase
        mode = self.params.CONTACT_PATH_MODE
        if mode == "macro_mcts":
            return self._search_macro_mcts(
                obs, state, hunt, tip, assault_ready, clock, deadline
            )
        return self._search_shallow(
            obs, state, hunt, tip, assault_ready, clock, deadline
        )

    # --- belief / cache -------------------------------------------------

    def _refresh_cache(self, obs, state, tip: Cell | None, deadline: Deadline) -> None:
        mem = state.memory
        cache = self._cache
        need_belief = (
            cache.cand_epoch != mem.cand_epoch
            or cache.enemy_obs_epoch != mem.enemy_obs_epoch
            or cache.terrain_epoch != mem.terrain_epoch
            or not cache.belief
        )
        need_foot = need_belief or cache.terrain_epoch != mem.terrain_epoch
        need_tip = tip != cache.tip or cache.terrain_epoch != mem.terrain_epoch

        if need_foot and not deadline.expired():
            enemies = mem.enemy_cells()
            cache.footprint_bfs = mem._bfs_multi(enemies) if enemies else {}
        if need_tip and tip is not None and not deadline.expired():
            cache.tip_bfs = mem._bfs_multi([tip])
            cache.tip = tip
        if need_belief and not deadline.expired():
            cache.belief = self._compute_belief(obs, state)
            cache.belief_sum = sum(cache.belief.values())
            cache.reveal_count = {}
            for cell in cache.belief:
                cache.reveal_count[cell] = self._reveal_mass(cell, cache.belief)
            cache.cand_epoch = mem.cand_epoch
            cache.enemy_obs_epoch = mem.enemy_obs_epoch
            cache.terrain_epoch = mem.terrain_epoch

    def _ensure_waypoint_bfs(self, state, waypoint: Cell) -> None:
        cache = self._cache
        if cache.waypoint == waypoint and cache.waypoint_bfs:
            return
        cache.waypoint_bfs = state.memory._bfs_multi([waypoint])
        cache.waypoint = waypoint

    def _compute_belief(self, obs, state) -> dict[Cell, float]:
        mem = state.memory
        params = self.params
        if not mem.candidates:
            return {}
        home = mem.own_general
        far = mem.H + mem.W
        foot = self._cache.footprint_bfs
        tip_bfs = self._cache.tip_bfs
        recent = set(
            mem.recent_enemy_cells(obs.turn, params.CONTACT_ENEMY_OBS_AGE)
        )
        belief: dict[Cell, float] = {}
        for cell in mem.candidates:
            r, c = cell
            w = 1.0
            # Geometry: prefer away from own general (spawn / opposite-side prior).
            if home is not None:
                d_home = abs(r - home[0]) + abs(c - home[1])
                w *= 1.0 + 0.04 * min(d_home, far)
            # Contact-backtrace: mass behind enemy footprint, not toward home.
            if foot:
                d_foot = foot.get(cell, far)
                w *= 1.0 + 0.08 * max(0, min(d_foot, params.HUNT_CONTACT_RADIUS))
                if home is not None:
                    # Penalize candidates that sit between home and the front.
                    mid_pen = 0.0
                    for er, ec in recent or mem.enemy_cells()[:12]:
                        if abs(er - home[0]) + abs(ec - home[1]) == 0:
                            continue
                        # Candidate closer to home than the contact → lower.
                        if abs(r - home[0]) + abs(c - home[1]) < abs(er - home[0]) + abs(
                            ec - home[1]
                        ):
                            mid_pen += 0.15
                    w /= 1.0 + mid_pen
            # Freshness / army-delta indicators near evidence.
            for er, ec in recent:
                age = max(0, obs.turn - mem.last_seen_turn[er][ec])
                freshness = 1.0 / (1.0 + 0.05 * age)
                dist = abs(r - er) + abs(c - ec)
                delta = mem.enemy_army_delta[er][ec]
                # Positive delta: reinforce near observed growth.
                # Negative delta: mass behind departure (away from tip if known).
                if delta > 0:
                    w *= 1.0 + 0.04 * min(delta, 8) * freshness / (1.0 + dist)
                elif delta < 0:
                    behind = dist
                    if tip_bfs:
                        behind = tip_bfs.get(cell, far)
                    w *= 1.0 + 0.03 * min(-delta, 8) * freshness / (1.0 + behind * 0.5)
            belief[cell] = max(w, 1e-6)
        total = sum(belief.values())
        if total <= 0:
            return {c: 1.0 / len(belief) for c in belief}
        return {c: v / total for c, v in belief.items()}

    def _reveal_mass(self, cell: Cell, belief: dict[Cell, float]) -> int:
        radius = self.params.HUNT_REVEAL_RADIUS
        r, c = cell
        count = 0
        for rr in range(r - radius, r + radius + 1):
            for cc in range(c - radius, c + radius + 1):
                if (rr, cc) in belief:
                    count += 1
        return count

    def _reveal_belief(self, cell: Cell) -> float:
        radius = self.params.HUNT_REVEAL_RADIUS
        r, c = cell
        mass = 0.0
        for rr in range(r - radius, r + radius + 1):
            for cc in range(c - radius, c + radius + 1):
                mass += self._cache.belief.get((rr, cc), 0.0)
        return mass

    # --- probe macros ---------------------------------------------------

    def _generate_probe_macros(
        self, obs, state, tip: Cell | None
    ) -> list[ProbeMacro]:
        belief = self._cache.belief
        if not belief:
            return []
        params = self.params
        far = state.H + state.W
        tip_bfs = self._cache.tip_bfs
        foot = self._cache.footprint_bfs
        evidence = state.memory.first_contact or (
            state.enemy_footprint()[0] if state.enemy_footprint() else None
        )
        if evidence is None:
            evidence = next(iter(belief))

        clusters = self._cluster_candidates(belief, params.CONTACT_CLUSTER_RADIUS)
        macros: list[ProbeMacro] = []

        for cells in clusters[:3]:
            medoid = self._weighted_medoid(cells, belief)
            if medoid is None:
                continue
            mass = sum(belief[c] for c in cells)
            dist = tip_bfs.get(medoid, far) if tip_bfs else far
            score = self._macro_score(medoid, mass, dist, evidence, foot)
            macros.append(
                ProbeMacro(
                    "cluster",
                    medoid,
                    evidence,
                    frozenset(cells),
                    score,
                )
            )
            frontier = self._frontier_on_route(obs, state, tip, medoid)
            if frontier is not None and frontier != medoid:
                fdist = tip_bfs.get(frontier, far) if tip_bfs else far
                fscore = self._macro_score(
                    frontier, mass * 0.85, fdist, evidence, foot
                )
                macros.append(
                    ProbeMacro(
                        "frontier",
                        frontier,
                        evidence,
                        frozenset(cells),
                        fscore,
                    )
                )

        # Mid-edge: between evidence and leading cluster.
        if clusters:
            lead = clusters[0]
            lead_med = self._weighted_medoid(lead, belief)
            mid = self._mid_edge_waypoint(obs, state, evidence, lead_med, belief)
            if mid is not None:
                mass = self._reveal_belief(mid)
                dist = tip_bfs.get(mid, far) if tip_bfs else far
                score = self._macro_score(mid, mass, dist, evidence, foot) + 0.15
                macros.append(
                    ProbeMacro(
                        "mid_edge",
                        mid,
                        evidence,
                        frozenset(lead),
                        score,
                    )
                )

        # Split: between top two disjoint clusters.
        if len(clusters) >= 2:
            a = self._weighted_medoid(clusters[0], belief)
            b = self._weighted_medoid(clusters[1], belief)
            split = self._split_waypoint(obs, state, a, b, belief)
            if split is not None:
                mass = self._reveal_belief(split)
                dist = tip_bfs.get(split, far) if tip_bfs else far
                score = self._macro_score(split, mass, dist, evidence, foot) + 0.2
                macros.append(
                    ProbeMacro(
                        "split",
                        split,
                        evidence,
                        frozenset(clusters[0]) | frozenset(clusters[1]),
                        score,
                    )
                )

        macros.sort(key=lambda m: (-m.score, m.waypoint[0], m.waypoint[1]))
        # Normalize scores to [0, 1] for sticky compare.
        if macros:
            top = macros[0].score
            floor = macros[-1].score
            span = max(top - floor, 1e-6)
            macros = [
                ProbeMacro(
                    m.kind,
                    m.waypoint,
                    m.evidence_anchor,
                    m.candidate_cells,
                    max(0.0, min(1.0, (m.score - floor) / span)),
                )
                for m in macros
            ]
            macros.sort(key=lambda m: (-m.score, m.waypoint[0], m.waypoint[1]))
        return macros[: params.CONTACT_MAX_MACROS]

    def _macro_score(
        self,
        waypoint: Cell,
        mass: float,
        dist: int,
        evidence: Cell,
        foot: dict[Cell, int],
    ) -> float:
        far = 64
        reveal = self._reveal_belief(waypoint)
        align = 0.0
        if foot:
            align = 1.0 / (1.0 + foot.get(waypoint, far))
        else:
            align = 1.0 / (1.0 + abs(waypoint[0] - evidence[0]) + abs(waypoint[1] - evidence[1]))
        return (
            4.0 * reveal
            + 2.0 * mass
            + 1.5 * align
            + 1.0 * (1.0 / (1.0 + dist))
            - 0.35 * (dist / max(far, 1))
        )

    def _cluster_candidates(
        self, belief: dict[Cell, float], radius: int
    ) -> list[list[Cell]]:
        remaining = sorted(belief.keys(), key=lambda c: (-belief[c], c[0], c[1]))
        clusters: list[list[Cell]] = []
        used: set[Cell] = set()
        for seed in remaining:
            if seed in used:
                continue
            cluster = [seed]
            used.add(seed)
            for cell in remaining:
                if cell in used:
                    continue
                if abs(cell[0] - seed[0]) + abs(cell[1] - seed[1]) <= radius:
                    cluster.append(cell)
                    used.add(cell)
            clusters.append(cluster)
            if len(clusters) >= 3:
                break
        clusters.sort(
            key=lambda cells: (-sum(belief[c] for c in cells), cells[0][0], cells[0][1])
        )
        return clusters

    def _weighted_medoid(
        self, cells: list[Cell], belief: dict[Cell, float]
    ) -> Cell | None:
        if not cells:
            return None
        best = None
        best_key = None
        for cand in cells:
            cost = 0.0
            for other in cells:
                d = abs(cand[0] - other[0]) + abs(cand[1] - other[1])
                cost += d * belief.get(other, 0.0)
            key = (cost, -belief.get(cand, 0.0), cand[0], cand[1])
            if best_key is None or key < best_key:
                best_key, best = key, cand
        return best

    def _frontier_on_route(
        self, obs, state, tip: Cell | None, goal: Cell
    ) -> Cell | None:
        if tip is None:
            return None
        dist = self._cache.tip_bfs
        if goal not in dist:
            return None
        # Walk from tip toward goal; first owned→non-own step destination.
        cur = tip
        guard = obs.H * obs.W + 2
        while cur != goal and guard > 0:
            guard -= 1
            best = None
            best_d = dist.get(cur, 10_000)
            for nr, nc in neighbors(obs.H, obs.W, *cur):
                if (nr, nc) not in dist:
                    continue
                if dist[(nr, nc)] < best_d:
                    best_d = dist[(nr, nc)]
                    best = (nr, nc)
            if best is None:
                return None
            if obs.owner_grid[cur[0]][cur[1]] == 1 and obs.owner_grid[best[0]][best[1]] != 1:
                if state.memory.is_passable_belief(best[0], best[1]):
                    return best
            cur = best
        return goal

    def _mid_edge_waypoint(
        self,
        obs,
        state,
        evidence: Cell,
        lead: Cell | None,
        belief: dict[Cell, float],
    ) -> Cell | None:
        if lead is None:
            return None
        # Candidates on the bounding box edge between evidence and lead.
        r0, c0 = evidence
        r1, c1 = lead
        mid_r = (r0 + r1) // 2
        mid_c = (c0 + c1) // 2
        # Prefer cells on the outer edge of the board toward the lead side.
        H, W = obs.H, obs.W
        edge_cells: list[Cell] = []
        if abs(c1 - c0) >= abs(r1 - r0):
            # Vertical edge probe (right/left side).
            col = W - 1 if c1 >= c0 else 0
            for r in range(min(r0, r1), max(r0, r1) + 1):
                edge_cells.append((r, col))
            edge_cells.append((mid_r, col))
        else:
            row = H - 1 if r1 >= r0 else 0
            for c in range(min(c0, c1), max(c0, c1) + 1):
                edge_cells.append((row, c))
            edge_cells.append((row, mid_c))
        edge_cells.append((mid_r, mid_c))

        best = None
        best_key = None
        for cell in edge_cells:
            r, c = cell
            if not (0 <= r < H and 0 <= c < W):
                continue
            if not state.memory.is_passable_belief(r, c):
                continue
            if state.memory.ever_seen[r][c] and (r, c) not in belief:
                # Already seen non-candidate — still ok as waypoint if passable.
                pass
            mass = self._reveal_belief(cell)
            dist = self._cache.tip_bfs.get(cell, H + W)
            key = (-mass, dist, r, c)
            if best_key is None or key < best_key:
                best_key, best = key, cell
        return best

    def _split_waypoint(
        self,
        obs,
        state,
        a: Cell | None,
        b: Cell | None,
        belief: dict[Cell, float],
    ) -> Cell | None:
        if a is None or b is None:
            return None
        mid = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)
        H, W = obs.H, obs.W
        candidates = [mid, a, b]
        for r in range(min(a[0], b[0]), max(a[0], b[0]) + 1):
            candidates.append((r, mid[1]))
        for c in range(min(a[1], b[1]), max(a[1], b[1]) + 1):
            candidates.append((mid[0], c))
        best = None
        best_key = None
        for cell in candidates:
            r, c = cell
            if not (0 <= r < H and 0 <= c < W):
                continue
            if not state.memory.is_passable_belief(r, c):
                continue
            da = abs(r - a[0]) + abs(c - a[1])
            db = abs(r - b[0]) + abs(c - b[1])
            balance = max(da, db)
            mass = self._reveal_belief(cell)
            dist = self._cache.tip_bfs.get(cell, H + W)
            key = (balance, -mass, dist, r, c)
            if best_key is None or key < best_key:
                best_key, best = key, cell
        return best

    # --- sticky commitment ----------------------------------------------

    def _commit_macro(
        self,
        obs,
        state,
        macros: list[ProbeMacro],
        tip: Cell | None,
    ) -> ContactCommitment | None:
        if not macros:
            return self.commitment
        best = macros[0]
        cur = self.commitment
        params = self.params

        if cur is None:
            return ContactCommitment(best, obs.turn, best.score, 0)

        # Invalidate current if broken.
        if self._commitment_invalid(obs, state, cur, tip):
            return ContactCommitment(
                best, obs.turn, best.score, cur.switch_count + 1
            )

        age = obs.turn - cur.committed_turn
        # Strong opposite-side evidence can force an early switch.
        opp = self._opposite_side_evidence(obs, state, cur)
        can_switch = age >= params.CONTACT_COMMIT_MIN_TURNS or opp
        if not can_switch:
            # Refresh score mirror only.
            cur.last_score = cur.macro.score
            return cur

        threshold = cur.last_score * params.CONTACT_SWITCH_RATIO + params.CONTACT_SWITCH_MARGIN
        # Prefer a macro whose score clearly beats the sticky one.
        challenger = best
        if best.waypoint == cur.macro.waypoint:
            # Same waypoint: keep.
            cur.last_score = max(cur.last_score, best.score)
            return cur
        if challenger.score >= threshold:
            return ContactCommitment(
                challenger, obs.turn, challenger.score, cur.switch_count + 1
            )
        cur.last_score = max(cur.last_score * 0.98, challenger.score * 0.5)
        return cur

    def _commitment_invalid(
        self, obs, state, cur: ContactCommitment, tip: Cell | None
    ) -> bool:
        wp = cur.macro.waypoint
        r, c = wp
        if not (0 <= r < obs.H and 0 <= c < obs.W):
            return True
        if not state.memory.is_passable_belief(r, c):
            return True
        # Waypoint reached / visible and no longer covers candidates.
        if state.memory.ever_seen[r][c]:
            if self._reveal_belief(wp) <= 1e-9 and wp not in state.memory.candidates:
                return True
        if tip is not None and tip == wp:
            if self._reveal_belief(wp) <= 1e-9:
                return True
        # No overlapping candidates left.
        if cur.macro.candidate_cells and not (
            cur.macro.candidate_cells & state.memory.candidates
        ):
            return True
        return False

    def _opposite_side_evidence(
        self, obs, state, cur: ContactCommitment
    ) -> bool:
        """New enemy cell far from current evidence anchor."""
        mem = state.memory
        anchor = cur.macro.evidence_anchor
        foot = self._cache.footprint_bfs
        recent = mem.recent_enemy_cells(obs.turn, 2)
        for cell in recent:
            if cell == anchor:
                continue
            if foot:
                d = foot.get(cell, 0)
                # Distance between new cell and anchor via BFS approx.
                d_anchor = abs(cell[0] - anchor[0]) + abs(cell[1] - anchor[1])
            else:
                d_anchor = abs(cell[0] - anchor[0]) + abs(cell[1] - anchor[1])
            if d_anchor >= self.params.CONTACT_OPP_SIDE_BFS:
                return True
        return False

    # --- shallow path scorer (Phase 1) ----------------------------------

    def _search_shallow(
        self,
        obs,
        state,
        hunt: Cell | None,
        tip: Cell | None,
        assault_ready: bool,
        clock: str,
        deadline: Deadline,
    ) -> Action:
        root_moves = self._root_moves(obs, state, hunt, tip, assault_ready, clock)
        self.stats.root_moves = len(root_moves)
        if not root_moves:
            return pass_action()

        scored: list[tuple[Action, float]] = []
        for action, prior in root_moves:
            if deadline.expired():
                break
            s = self._shallow_action_score(
                obs, state, action, hunt, tip, assault_ready, prior
            )
            scored.append((action, s))
        if not scored:
            return root_moves[0][0]

        scored.sort(
            key=lambda item: (
                -item[1],
                -self._vision_gain(obs, state, item[0]),
                item[0][1],
                item[0][2],
            )
        )
        best_action, best_score = scored[0]
        # Diagnostics compatible with prior UCT fields.
        self.stats.iterations = len(scored)
        self.stats.best_visits = 1
        self.stats.prior0_visits = 1
        self.stats.prior_rank = 0
        prior_best = root_moves[0][0]
        self.stats.overrode = 0 if best_action == prior_best else 1
        return best_action

    def _shallow_action_score(
        self,
        obs,
        state,
        action: Action,
        hunt: Cell | None,
        tip: Cell | None,
        assault_ready: bool,
        prior: float,
    ) -> float:
        if action[0] != 0:
            return prior * 0.01
        _, r, c, d, _ = action
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        dr, dc = dirs[d]
        nr, nc = r + dr, c + dc
        if not (0 <= nr < obs.H and 0 <= nc < obs.W):
            return -1.0
        score = self._score(obs, state, r, c, nr, nc, hunt, assault_ready)
        if score < 0:
            return -1.0
        # BFS closure to waypoint.
        if hunt is not None and self._cache.waypoint_bfs:
            before = self._cache.waypoint_bfs.get((r, c), obs.H + obs.W)
            after = self._cache.waypoint_bfs.get((nr, nc), obs.H + obs.W)
            if after < before:
                score += 80.0
            elif after > before:
                score -= 40.0
        # One-step vision gain on candidates.
        score += 25.0 * self._cell_reveal_belief(nr, nc)
        # Two-step look-ahead: best next closure.
        if hunt is not None:
            best_next = 0.0
            for n2r, n2c in neighbors(obs.H, obs.W, nr, nc):
                if is_wall(obs.type_grid, n2r, n2c):
                    continue
                gain = self._cell_reveal_belief(n2r, n2c)
                close = 0.0
                if self._cache.waypoint_bfs:
                    d1 = self._cache.waypoint_bfs.get((nr, nc), 99)
                    d2 = self._cache.waypoint_bfs.get((n2r, n2c), 99)
                    if d2 < d1:
                        close = 1.0
                best_next = max(best_next, 12.0 * gain + 8.0 * close)
            score += best_next
        if state.chain_head is not None and (r, c) == state.chain_head:
            score += 20.0
        if tip is not None and (r, c) == tip:
            score += 15.0
        return score + 0.01 * prior

    def _cell_reveal_belief(self, r: int, c: int) -> float:
        return self._reveal_belief((r, c))

    def _vision_gain(self, obs, state, action: Action) -> int:
        if action[0] != 0:
            return 0
        _, r, c, d, _ = action
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        dr, dc = dirs[d]
        nr, nc = r + dr, c + dc
        if not (0 <= nr < obs.H and 0 <= nc < obs.W):
            return 0
        return self._cache.reveal_count.get((nr, nc), 0)

    # --- Phase 2 macro MCTS ---------------------------------------------

    def _search_macro_mcts(
        self,
        obs,
        state,
        hunt: Cell | None,
        tip: Cell | None,
        assault_ready: bool,
        clock: str,
        deadline: Deadline,
    ) -> Action:
        """Multi-depth macro tree over tip steps toward committed / alt waypoints."""
        root_moves = self._root_moves(obs, state, hunt, tip, assault_ready, clock)
        # Also allow tip steps toward top alternate macros if committed.
        if state.contact_commitment is not None and tip is not None and clock == "wave":
            blocked = lambda rr, cc: is_wall(obs.type_grid, rr, cc)
            alts = self._generate_probe_macros(obs, state, tip)[:3]
            for macro in alts:
                act = step_toward(obs, tip, macro.waypoint, blocked)
                if act is not None:
                    root_moves.append((act, 50.0 + 40.0 * macro.score))

        root_moves = prune_by_clock(
            obs, root_moves, clock, hunt, state.muster or tip
        )
        head = tip if (clock == "wave" and tip is not None) else state.chain_head
        root_moves = prefer_chain_roots(obs, head, root_moves, self.params)

        self.stats.root_moves = len(root_moves)
        if not root_moves:
            return pass_action()

        max_root = min(
            self.params.CONTACT_MACRO_ROOTS,
            self.params.MCTS_MAX_ROOT_WAVE
            if clock == "wave"
            else self.params.MCTS_MAX_ROOT_GATHER,
        )
        root_moves.sort(key=lambda item: -item[1])
        root = _Node(action=None, prior=1.0)
        for action, prior in root_moves[:max_root]:
            root.children.append(_Node(action=action, prior=max(prior, 0.01)))

        c = self.params.MCTS_C
        depth = self.params.CONTACT_MACRO_DEPTH
        while not deadline.expired():
            node = max(
                root.children, key=lambda n: n.uct(root.visits, c) + n.prior
            )
            if node.action is None:
                break
            value = self._macro_rollout(
                obs, state, node.action, hunt, tip, assault_ready, depth, deadline
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

    def _macro_rollout(
        self,
        obs,
        state,
        action: Action,
        hunt: Cell | None,
        tip: Cell | None,
        assault_ready: bool,
        depth: int,
        deadline: Deadline,
    ) -> float:
        if deadline.expired() or action[0] != 0:
            return 0.0
        value = self._shallow_action_score(
            obs, state, action, hunt, tip, assault_ready, 1.0
        ) / 200.0
        # Simulated multi-step progress toward hunt without mutating obs.
        _, r, c, d, _ = action
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        dr, dc = dirs[d]
        cr, cc = r + dr, c + dc
        for step in range(1, depth):
            if hunt is None or deadline.expired():
                break
            if not (0 <= cr < obs.H and 0 <= cc < obs.W):
                break
            value += 0.4 * self._cell_reveal_belief(cr, cc)
            if self._cache.waypoint_bfs:
                here = self._cache.waypoint_bfs.get((cr, cc), 99)
                # Greedy step closing on waypoint.
                best = None
                best_d = here
                for nr, nc in neighbors(obs.H, obs.W, cr, cc):
                    if is_wall(obs.type_grid, nr, nc):
                        continue
                    dd = self._cache.waypoint_bfs.get((nr, nc), 99)
                    if dd < best_d:
                        best_d = dd
                        best = (nr, nc)
                if best is None:
                    break
                if best_d < here:
                    value += 0.6
                cr, cc = best
            else:
                break
        return value

    # --- roots / scoring (shared) ---------------------------------------

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
        dead = state.dead_pockets
        sector_cands = self._contact_sector_candidates(state)
        blocked = lambda rr, cc: is_wall(obs.type_grid, rr, cc)
        out: list[tuple[Action, float]] = []
        muster = state.muster or tip

        if clock == "gather":
            # Soft feed to tip only; one stack task (Kubic).
            rally = tip or muster
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
            # Wave: single tip → committed waypoint.
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
                gact = gather_toward(obs, hunt, blocked)
                if gact is not None:
                    out.append((gact, 90.0 if assault_ready else 70.0))

            # On-route neutrals/enemies from tip + chain only.
            seeds = []
            if tip is not None:
                seeds.append(tip)
            if state.chain_head is not None:
                seeds.append(state.chain_head)
            for r, c in seeds:
                if obs.owner_grid[r][c] != 1 or obs.army_grid[r][c] <= 1:
                    continue
                for nr, nc in neighbors(obs.H, obs.W, r, c):
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
