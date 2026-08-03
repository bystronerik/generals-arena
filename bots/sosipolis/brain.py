"""Sosipolis execution: Kubic priority shell + Search / Contact / Strike MCTS."""
from __future__ import annotations

import time

from components.army import largest_owned_stack, move_action, neighbors, pass_action
from components.clock import Deadline, ms_since
from components.contact_mcts import ContactMCTS
from components.conveyor import action_src_dst, has_leave1_move, mod50_phase, update_chain
from components.economy import Economy, count_owned_castles
from components.expand import expand_move, far_haul_capture
from components.opening import decide_opening
from components.search_mcts import SearchMCTS
from components.strike_mcts import StrikeMCTS
from components.threat import imminent_loss_move, recall_move
from components.tip import (
    feed_tip_action,
    select_mass_tip,
    tip_below_sight_floor,
    tip_is_ready,
)
from params import PARAMS, Params
from state import GameState


def _cell_tok(cell) -> str:
    if cell is None:
        return "none"
    return f"{cell[0]},{cell[1]}"


class Agent:
    def __init__(self, player_id: int, H: int, W: int, params: Params | None = None):
        self.player_id = player_id
        self.H = H
        self.W = W
        self.params = params or PARAMS
        self.state = GameState(H, W, self.params)
        self.search = SearchMCTS(self.params)
        self.contact = ContactMCTS(self.params)
        self.strike = StrikeMCTS(self.params)
        self.economy = Economy(self.params)
        self._first_move = True
        # Probe fields
        self.phase = "search"
        self.searched = False
        self.search_iters = 0
        self.move_ms = 0
        self.candidate_count = 0
        self.top_section = 0
        self.pocket_skips = 0
        self.enemy_general_sighted = 0
        self.castles_built_probe = 0
        self.clock_phase = "wave"
        self.chain_head = None
        self.recall_fired = 0
        self.land_at_50 = None
        self.first_castle_turn = -1
        self._prev_castles = 0
        self.enemy_gen = "none"
        self.tip = None
        self.tip_army = 0
        self.tip_dist_goal = -1
        self.branch = "mcts"
        self.toward = 0
        self.objective = None
        self.chain_continued = 0
        self.hunt = None
        self.muster = None
        self.tip_ready = 0
        self.root_n = 0
        self.overrode = 0
        self.prior_rank = -1
        self.best_visits = 0
        self.prior0_visits = 0
        self._chain_head_before = None
        self.contact_waypoint = None
        self.contact_macro = "none"
        self.contact_commit_turn = -1
        self.contact_switches = 0
        self.contact_macro_score = 0.0
        self.contact_candidate_mass = 0.0

    def _clear_mcts_diag(self) -> None:
        self.overrode = 0
        self.prior_rank = -1
        self.best_visits = 0
        self.prior0_visits = 0
        self.root_n = 0

    def _pull_mcts_diag(self, stats) -> None:
        self.overrode = int(getattr(stats, "overrode", 0))
        self.prior_rank = int(getattr(stats, "prior_rank", -1))
        self.best_visits = int(getattr(stats, "best_visits", 0))
        self.prior0_visits = int(getattr(stats, "prior0_visits", 0))
        self.root_n = int(getattr(stats, "root_moves", 0))

    def act(self, obs):
        start = time.monotonic()
        try:
            return self._act(obs, start)
        except Exception:
            self.branch = "fault"
            self.move_ms = ms_since(start)
            return pass_action()

    def _act(self, obs, start: float):
        if self._first_move:
            deadline_end = start + self.params.FIRST_MOVE_GRACE_MS / 1000.0
        else:
            deadline_end = start + self.params.LATENCY_CAP_MS / 1000.0
        hard = Deadline.until(deadline_end)

        self.state.update(obs)
        if self._first_move:
            self.state.precompute(hard)
            self._first_move = False
        elif obs.turn % 25 == 0 and not hard.expired():
            self.state.refresh_pockets_light()

        self._chain_head_before = self.state.chain_head
        self.phase = self.state.phase
        self.clock_phase = self.state.clock_phase
        self.chain_head = self.state.chain_head
        self.candidate_count = len(self.state.memory.candidates)
        self.top_section = self.state.sections.top_section()
        self.pocket_skips = self.state.pocket_skip_count
        self.enemy_general_sighted = 1 if self.state.memory.enemy_general else 0
        self.castles_built_probe = count_owned_castles(obs)
        self.state.castles_owned = self.castles_built_probe
        if (
            self.castles_built_probe > self._prev_castles
            and self.first_castle_turn < 0
        ):
            self.first_castle_turn = int(obs.turn)
        self._prev_castles = self.castles_built_probe
        self.land_at_50 = self.state.land_at_50
        self.recall_fired = self.state.recall_fired
        eg = self.state.memory.enemy_general
        self.enemy_gen = _cell_tok(eg)

        # Pre-contact: MapMemory hunt. Contact: ContactMCTS owns the probe.
        if self.state.memory.enemy_general is None and self.state.phase == "search":
            self.state.memory.hunt_target(obs, self.params)
            self.state.objective = self.state.memory.hunt_cell or self.state.objective

        # Keep a hunt-facing tip so contact gather/wave share one stack.
        goal_for_tip = self.state.memory.enemy_general or self.state.objective
        if goal_for_tip is None and self.state.phase == "search":
            goal_for_tip = self.state.memory.hunt_cell
        # Provisional tip before contact prepare (uses prior commitment/fog objective).
        if goal_for_tip is not None and self.state.phase in ("search", "contact"):
            tip = select_mass_tip(
                obs,
                goal_for_tip,
                self.params,
                self.state.strike_tip,
                self.state.strike_tip_turn,
            )
            if tip is not None:
                self.state.strike_tip = tip
                self.state.strike_tip_turn = obs.turn
                self.state.muster = tip

        if self.state.phase == "contact" and self.state.memory.enemy_general is None:
            prep_ms = min(
                hard.remaining_ms(), float(self.params.CONTACT_PREP_BUDGET_MS)
            )
            self.contact.prepare_contact(obs, self.state, Deadline(prep_ms))
            commit = self.state.contact_commitment
            if commit is not None:
                self.state.objective = commit.macro.waypoint
                self.contact_waypoint = commit.macro.waypoint
                self.contact_macro = commit.macro.kind
                self.contact_commit_turn = commit.committed_turn
                self.contact_switches = commit.switch_count
                self.contact_macro_score = float(commit.last_score)
                self.contact_candidate_mass = float(
                    self.contact.stats.candidate_mass
                )
                # Re-aim tip at the committed probe waypoint.
                tip = select_mass_tip(
                    obs,
                    commit.macro.waypoint,
                    self.params,
                    self.state.strike_tip,
                    self.state.strike_tip_turn,
                )
                if tip is not None:
                    self.state.strike_tip = tip
                    self.state.strike_tip_turn = obs.turn
                    self.state.muster = tip

        self.hunt = (
            self.state.objective
            if self.state.phase == "contact"
            else self.state.memory.hunt_cell
        )
        self.muster = self.state.muster
        self.objective = self.state.objective
        self._clear_mcts_diag()

        # 1. Lethal kill
        kill = self._kill_shot(obs)
        if kill is not None:
            return self._finish(obs, start, kill, searched=False, branch="kill")

        # 2. Pass only when no leave-1 move
        if not has_leave1_move(obs):
            return self._finish(
                obs, start, pass_action(), searched=False, branch="pass"
            )

        # 3. Rare recall (includes imminent adjacent loss as subset first)
        defense = imminent_loss_move(obs, self.state)
        if defense is not None:
            self.state.recall_fired += 1
            return self._finish(
                obs, start, defense, searched=False, branch="recall"
            )
        recall = recall_move(obs, self.state, self.params)
        if recall is not None:
            self.state.recall_fired += 1
            return self._finish(
                obs, start, recall, searched=False, branch="recall"
            )

        # 4. Opening t <= 50
        if obs.turn <= self.params.OPEN_END:
            rem = min(hard.remaining_ms(), float(self.params.WAVE_BUDGET_MS))
            opening = decide_opening(
                obs, self.state, self.search, Deadline(rem), self.params
            )
            if opening is not None:
                self.search_iters = int(self.search.stats.iterations)
                self._pull_mcts_diag(self.search.stats)
                return self._finish(
                    obs, start, opening, searched=True, branch="opening"
                )

        # 5. Castle economy (never before CASTLE_START_TURN=116)
        castle_move = self.economy.decide(obs, self.state, hard)
        if castle_move is not None:
            return self._finish(
                obs, start, castle_move, searched=False, branch="castle"
            )

        # 6. Post-sight tip floor then march
        if self.state.phase == "strike":
            feed = self._tip_feed_wave(obs)
            if feed is not None:
                return self._finish(
                    obs, start, feed, searched=False, branch="tip_feed"
                )

        # 7. Mod-50 clock + info-phase MCTS
        clock = mod50_phase(obs.turn, self.params)
        self.state.clock_phase = clock
        self.clock_phase = clock
        remaining = hard.remaining_ms()
        if clock == "gather":
            budget = min(remaining, float(self.params.GATHER_BUDGET_MS))
        else:
            budget = min(remaining, float(self.params.WAVE_BUDGET_MS))

        # Gather that would haul army across the map converts to land instead.
        if clock == "gather" and self.state.phase != "strike":
            take = far_haul_capture(obs, self.state, self.params, self.state.muster)
            if take is not None:
                return self._finish(
                    obs, start, take, searched=False, branch="expand"
                )

        if self.state.phase == "strike":
            budget = min(budget, float(self.params.STRIKE_MARCH_BUDGET_MS))
            move = self.strike.search(obs, self.state, Deadline(budget))
            self.search_iters = self.strike.stats.iterations
            self._pull_mcts_diag(self.strike.stats)
        elif self.state.phase == "contact":
            # Not expanding here on purpose. Taking neutrals instead of pushing
            # after contact grew our land to Kubic's 0.24 share and still lost
            # more: macaria's peak land went 0.36 -> 0.51 once the pressure came
            # off. Contesting their ground is what holds them down.
            budget = min(budget, float(self.params.CONTACT_BUDGET_MS))
            if self.params.CONTACT_PATH_MODE == "macro_mcts":
                score_cap = budget
            else:
                score_cap = min(budget, float(self.params.CONTACT_PATH_SCORE_BUDGET_MS))
            move = self.contact.search(obs, self.state, Deadline(score_cap))
            self.search_iters = self.contact.stats.iterations
            self._pull_mcts_diag(self.contact.stats)
            st = self.contact.stats
            if st.waypoint is not None:
                self.contact_waypoint = st.waypoint
            if st.macro_kind != "none":
                self.contact_macro = st.macro_kind
            self.contact_switches = st.switch_count
            self.contact_candidate_mass = float(st.candidate_mass)
        else:
            # Pre-contact wave: take land. Kubic captures ~0.5 neutrals a tick
            # before contact — one per unit the general makes — and land is the
            # compounding resource (RULES §04: every cell grows every 50).
            # Marching a fog-frontier objective instead spent the wave walking
            # over ground we already owned.
            if clock == "wave":
                take = expand_move(obs, self.state, self.params)
                if take is not None:
                    return self._finish(
                        obs, start, take, searched=False, branch="expand"
                    )
            budget = min(budget, float(self.params.SEARCH_BUDGET_MS))
            move = self.search.search(obs, self.state, Deadline(budget))
            self.search_iters = self.search.stats.iterations
            self._pull_mcts_diag(self.search.stats)

        if move is None:
            move = pass_action()
        return self._finish(obs, start, move, searched=True, branch="mcts")

    def _finish(self, obs, start: float, move, searched: bool, branch: str):
        self.branch = branch
        self.searched = searched
        if not searched:
            self.search_iters = 0

        head_before = self._chain_head_before
        goal = self.state.objective or self.state.memory.enemy_general
        self.objective = goal

        pair = action_src_dst(move)
        if pair is None:
            self.toward = 0
            self.chain_continued = 0
        else:
            src, dst = pair
            self.chain_continued = (
                1 if head_before is not None and src == head_before else 0
            )
            if goal is None:
                self.toward = 0
            else:
                before = abs(src[0] - goal[0]) + abs(src[1] - goal[1])
                after = abs(dst[0] - goal[0]) + abs(dst[1] - goal[1])
                self.toward = 1 if after < before else 0

        tip = self._resolve_tip(obs, goal)
        self.tip = tip
        if tip is None:
            self.tip_army = 0
            self.tip_dist_goal = -1
            self.tip_ready = 0
        else:
            self.tip_army = int(obs.army_grid[tip[0]][tip[1]])
            if goal is None:
                self.tip_dist_goal = -1
                self.tip_ready = 0
            else:
                self.tip_dist_goal = abs(tip[0] - goal[0]) + abs(tip[1] - goal[1])
                self.tip_ready = 1 if tip_is_ready(obs, tip, goal, self.params) else 0

        if self.state.phase == "contact" and self.state.objective is not None:
            self.hunt = self.state.objective
        else:
            self.hunt = self.state.memory.hunt_cell
        self.muster = self.state.muster or tip

        update_chain(self.state, move)
        self.chain_head = self.state.chain_head
        self.recall_fired = self.state.recall_fired
        self.move_ms = ms_since(start)
        return move

    def _resolve_tip(self, obs, goal):
        tip = self.state.strike_tip
        if tip is not None:
            r, c = tip
            if obs.owner_grid[r][c] == 1 and obs.army_grid[r][c] > 1:
                return tip
        if goal is not None:
            tip = select_mass_tip(
                obs,
                goal,
                self.params,
                self.state.strike_tip,
                self.state.strike_tip_turn,
            )
            if tip is not None:
                return tip
        return largest_owned_stack(obs)

    def _tip_feed_wave(self, obs):
        """Exclusive gather onto tip while below TIP_AT_SIGHT_FLOOR."""
        goal = self.state.memory.enemy_general
        if goal is None:
            return None
        tip = select_mass_tip(
            obs,
            goal,
            self.params,
            self.state.strike_tip,
            self.state.strike_tip_turn,
        )
        if tip is None:
            return None
        self.state.strike_tip = tip
        self.state.strike_tip_turn = obs.turn
        self.state.muster = tip
        if not tip_below_sight_floor(obs, tip, self.params):
            return None
        return feed_tip_action(obs, tip)

    def _kill_shot(self, obs):
        """Capture a visible or remembered enemy general if any stack can."""
        from params import T_GENERAL

        goals = []
        remembered = self.state.memory.enemy_general
        if remembered is not None:
            goals.append(remembered)
        for r in range(obs.H):
            for c in range(obs.W):
                if obs.owner_grid[r][c] != 2:
                    continue
                if obs.type_grid[r][c] != T_GENERAL:
                    continue
                cell = (r, c)
                if cell not in goals:
                    goals.append(cell)
                    # Latch immediately so strike phase engages next lines.
                    self.state.memory.enemy_general = cell
                    self.state.phase = "strike"

        for goal in goals:
            gr, gc = goal
            if obs.turn >= self.params.DEATHTOUCH_TURN:
                # RULES §07: one attacking unit wins; leave-1 needs army >= 2.
                need = 1
            elif obs.owner_grid[gr][gc] == 2:
                need = obs.army_grid[gr][gc] + self.params.FINISH_MARGIN
            else:
                need = self.params.FINISH_MARGIN
            for r, c in neighbors(obs.H, obs.W, gr, gc):
                if obs.owner_grid[r][c] != 1:
                    continue
                if obs.army_grid[r][c] - 1 >= need:
                    return move_action(r, c, gr, gc, 0)
        return None
