"""Sosipolis execution: economy then Search / Contact / Strike MCTS."""
from __future__ import annotations

import time

from components.army import (
    move_action,
    neighbors,
    pass_action,
)
from components.clock import Deadline, ms_since
from components.contact_mcts import ContactMCTS
from components.economy import Economy, count_owned_castles
from components.search_mcts import SearchMCTS
from components.strike_mcts import StrikeMCTS
from params import PARAMS, Params
from state import GameState


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

    def act(self, obs):
        start = time.monotonic()
        try:
            return self._act(obs, start)
        except Exception:
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

        self.phase = self.state.phase
        self.candidate_count = len(self.state.memory.candidates)
        self.top_section = self.state.sections.top_section()
        self.pocket_skips = self.state.pocket_skip_count
        self.enemy_general_sighted = 1 if self.state.memory.enemy_general else 0
        self.castles_built_probe = count_owned_castles(obs)
        self.state.castles_owned = self.castles_built_probe

        kill = self._kill_shot(obs)
        if kill is not None:
            self.searched = False
            self.search_iters = 0
            self.move_ms = ms_since(start)
            return kill

        defense = self._defense_move(obs)
        if defense is not None:
            self.searched = False
            self.search_iters = 0
            self.move_ms = ms_since(start)
            return defense

        castle_move = self.economy.decide(obs, self.state, hard)
        if castle_move is not None:
            self.searched = False
            self.search_iters = 0
            self.move_ms = ms_since(start)
            return castle_move

        remaining = hard.remaining_ms()
        if self.state.phase == "strike":
            budget = min(remaining, float(self.params.STRIKE_BUDGET_MS))
            move = self.strike.search(obs, self.state, Deadline(budget))
            self.search_iters = self.strike.stats.iterations
        elif self.state.phase == "contact":
            budget = min(remaining, float(self.params.CONTACT_BUDGET_MS))
            move = self.contact.search(obs, self.state, Deadline(budget))
            self.search_iters = self.contact.stats.iterations
        else:
            budget = min(remaining, float(self.params.SEARCH_BUDGET_MS))
            move = self.search.search(obs, self.state, Deadline(budget))
            self.search_iters = self.search.stats.iterations

        self.searched = True
        if move is None:
            move = pass_action()

        self.move_ms = ms_since(start)
        return move

    def _kill_shot(self, obs):
        goal = self.state.memory.enemy_general
        if goal is None:
            return None
        gr, gc = goal
        if obs.turn >= self.params.DEATHTOUCH_TURN:
            need = 2
        else:
            need = obs.army_grid[gr][gc] + self.params.FINISH_MARGIN
            if obs.owner_grid[gr][gc] != 2:
                need = self.params.FINISH_MARGIN
        for r, c in neighbors(obs.H, obs.W, gr, gc):
            if obs.owner_grid[r][c] != 1:
                continue
            if obs.army_grid[r][c] - 1 >= need:
                return move_action(r, c, gr, gc, 0)
        return None

    def _defense_move(self, obs):
        home = self.state.memory.own_general
        if home is None:
            return None
        hr, hc = home
        threat = None
        threat_army = 0
        for r, c in neighbors(obs.H, obs.W, hr, hc):
            if obs.owner_grid[r][c] == 2 and obs.army_grid[r][c] > threat_army:
                threat_army = obs.army_grid[r][c]
                threat = (r, c)
        if threat is None:
            return None
        tr, tc = threat
        for r, c in neighbors(obs.H, obs.W, tr, tc):
            if obs.owner_grid[r][c] != 1:
                continue
            if obs.army_grid[r][c] - 1 > obs.army_grid[tr][tc]:
                return move_action(r, c, tr, tc, 0)
        for r, c in neighbors(obs.H, obs.W, hr, hc):
            if obs.owner_grid[r][c] == 1 and obs.army_grid[r][c] > 1:
                return move_action(r, c, hr, hc, 0)
        return None
