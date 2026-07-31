"""
general_hunter — scout, snipe, and deathtouch beeline bot.

Maintains an enemy-general candidate prior, probes with split detachments,
expands with smallest-sufficient-source scoring, and executes win checks and
chase defence. Starts final approach at APPROACH_START (450) once the enemy
general is known.

See docs/bots/general-hunter.md and docs/research/strategies/optimize-existing.md.
"""
from _common.strategy_common import (
    PASS,
    DIRECTIONS,
    StrategyContext,
    enemy_adjacent_to_general,
    locate_own_general,
)

RESERVE_OPENING_END = 60
DEFEND_FROM = 500
SENTRY_FROM = 450
DEATHTOUCH_TURN = 800
APPROACH_START = 450
MIN_GENERAL_DISTANCE = 17

_STRATEGY = StrategyContext(
    reserve_opening_end=RESERVE_OPENING_END,
    defend_from=DEFEND_FROM,
    sentry_from=SENTRY_FROM,
    deathtouch_turn=DEATHTOUCH_TURN,
    min_general_distance=MIN_GENERAL_DISTANCE,
)


class Agent:
    """Scout-driven decisiveness bot with reserve and chase defence."""

    def __init__(self, player_id, H, W):
        self.player_id = player_id
        self.H = H
        self.W = W
        self.general_pos = None
        self.belief = _STRATEGY.BeliefState()
        self.active_probes = set()

    def act(self, obs):
        if self.general_pos is None:
            self.general_pos = locate_own_general(obs)
        self.belief.update(obs, self.general_pos)
        self._update_active_probes(obs)

        chase = _STRATEGY.chase_defence(obs, self.general_pos)
        if chase is not None:
            return self._finish_move(chase)

        w1 = _STRATEGY.win_check_w1(obs, self.belief.enemy_general, self.general_pos)
        if w1 is not None:
            return self._finish_move(w1)

        if not enemy_adjacent_to_general(obs, self.general_pos):
            w2 = _STRATEGY.win_check_w2(obs, self.belief.enemy_general, self.general_pos)
            if w2 is not None:
                return self._finish_move(w2)

        if self.belief.enemy_general is not None:
            eta = _STRATEGY.eta_to_general(obs, self.belief.enemy_general)
            eta_deadline = DEATHTOUCH_TURN - eta - 5 if eta is not None else DEATHTOUCH_TURN
            if obs.turn >= APPROACH_START or obs.turn >= eta_deadline:
                approach = _STRATEGY.final_approach_move(
                    obs, self.belief.enemy_general, self.general_pos
                )
                if approach is not None:
                    return self._finish_move(approach)

        capture = _STRATEGY.best_capture_move(obs, self.general_pos)
        if capture is not None:
            return self._finish_move(capture)

        sentry = _STRATEGY.sentry_convey(obs, self.general_pos)
        if sentry is not None:
            return self._finish_move(sentry)

        march = _STRATEGY.march_toward_frontier(obs, self.general_pos)
        if march is not None:
            return self._finish_move(march)

        candidates = self.belief.candidates or set()
        probe = _STRATEGY.probe_move(
            obs, self.general_pos, candidates, active_probes=self.active_probes
        )
        if probe is not None:
            return self._finish_move(probe)

        return self._finish_move(PASS)

    def _finish_move(self, move):
        self._advance_probe_tracking(move)
        return move

    def _update_active_probes(self, obs):
        """Keep dispatch-tracked probe cells that remain owned."""
        still_active = set()
        for r, c in self.active_probes:
            if obs.owner_grid[r][c] == 1:
                still_active.add((r, c))
        self.active_probes = still_active

    def _advance_probe_tracking(self, move):
        """Track probe cells across split dispatches and full-stack steps."""
        pass_flag, r, c, d, split = move
        if pass_flag != 0:
            return
        dr, dc = DIRECTIONS[d]
        dest = (r + dr, c + dc)
        src = (r, c)
        if src in self.active_probes:
            self.active_probes.discard(src)
            if split == 0:
                self.active_probes.add(dest)
        elif split == 1:
            self.active_probes.add(dest)
