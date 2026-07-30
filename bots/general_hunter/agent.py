"""
general_hunter — scout, snipe, and deathtouch beeline bot.

Maintains an enemy-general candidate prior, probes with split detachments,
expands with smallest-sufficient-source scoring, and executes win checks and
chase defence. Starts final approach at APPROACH_START (450) once the enemy
general is known.

See docs/bots/general-hunter.md and docs/research/strategies/optimize-existing.md.
"""
from strategy_common import (
    PASS,
    APPROACH_START,
    DEATHTOUCH_TURN,
    DIRECTIONS,
    BeliefState,
    best_capture_move,
    chase_defence,
    enemy_adjacent_to_general,
    eta_to_general,
    final_approach_move,
    locate_own_general,
    march_toward_frontier,
    probe_move,
    sentry_convey,
    win_check_w1,
    win_check_w2,
)


class Agent:
    """Scout-driven decisiveness bot with reserve and chase defence."""

    def __init__(self, player_id, H, W):
        self.player_id = player_id
        self.H = H
        self.W = W
        self.general_pos = None
        self.belief = BeliefState()
        self.active_probes = set()

    def act(self, obs):
        if self.general_pos is None:
            self.general_pos = locate_own_general(obs)
        self.belief.update(obs, self.general_pos)
        self._update_active_probes(obs)

        chase = chase_defence(obs, self.general_pos)
        if chase is not None:
            return chase

        w1 = win_check_w1(obs, self.belief.enemy_general, self.general_pos)
        if w1 is not None:
            return w1

        if not enemy_adjacent_to_general(obs, self.general_pos):
            w2 = win_check_w2(obs, self.belief.enemy_general, self.general_pos)
            if w2 is not None:
                return w2

        if self.belief.enemy_general is not None:
            eta = eta_to_general(obs, self.belief.enemy_general)
            eta_deadline = DEATHTOUCH_TURN - eta - 5 if eta is not None else DEATHTOUCH_TURN
            if obs.turn >= APPROACH_START or obs.turn >= eta_deadline:
                approach = final_approach_move(
                    obs, self.belief.enemy_general, self.general_pos
                )
                if approach is not None:
                    return approach

        capture = best_capture_move(obs, self.general_pos)
        if capture is not None:
            return capture

        sentry = sentry_convey(obs, self.general_pos)
        if sentry is not None:
            return sentry

        march = march_toward_frontier(obs, self.general_pos)
        if march is not None:
            return march

        candidates = self.belief.candidates or set()
        probe = probe_move(
            obs, self.general_pos, candidates, active_probes=self.active_probes
        )
        if probe is not None:
            self._register_probe_dispatch(probe)
            return probe

        return PASS

    def _update_active_probes(self, obs):
        """Keep dispatch-tracked probe cells that remain owned."""
        still_active = set()
        for r, c in self.active_probes:
            if obs.owner_grid[r][c] == 1:
                still_active.add((r, c))
        self.active_probes = still_active

    def _register_probe_dispatch(self, move):
        """Record the destination of a split=1 probe dispatch."""
        _, r, c, d, split = move
        if split != 1:
            return
        dr, dc = DIRECTIONS[d]
        self.active_probes.add((r + dr, c + dc))
