"""
expand_plus — frontier-directed expansion with win checks and scouting.

Greedy capture using smallest-sufficient-source scoring, general reserve,
BFS march toward the frontier, and candidate probing when idle.

See docs/bots/expand-plus.md and docs/research/strategies/optimize-existing.md.
"""
from _common.strategy_common import (
    PASS,
    StrategyContext,
    enemy_adjacent_to_general,
    locate_own_general,
)

RESERVE_OPENING_END = 60
DEFEND_FROM = 780
SENTRY_FROM = 700
DEATHTOUCH_TURN = 800
MIN_GENERAL_DISTANCE = 17

_STRATEGY = StrategyContext(
    reserve_opening_end=RESERVE_OPENING_END,
    reserve_max=None,
    defend_from=DEFEND_FROM,
    sentry_from=SENTRY_FROM,
    deathtouch_turn=DEATHTOUCH_TURN,
    min_general_distance=MIN_GENERAL_DISTANCE,
    enemy_dest_value=6,
)


class Agent:
    """Competent expander: reserve, win checks, frontier march, probe."""

    def __init__(self, player_id, H, W):
        self.player_id = player_id
        self.H = H
        self.W = W
        self.general_pos = None
        self.belief = _STRATEGY.BeliefState()

    def act(self, obs):
        if self.general_pos is None:
            self.general_pos = locate_own_general(obs)
        self.belief.update(obs, self.general_pos)

        chase = _STRATEGY.chase_defence(obs, self.general_pos)
        if chase is not None:
            return chase

        w1 = _STRATEGY.win_check_w1(obs, self.belief.enemy_general, self.general_pos)
        if w1 is not None:
            return w1

        if enemy_adjacent_to_general(obs, self.general_pos):
            pass
        else:
            w2 = _STRATEGY.win_check_w2(obs, self.belief.enemy_general, self.general_pos)
            if w2 is not None:
                return w2

        capture = _STRATEGY.best_capture_move(obs, self.general_pos)
        if capture is not None:
            return capture

        march = _STRATEGY.march_toward_frontier(obs, self.general_pos)
        if march is not None:
            return march

        probe = _STRATEGY.probe_move(obs, self.general_pos, self.belief.candidates or set())
        if probe is not None:
            return probe

        return PASS
