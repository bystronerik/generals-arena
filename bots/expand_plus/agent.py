"""
expand_plus — frontier-directed expansion with win checks and scouting.

Greedy capture using smallest-sufficient-source scoring, general reserve,
BFS march toward the frontier, and candidate probing when idle.

See docs/bots/expand-plus.md and docs/research/strategies/optimize-existing.md.
"""
from strategy_common import (
    PASS,
    BeliefState,
    best_capture_move,
    chase_defence,
    enemy_adjacent_to_general,
    locate_own_general,
    march_toward_frontier,
    probe_move,
    win_check_w1,
    win_check_w2,
)


class Agent:
    """Competent expander: reserve, win checks, frontier march, probe."""

    def __init__(self, player_id, H, W):
        self.player_id = player_id
        self.H = H
        self.W = W
        self.general_pos = None
        self.belief = BeliefState()

    def act(self, obs):
        if self.general_pos is None:
            self.general_pos = locate_own_general(obs)
        self.belief.update(obs, self.general_pos)

        chase = chase_defence(obs, self.general_pos)
        if chase is not None:
            return chase

        w1 = win_check_w1(obs, self.belief.enemy_general, self.general_pos)
        if w1 is not None:
            return w1

        if enemy_adjacent_to_general(obs, self.general_pos):
            pass
        else:
            w2 = win_check_w2(obs, self.belief.enemy_general, self.general_pos)
            if w2 is not None:
                return w2

        capture = best_capture_move(obs, self.general_pos)
        if capture is not None:
            return capture

        march = march_toward_frontier(obs, self.general_pos)
        if march is not None:
            return march

        probe = probe_move(obs, self.general_pos, self.belief.candidates or set())
        if probe is not None:
            return probe

        return PASS

    def telemetry_extras(self):
        if self.belief.first_sighting_turn is None:
            return {"enemy_general_sighted": 0}
        return {
            "enemy_general_sighted": 1,
            "first_sighting_turn": self.belief.first_sighting_turn,
        }
