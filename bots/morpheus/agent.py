"""
Morpheus agent — runtime controller with deadline degradation.

Part 01 kept a pass-only shell. Part 07 owns the turn work order through
``RuntimeController``. The frozen export still loads on startup.
"""

from inference import load_default_session, warm_export_batches
from runtime import PASS, RuntimeConfig, RuntimeController
from search import UniformEvaluator


class Agent:
    """Deadline-aware Morpheus seat. Probe reads public counter attributes."""

    def __init__(self, player_id, H, W):
        self.player_id = player_id
        self.H = H
        self.W = W
        self._session = load_default_session()
        warm_export_batches(self._session)
        # Search uses the uniform evaluator until Part 09 couples the measured
        # network leaf budget. The export session stays warm for later parts.
        self._controller = RuntimeController(
            seat=int(player_id),
            H=int(H),
            W=int(W),
            evaluator=UniformEvaluator(0.0),
            config=RuntimeConfig(),
        )
        self._mirror_metrics()

    def _mirror_metrics(self) -> None:
        c = self._controller
        self.completed_simulations = c.completed_simulations
        self.forward_equivalents = c.forward_equivalents
        self.belief_ess = c.belief_ess
        self.recovery = c.recovery
        self.tree_size = c.tree_size
        self.fallback_level = c.fallback_level
        self.move_ms = c.move_ms
        self.search_iters = c.completed_simulations
        self.cost_belief_ms = c.cost_belief_ms
        self.cost_root_ms = c.cost_root_ms
        self.cost_search_ms = c.cost_search_ms
        self.cost_reply_ms = c.cost_reply_ms

    def act(self, obs):
        action = self._controller.decide(obs)
        self._mirror_metrics()
        return action
