"""
Morpheus agent — deployment-configured runtime with network leaf budget.

Part 09 couples the measured deployment.json fields, NetworkEvaluator, and
belief-proposal policy. Probe counters stay passive.
"""

from deployment import try_load_deployment
from evaluator import NetworkEvaluator
from inference import load_default_session
from runtime import RuntimeController


class Agent:
    """Deadline-aware Morpheus seat. Probe reads public counter attributes."""

    def __init__(self, player_id, H, W):
        self.player_id = player_id
        self.H = H
        self.W = W
        self._deployment = try_load_deployment()
        self._session = load_default_session()
        shapes = tuple(self._deployment.warmup_batch_shapes) or (1, 4, 64)
        # warm_export_batches uses fixed (1,4,64); exercise configured shapes.
        import torch
        from network import IN_CHANNELS, BOARD

        gen = torch.Generator(device="cpu")
        gen.manual_seed(0)
        for batch in shapes:
            x = torch.randn(int(batch), IN_CHANNELS, BOARD, BOARD, generator=gen)
            self._session.forward_policy(x)
            self._session.forward_policy_wdl(x)
        self._evaluator = NetworkEvaluator(self._session)
        runtime_cfg = self._deployment.to_runtime_config()
        runtime_cfg.max_proposal_batch = int(self._deployment.max_proposal_batch)
        # proposal_policy=None: belief particles advance on uniform legal
        # enemy actions, not the policy net. Measured vs macaria (100 games
        # per arm, docs/research/measurements/belief-ablation-macaria.md):
        # the learned proposal showed no benefit (-0.07 +/- 0.13 paired),
        # and skipping its forward returns ~13 ms/turn to the search budget.
        self._controller = RuntimeController(
            seat=int(player_id),
            H=int(H),
            W=int(W),
            evaluator=self._evaluator,
            config=runtime_cfg,
            proposal_policy=None,
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
        self.belief_plus_root_ok = c.belief_plus_root_ok
        # Expose named component timings for the instrumented unbundled path.
        self.component_ms = dict(c.metrics.component_ms)
        self.component_calls = dict(c.component_calls)
        self.forward_by_consumer = dict(c.forward_by_consumer)
        self.proposal_n_particles = c.proposal_n_particles
        self.proposal_n_singleton_particles = c.proposal_n_singleton_particles
        self.proposal_n_unique_info_keys = c.proposal_n_unique_info_keys
        self.proposal_n_unique_policy_inputs = c.proposal_n_unique_policy_inputs
        self.proposal_n_policy_batches = c.proposal_n_policy_batches
        self.root_pass_prior_milli = c.root_pass_prior_milli
        self.root_top_action = c.root_top_action
        self.root_top_prior_milli = c.root_top_prior_milli
        self.chosen_action = c.chosen_action
        self.chosen_is_pass = c.chosen_is_pass
        self.policy_fallback_is_pass = c.policy_fallback_is_pass
        self.root_legal_nonpass = c.root_legal_nonpass
        self.has_root_result = c.has_root_result

    def act(self, obs):
        action = self._controller.decide(obs)
        self._mirror_metrics()
        return action
