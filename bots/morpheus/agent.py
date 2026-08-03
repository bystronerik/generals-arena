"""
Morpheus protocol shell — pass-only fixture with artifact load validation.

Later parts replace the pass-only body with search and belief. Part 04 loads and
warms the frozen static 8-bit export on startup.
"""

from inference import load_default_session, warm_export_batches

# Protocol-safe no-op. Always legal under the competition action schema.
PASS = (1, 0, 0, 0, 0)


class Agent:
    """Pass on every observation. Loads the versioned export artifact on startup."""

    def __init__(self, player_id, H, W):
        self.player_id = player_id
        self.H = H
        self.W = W
        self._session = load_default_session()
        warm_export_batches(self._session)

    def act(self, obs):
        return PASS
