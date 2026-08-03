"""
Morpheus protocol shell — pass-only fixture.

Retires startup, frame parsing, reply serialization, and EOF risk before any
model, search, or belief lands. Later parts replace this body.
"""

# Protocol-safe no-op. Always legal under the competition action schema.
PASS = (1, 0, 0, 0, 0)


class Agent:
    """Pass on every observation. No model, search, or telemetry."""

    def __init__(self, player_id, H, W):
        self.player_id = player_id
        self.H = H
        self.W = W

    def act(self, obs):
        return PASS
