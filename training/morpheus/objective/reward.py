"""Terminal WDL reward only. No land, army, castle, sight, or length shaping."""

from __future__ import annotations

from typing import Any, Mapping

# Canonical terminal reward (discount = 1). Nonterminal contribution is 0.
WIN = 1.0
DRAW = 0.0
LOSS = -1.0
DISCOUNT = 1.0

# Names that must never appear as reward terms.
FORBIDDEN_REWARD_TERMS = frozenset(
    {
        "land",
        "army",
        "castles",
        "castle",
        "sight",
        "game_length",
        "length",
        "turns",
        "shaping",
    }
)


def terminal_reward(winner: str, *, seat: int) -> float:
    """Seat-perspective terminal reward: win=+1, draw=0, loss=-1."""
    if seat not in (0, 1):
        raise ValueError(f"seat must be 0 or 1, got {seat}")
    if winner == "draw":
        return DRAW
    if winner == "a":
        return WIN if seat == 0 else LOSS
    if winner == "b":
        return WIN if seat == 1 else LOSS
    raise ValueError(f"unknown winner {winner!r}")


def wdl_one_hot(winner: str, *, seat: int) -> tuple[float, float, float]:
    """Perspective WDL class probabilities: (win, draw, loss)."""
    r = terminal_reward(winner, seat=seat)
    if r > 0.0:
        return (1.0, 0.0, 0.0)
    if r < 0.0:
        return (0.0, 0.0, 1.0)
    return (0.0, 1.0, 0.0)


def assert_no_shaping(reward_spec: Mapping[str, Any] | None = None) -> None:
    """Fail closed if a reward dict names a rejected shaping term."""
    if reward_spec is None:
        return
    for key in reward_spec:
        name = str(key).lower()
        if name in FORBIDDEN_REWARD_TERMS:
            raise ValueError(
                f"reward must not include shaping term {key!r}; "
                "only terminal WDL is allowed"
            )
        if name not in {"win", "draw", "loss", "discount", "nonterminal"}:
            raise ValueError(
                f"unknown reward field {key!r}; allowed: "
                "win, draw, loss, discount, nonterminal"
            )


def reward_contract() -> dict[str, float]:
    """Explicit terminal-only reward contract for run manifests."""
    return {
        "win": WIN,
        "draw": DRAW,
        "loss": LOSS,
        "discount": DISCOUNT,
        "nonterminal": 0.0,
    }
