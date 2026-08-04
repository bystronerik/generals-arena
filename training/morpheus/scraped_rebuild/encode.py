"""Map reconstructed actions to competition Action5 tuples."""

from __future__ import annotations

from typing import Sequence

# Matches bots/morpheus/transition.py and the competition protocol.
PASS_ACTION5: tuple[int, int, int, int, int] = (1, 0, 0, 0, 0)
BUILD = 2
DIRECTIONS: tuple[tuple[int, int], ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))


def direction_index(src: Sequence[int], dst: Sequence[int]) -> int:
    """Return the 0–3 direction code from ``src`` to ``dst``."""
    dr = int(dst[0]) - int(src[0])
    dc = int(dst[1]) - int(src[1])
    for i, (edr, edc) in enumerate(DIRECTIONS):
        if dr == edr and dc == edc:
            return i
    raise ValueError(f"not an orthogonal step: {tuple(src)} -> {tuple(dst)}")


def action_dict_to_action5(act: dict) -> tuple[int, int, int, int, int]:
    """Convert one inferred action dict to Action5."""
    kind = str(act.get("kind", "pass"))
    if kind in ("pass", "unresolved"):
        return PASS_ACTION5
    if kind == "build":
        cell = act["cell"]
        return (BUILD, int(cell[0]), int(cell[1]), 0, 0)
    if kind == "move":
        src = act["src"]
        dst = act["dst"]
        d = direction_index(src, dst)
        split = int(act.get("split", 0))
        return (0, int(src[0]), int(src[1]), d, split)
    raise ValueError(f"unknown action kind {kind!r}")


def tick_to_joint_action5(tick: dict) -> tuple[
    tuple[int, int, int, int, int],
    tuple[int, int, int, int, int],
]:
    """Return ``(action_a, action_b)`` for one inferred tick."""
    return (
        action_dict_to_action5(tick["p0"]),
        action_dict_to_action5(tick["p1"]),
    )
