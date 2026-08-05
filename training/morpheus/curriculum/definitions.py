"""Immutable constants and seed derivation for curriculum items."""

from __future__ import annotations

import hashlib

# Class ids match docs/bots/morpheus/training.md (1-based).
CLASS_TACTICAL = 1
CLASS_AFTER_SIGHT = 2
CLASS_CONTACT = 3
CLASS_PRE_CONTACT = 4
CLASS_FULL_START = 5

CLASS_NAMES: dict[int, str] = {
    CLASS_TACTICAL: "tactical_capture_or_defense",
    CLASS_AFTER_SIGHT: "after_enemy_general_sight",
    CLASS_CONTACT: "contact_both_generals_alive",
    CLASS_PRE_CONTACT: "pre_contact",
    CLASS_FULL_START: "full_competition_start",
}

# Turns before a decisive terminal capture that still count as class 1.
TACTICAL_HORIZON = 8

# Source labels that must never enter a curriculum manifest.
BANNED_SOURCE_TOKENS: frozenset[str] = frozenset(
    {
        "resbot",
        "scraped_resbot",
        "leaderboard_resbot",
    }
)

SOURCE_FULL_START = "full_start"
SOURCE_FIXED_PANEL = "fixed_panel"


def is_banned_source(label: str) -> bool:
    """True when a source label is raw ResBot / scraped data (not reconstructions).

    Labels ending with ``_reconstructions`` are allowed so rebuilt trajectories
    can enter the curriculum under a measurable separate source.
    """
    normalized = str(label).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized.endswith("_reconstructions"):
        return False
    if normalized in BANNED_SOURCE_TOKENS:
        return True
    return "resbot" in normalized


def belief_rng_seed(
    *,
    engine_version: str,
    map_seed: int,
    source_label: str,
    game_id: str | None,
    seat: int,
) -> int:
    """Derive a deterministic particle RNG seed from immutable game+seat fields.

    Seed is shared across all ``prefix_len`` values of the same game so
    materialize can continue one belief history instead of restarting per item.
    """
    if seat not in (0, 1):
        raise ValueError(f"seat must be 0 or 1, got {seat}")
    payload = (
        f"{engine_version}|{int(map_seed)}|{source_label}|"
        f"{game_id or ''}|{int(seat)}"
    ).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") % (2**63)


def item_id_for(
    *,
    engine_version: str,
    map_seed: int,
    source_label: str,
    prefix_len: int,
    game_id: str | None,
    class_id: int,
) -> str:
    """Stable short id for one curriculum item."""
    payload = (
        f"{engine_version}|{int(map_seed)}|{source_label}|"
        f"{int(prefix_len)}|{game_id or ''}|{int(class_id)}"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]
