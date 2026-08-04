"""Source labels for reconstructed scraped replays."""

from __future__ import annotations

RECONSTRUCTIONS_SUFFIX = "_reconstructions"


def source_label_for(player: str) -> str:
    """Return ``<player>_reconstructions`` for curriculum indexing."""
    name = str(player).strip()
    if not name:
        raise ValueError("player name must be non-empty")
    if name.endswith(RECONSTRUCTIONS_SUFFIX):
        return name
    return f"{name}{RECONSTRUCTIONS_SUFFIX}"


def is_reconstruction_source(label: str) -> bool:
    """True when a source label is an allowed reconstruction tag."""
    normalized = str(label).strip().lower().replace("-", "_").replace(" ", "_")
    return normalized.endswith(RECONSTRUCTIONS_SUFFIX)
