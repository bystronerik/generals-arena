"""Save and reload float Morpheus pilot checkpoints."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[3]
MORPHEUS_BOT = REPO / "bots" / "morpheus"


def _ensure_bot_path() -> None:
    for entry in (REPO, REPO / "bots", MORPHEUS_BOT):
        s = str(entry)
        if s not in sys.path:
            sys.path.insert(0, s)


def save_pilot_checkpoint(
    model: torch.nn.Module,
    directory: Path,
    *,
    meta: dict[str, Any] | None = None,
) -> Path:
    """Write ``state_dict.pt`` + ``meta.json`` reloadable by export helpers."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload = dict(meta or {})
    payload.setdefault("n_blocks", len(getattr(model, "blocks", []) or []) or 12)
    payload.setdefault("seed", 0)
    (directory / "meta.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    torch.save(model.state_dict(), directory / "state_dict.pt")
    return directory


def load_pilot_checkpoint(directory: Path):
    """Load a MorpheusNet from a pilot checkpoint directory."""
    _ensure_bot_path()
    from export import build_model_from_checkpoint, load_checkpoint_state

    checkpoint = load_checkpoint_state(Path(directory))
    return build_model_from_checkpoint(checkpoint), checkpoint.get("meta") or {}
