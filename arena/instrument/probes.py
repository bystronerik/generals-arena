"""
Load and call a bot's per-turn probe.

A probe is `bots/<name>/probe.py` exposing `extras(agent) -> dict`: a passive
read of agent attributes, sampled once per turn on recorded matches. Bots
without one are fine — they get engine-side trajectories and no bot trace.

Loaded by file path under a private module name, never by import: `probe` must
stay unimportable from inside a bot's closure (`fingerprint` raises if a
closure module imports it), and loading it here must not put it anywhere an
agent could reach it.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

from arena.records.fingerprint import PROBE_FILENAME


class ProbeError(RuntimeError):
    """A probe exists but could not be loaded or called."""


def probe_path(bot_dir: Path) -> Path:
    return bot_dir / PROBE_FILENAME


def load_probe(bot_dir: Path) -> ModuleType | None:
    """Import `bots/<name>/probe.py`, or None when the bot has no probe."""
    path = probe_path(bot_dir)
    if not path.is_file():
        return None

    # A name no `import` statement can reach, so nothing inside a bot can pick
    # this module up by accident, and two probes never collide.
    spec = importlib.util.spec_from_file_location(
        f"_arena_probe_{bot_dir.name}", path
    )
    if spec is None or spec.loader is None:
        raise ProbeError(f"cannot load probe at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "extras"):
        raise ProbeError(f"{path} defines no extras(agent) -> dict")
    return module


def probe_extras(probe: ModuleType | None, agent) -> dict:
    """
    Sample the agent through its probe. `{}` when the bot has no probe.

    Errors are **not** swallowed. A probe that reads an attribute an agent
    refactor renamed should abort the recorded match, not quietly log a
    partial trace that looks like evidence.
    """
    if probe is None:
        return {}
    extras = probe.extras(agent)
    if not isinstance(extras, dict):
        raise ProbeError(
            f"{probe.__file__} returned {type(extras).__name__}, expected dict"
        )
    return extras
