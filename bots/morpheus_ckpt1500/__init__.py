"""Morpheus bot closure — shared observation, network, and inference modules."""
from __future__ import annotations

import sys
from pathlib import Path

_BOT = Path(__file__).resolve().parent
_s = str(_BOT)
if _s not in sys.path:
    sys.path.insert(0, _s)
