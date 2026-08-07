"""Put the bot root on sys.path for sibling imports."""
from __future__ import annotations

import sys
from pathlib import Path

_BOT_DIR = Path(__file__).resolve().parent.parent
_s = str(_BOT_DIR)
if _s not in sys.path:
    sys.path.insert(0, _s)
