"""Put the bot root and this tests dir on sys.path for sibling imports."""
from __future__ import annotations

import sys
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
_BOT_DIR = _TESTS_DIR.parent

for _path in (_BOT_DIR, _TESTS_DIR):
    _s = str(_path)
    if _s not in sys.path:
        sys.path.insert(0, _s)
