"""Arena bot package — ensure stdio and competition-module imports resolve."""
from __future__ import annotations

import sys
from pathlib import Path

_bots_dir = Path(__file__).resolve().parent
_cm_root = _bots_dir.parent / "competition-module"
for _path in (_bots_dir, _cm_root):
    _entry = str(_path)
    if _entry not in sys.path:
        sys.path.insert(0, _entry)
