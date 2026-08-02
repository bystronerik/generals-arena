"""Competition stdio entrypoint for sosipolis."""
from __future__ import annotations

import sys
from pathlib import Path

_DIR = Path(__file__).resolve().parent
if str(_DIR) not in sys.path:
    sys.path.insert(0, str(_DIR))

from brain import Agent
from stdio import run_stdio

if __name__ == "__main__":
    run_stdio(Agent)
