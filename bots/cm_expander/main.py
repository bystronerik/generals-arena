"""Competition stdio entrypoint — upstream benchmark agent via cm_adapter."""
import sys
from pathlib import Path

_BOTS = Path(__file__).resolve().parent.parent
if str(_BOTS) not in sys.path:
    sys.path.insert(0, str(_BOTS))

from _common.wire import run_stdio
from agent import Agent

if __name__ == "__main__":
    run_stdio(Agent)
